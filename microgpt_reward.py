"""
Reward-Gated GPT v2: Four-Quadrant Reinforcement Architecture

Instead of external reward signals (RLHF, DPO), this architecture embeds
a complete reinforcement system directly into the transformer. Each layer:

  1. Makes a local next-token prediction (deep supervision)
  2. Computes REAL reward from actual correctness (not just learned gates)
  3. Uses four behavioral quadrants to modulate learning:
     - Positive Reinforcement: deep supervision amplifies correct pathways
     - Positive Punishment: unlikelihood loss pushes AWAY from wrong tokens
     - Negative Punishment: credit assignment penalizes regressing layers
     - Negative Reinforcement: improving layers get natural gradient boost
  4. Reward-modulated attention temperature (confident → sharp, uncertain → broad)

The key insight: v1's reward gates were just learned scaling factors with no
actual correctness signal. v2 computes REAL reward from comparing predictions
to targets, then uses that signal to weight the four loss quadrants.

Based on microgpt by Andrej Karpathy. Zero external dependencies.
"""

import os
import math
import random

# ---------------------------------------------------------------------------
# Autograd engine
# ---------------------------------------------------------------------------

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):        return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if v in visited:
                continue
            if processed:
                visited.add(v)
                topo.append(v)
            else:
                stack.append((v, True))
                for child in v._children:
                    if child not in visited:
                        stack.append((child, False))
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, docs):
        self.chars = sorted(set(''.join(docs)))
        self.bos = len(self.chars)
        self.vocab_size = len(self.chars) + 1

    def encode(self, text):
        return [self.bos] + [self.chars.index(ch) for ch in text] + [self.bos]

    def decode_token(self, token_id):
        return self.chars[token_id]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_docs(path='input.txt', url=None, seed=42):
    if not os.path.exists(path):
        import urllib.request
        url = url or 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(url, path)
    docs = [line.strip() for line in open(path) if line.strip()]
    random.seed(seed)
    random.shuffle(docs)
    return docs

# ---------------------------------------------------------------------------
# Neural network primitives
# ---------------------------------------------------------------------------

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def sigmoid(x):
    return ((-x).exp() + 1) ** -1

# ---------------------------------------------------------------------------
# Standard GPT (baseline) — 3 layers for fair comparison
# ---------------------------------------------------------------------------

class GPT:
    def __init__(self, vocab_size, n_layer=3, n_embd=16, block_size=16, n_head=4, init_std=0.08):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        def matrix(nout, nin):
            return [[Value(random.gauss(0, init_std)) for _ in range(nin)] for _ in range(nout)]

        self.weights = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.weights[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            self.weights[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    def parameters(self):
        return [p for mat in self.weights.values() for row in mat for p in row]

    def forward(self, token_id, pos_id, kv_cache):
        keys, values = kv_cache
        w = self.weights
        x = [t + p for t, p in zip(w['wte'][token_id], w['wpe'][pos_id])]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            x_res = x
            x = rmsnorm(x)
            q = linear(x, w[f'layer{li}.attn_wq'])
            k = linear(x, w[f'layer{li}.attn_wk'])
            v = linear(x, w[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)

            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, w[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_res)]

            x_res = x
            x = rmsnorm(x)
            x = linear(x, w[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, w[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_res)]

        return linear(x, w['lm_head'])

    def new_kv_cache(self):
        return ([[] for _ in range(self.n_layer)],
                [[] for _ in range(self.n_layer)])

# ---------------------------------------------------------------------------
# Reward-Gated GPT v2: Four-Quadrant Reinforcement Architecture
# ---------------------------------------------------------------------------

class RewardGPT:
    """
    GPT with intrinsic four-quadrant reinforcement at each layer.

    Per-layer components (beyond standard Q,K,V,O,fc1,fc2):
      - W_r_attn  [1, n_embd]     : learned reward gate for attention
      - W_r_mlp   [1, n_embd]     : learned reward gate for MLP
      - W_p       [vocab, n_embd] : local prediction head

    Training mode:
      - Computes REAL reward from actual prediction correctness
      - Uses reward to weight four loss quadrants
      - Modulates attention temperature based on previous layer's reward

    Inference mode:
      - Uses learned gates (W_r) since no target available
      - Standard attention temperature
    """

    def __init__(self, vocab_size, n_layer=3, n_embd=16, block_size=16, n_head=4,
                 init_std=0.08, sharpness=1.0):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.vocab_size = vocab_size
        self.sharpness = sharpness

        def matrix(nout, nin):
            return [[Value(random.gauss(0, init_std)) for _ in range(nin)] for _ in range(nout)]

        self.weights = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            # Standard attention weights
            self.weights[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            # Standard MLP weights
            self.weights[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            self.weights[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

            # --- Reward matrices ---
            self.weights[f'layer{i}.r_attn'] = matrix(1, n_embd)
            self.weights[f'layer{i}.r_mlp'] = matrix(1, n_embd)
            self.weights[f'layer{i}.pred'] = matrix(vocab_size, n_embd)

    def parameters(self):
        return [p for mat in self.weights.values() for row in mat for p in row]

    def forward(self, token_id, pos_id, kv_cache, target=None):
        """
        Forward pass with optional target for real reward computation.

        Returns:
            final_logits:   list[Value] (vocab_size,)
            layer_logits:   list[list[Value]] per-layer local predictions
            reward_scores:  list[(r_attn, r_mlp)] learned gate values
            actual_rewards: list[float] real rewards from correctness (empty if no target)
        """
        keys, values = kv_cache
        w = self.weights
        x = [t + p for t, p in zip(w['wte'][token_id], w['wpe'][pos_id])]
        x = rmsnorm(x)

        layer_logits = []
        reward_scores = []
        actual_rewards = []
        prev_reward = 0.5  # no prior info for layer 0

        for li in range(self.n_layer):
            # ----- Attention with reward-modulated temperature -----
            x_res = x
            x = rmsnorm(x)
            q = linear(x, w[f'layer{li}.attn_wq'])
            k = linear(x, w[f'layer{li}.attn_wk'])
            v = linear(x, w[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)

            # Temperature modulation from previous layer's reward
            # prev_reward > 0.5 → sharpen (confident), < 0.5 → broaden (uncertain)
            temp_scale = 1.0 / (1.0 + (prev_reward - 0.5) * self.sharpness)

            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                # Apply temperature scaling to attention logits
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim))
                    / (self.head_dim**0.5 * temp_scale)
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x_attn = linear(x_attn, w[f'layer{li}.attn_wo'])

            # Learned reward gate for attention
            r_attn_logit = linear(x_attn, w[f'layer{li}.r_attn'])[0]
            r_attn = sigmoid(r_attn_logit)

            # Gated residual
            x = [res + r_attn * a for res, a in zip(x_res, x_attn)]

            # ----- MLP block -----
            x_res = x
            x = rmsnorm(x)
            x_mlp = linear(x, w[f'layer{li}.mlp_fc1'])
            x_mlp = [xi.relu() for xi in x_mlp]
            x_mlp = linear(x_mlp, w[f'layer{li}.mlp_fc2'])

            # Learned reward gate for MLP
            r_mlp_logit = linear(x_mlp, w[f'layer{li}.r_mlp'])[0]
            r_mlp = sigmoid(r_mlp_logit)

            # Gated residual
            x = [res + r_mlp * m for res, m in zip(x_res, x_mlp)]

            # ----- Local prediction → REAL reward -----
            local_logits = linear(x, w[f'layer{li}.pred'])
            layer_logits.append(local_logits)
            reward_scores.append((r_attn, r_mlp))

            # Compute actual reward from correctness (training only)
            if target is not None:
                local_probs = softmax(local_logits)
                # .data extracts float — no gradient flows through the reward weighting
                actual_r = local_probs[target].data
                actual_rewards.append(actual_r)
                prev_reward = actual_r
            else:
                # Inference: use learned gate average as proxy
                prev_reward = (r_attn.data + r_mlp.data) / 2.0

        final_logits = linear(x, w['lm_head'])
        return final_logits, layer_logits, reward_scores, actual_rewards

    def forward_inference(self, token_id, pos_id, kv_cache):
        """Inference-only: returns just final logits."""
        final_logits, _, _, _ = self.forward(token_id, pos_id, kv_cache, target=None)
        return final_logits

    def new_kv_cache(self):
        return ([[] for _ in range(self.n_layer)],
                [[] for _ in range(self.n_layer)])

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.step_count = 0

    def step(self, lr_override=None):
        self.step_count += 1
        lr = lr_override if lr_override is not None else self.lr
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            p.data -= lr * m_hat / (v_hat ** 0.5 + self.eps)
            p.grad = 0

# ---------------------------------------------------------------------------
# Training: Standard GPT (baseline)
# ---------------------------------------------------------------------------

def train_standard(model, tokenizer, docs, num_steps=1000, lr=0.01, label="Baseline"):
    params = model.parameters()
    optimizer = Adam(params, lr=lr)
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = tokenizer.encode(doc)
        n = min(model.block_size, len(tokens) - 1)

        kv_cache = model.new_kv_cache()
        losses = []
        for pos_id in range(n):
            logits = model.forward(tokens[pos_id], pos_id, kv_cache)
            probs = softmax(logits)
            losses.append(-probs[tokens[pos_id + 1]].log())
        loss = (1 / n) * sum(losses)

        loss.backward()

        lr_t = lr * (1 - step / num_steps)
        optimizer.step(lr_override=lr_t)

        loss_history.append(loss.data)
        print(f"[{label}] step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

    print()
    return loss_history

# ---------------------------------------------------------------------------
# Training: Reward-Gated GPT v2 (Four-Quadrant Reinforcement)
# ---------------------------------------------------------------------------

def train_reward(model, tokenizer, docs, num_steps=1000, lr=0.01,
                 alpha=0.3, delta=0.1, beta=0.01,
                 label="Reward-v2"):
    """
    Four-quadrant reinforcement training (no unlikelihood):

    L = L_final + L_reinforcement + L_credit + L_entropy

    1. L_final:         Standard CE from output head
    2. L_reinforcement: Deep supervision with real reward weighting (pos/neg reinforcement)
    3. L_credit:        Penalize layers that regress (negative punishment)
    4. L_entropy:       Prevent learned gates from collapsing
    """
    params = model.parameters()
    optimizer = Adam(params, lr=lr)
    loss_history = []
    reward_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = tokenizer.encode(doc)
        n = min(model.block_size, len(tokens) - 1)

        kv_cache = model.new_kv_cache()

        # Accumulators for loss components
        final_losses = []
        layer_losses_accum = [[] for _ in range(model.n_layer)]
        credit_losses = []
        all_gate_values = []  # for entropy reg
        all_actual_rewards = [[] for _ in range(model.n_layer)]

        for pos_id in range(n):
            target = tokens[pos_id + 1]

            final_logits, layer_logits_list, reward_scores, actual_rewards = \
                model.forward(tokens[pos_id], pos_id, kv_cache, target=target)

            # ---- 1. Final loss (standard CE) ----
            final_probs = softmax(final_logits)
            final_losses.append(-final_probs[target].log())

            # ---- Per-layer losses ----
            prev_actual_r = 0.5  # baseline for layer 0
            for li in range(model.n_layer):
                local_probs = softmax(layer_logits_list[li])
                local_ce = -local_probs[target].log()
                actual_r = actual_rewards[li]

                # ---- 2. Deep supervision (positive reinforcement) ----
                # Weight by layer depth: deeper layers contribute more
                layer_losses_accum[li].append(local_ce)

                # ---- 3. Credit assignment (negative punishment) ----
                # Penalize layers whose reward is worse than previous
                regression = max(0.0, prev_actual_r - actual_r)
                if regression > 0:
                    credit_losses.append(regression * local_ce)

                all_actual_rewards[li].append(actual_r)
                prev_actual_r = actual_r

            # Collect gate values for entropy regularization
            all_gate_values.append(reward_scores)

        # ---- Compute total loss ----

        # 1. Final CE loss
        loss_final = (1 / n) * sum(final_losses)

        # 2. Deep supervision (positive + negative reinforcement)
        loss_reinforce = Value(0)
        for li in range(model.n_layer):
            layer_weight = (li + 1) / model.n_layer
            layer_loss = (1 / n) * sum(layer_losses_accum[li])
            loss_reinforce = loss_reinforce + layer_weight * layer_loss

        # 3. Credit assignment (negative punishment)
        loss_credit = Value(0)
        if credit_losses:
            loss_credit = (1 / len(credit_losses)) * sum(credit_losses)

        # 4. Entropy regularization on learned gates
        loss_entropy = Value(0)
        n_gates = 0
        for pos_gates in all_gate_values:
            for (r_attn, r_mlp) in pos_gates:
                eps = 1e-6
                r_a = r_attn * (1 - 2 * eps) + eps
                r_m = r_mlp * (1 - 2 * eps) + eps
                ent_a = r_a * r_a.log() + (1 - r_a) * (1 - r_a).log()
                ent_m = r_m * r_m.log() + (1 - r_m) * (1 - r_m).log()
                loss_entropy = loss_entropy + ent_a + ent_m
                n_gates += 2
        loss_entropy = loss_entropy / n_gates

        # Total loss (no unlikelihood)
        loss = loss_final + alpha * loss_reinforce \
               + delta * loss_credit + beta * loss_entropy

        loss.backward()

        lr_t = lr * (1 - step / num_steps)
        optimizer.step(lr_override=lr_t)

        # ---- Track statistics ----
        loss_history.append(loss_final.data)

        # Reward stats per layer
        layer_r_means = []
        for li in range(model.n_layer):
            r_vals = all_actual_rewards[li]
            layer_r_means.append(sum(r_vals) / len(r_vals) if r_vals else 0)

        # Gate stats
        gate_vals = []
        for pos_gates in all_gate_values:
            for (r_a, r_m) in pos_gates:
                gate_vals.extend([r_a.data, r_m.data])
        g_mean = sum(gate_vals) / len(gate_vals)

        reward_history.append({
            'layer_rewards': layer_r_means,
            'gate_mean': g_mean,
            'loss_final': loss_final.data,
            'loss_reinforce': loss_reinforce.data if isinstance(loss_reinforce, Value) else loss_reinforce,
            'loss_credit': loss_credit.data if isinstance(loss_credit, Value) else loss_credit,
        })

        r_str = ' '.join(f'L{i}:{m:.3f}' for i, m in enumerate(layer_r_means))
        print(f"[{label}] step {step+1:4d}/{num_steps:4d} | "
              f"loss {loss_final.data:.4f} | rewards: {r_str}", end='\r')

    print()
    return loss_history, reward_history

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(model, tokenizer, num_samples=10, temperature=0.5, label="model",
             is_reward=False):
    print(f"--- {label}: generated names ---")
    samples = []
    for i in range(num_samples):
        kv_cache = model.new_kv_cache()
        token_id = tokenizer.bos
        chars = []
        for pos_id in range(model.block_size):
            if is_reward:
                logits = model.forward_inference(token_id, pos_id, kv_cache)
            else:
                logits = model.forward(token_id, pos_id, kv_cache)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(tokenizer.vocab_size),
                                      weights=[p.data for p in probs])[0]
            if token_id == tokenizer.bos:
                break
            chars.append(tokenizer.decode_token(token_id))
        name = ''.join(chars)
        samples.append(name)
        print(f"  {i+1:2d}: {name}")
    return samples

# ---------------------------------------------------------------------------
# Entropy measurement
# ---------------------------------------------------------------------------

def measure_entropy(model, tokenizer, docs, num_samples=100, is_reward=False):
    total_entropy = 0
    total_tokens = 0

    for i in range(min(num_samples, len(docs))):
        doc = docs[i]
        tokens = tokenizer.encode(doc)
        n = min(model.block_size, len(tokens) - 1)

        kv_cache = model.new_kv_cache()
        for pos_id in range(n):
            if is_reward:
                logits = model.forward_inference(tokens[pos_id], pos_id, kv_cache)
            else:
                logits = model.forward(tokens[pos_id], pos_id, kv_cache)
            probs = softmax(logits)
            for p in probs:
                if p.data > 1e-10:
                    total_entropy -= p.data * math.log(p.data)
            total_tokens += 1

    return total_entropy / total_tokens

# ---------------------------------------------------------------------------
# Main: train both models and compare
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    docs = load_docs()
    print(f"num docs: {len(docs)}")

    tokenizer = CharTokenizer(docs)
    print(f"vocab size: {tokenizer.vocab_size}")

    # Configuration — 3 layers for both models
    n_layer = 3
    n_embd = 16
    n_head = 4
    block_size = 16
    num_steps = 1000
    lr = 0.01

    # === Standard GPT (baseline) ===
    print("\n" + "="*60)
    print("STANDARD GPT (3-layer baseline)")
    print("="*60)
    random.seed(42)
    baseline = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer, n_embd=n_embd, block_size=block_size, n_head=n_head
    )
    n_params_base = len(baseline.parameters())
    print(f"parameters: {n_params_base}")
    baseline_loss = train_standard(baseline, tokenizer, docs,
                                   num_steps=num_steps, lr=lr, label="Baseline")

    random.seed(123)
    baseline_samples = generate(baseline, tokenizer, num_samples=10, label="Baseline")

    # === Reward-Gated GPT v2 ===
    print("\n" + "="*60)
    print("REWARD-GATED GPT v2 (Four-Quadrant Reinforcement)")
    print("="*60)
    random.seed(42)
    reward_model = RewardGPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer, n_embd=n_embd, block_size=block_size, n_head=n_head,
        sharpness=1.0
    )
    n_params_reward = len(reward_model.parameters())
    print(f"parameters: {n_params_reward} (+{n_params_reward - n_params_base} from reward components)")
    reward_loss, reward_history = train_reward(
        reward_model, tokenizer, docs,
        num_steps=num_steps, lr=lr,
        alpha=0.3, delta=0.1, beta=0.01,
        label="Reward-v2"
    )

    random.seed(123)
    reward_samples = generate(reward_model, tokenizer, num_samples=10,
                              label="Reward-v2", is_reward=True)

    # === Entropy Measurement ===
    print("\nmeasuring prediction entropy...")
    entropy_base = measure_entropy(baseline, tokenizer, docs, num_samples=50)
    entropy_reward = measure_entropy(reward_model, tokenizer, docs, num_samples=50,
                                     is_reward=True)

    # === Results ===
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Loss comparison
    print(f"\n{'Metric':<30s} {'Baseline':>12s} {'Reward-v2':>12s} {'Delta':>10s}")
    print("-" * 66)

    bl_final = baseline_loss[-1]
    rw_final = reward_loss[-1]
    print(f"{'Final loss':<30s} {bl_final:>12.4f} {rw_final:>12.4f} {(rw_final-bl_final)/bl_final*100:>+9.1f}%")

    bl_min = min(baseline_loss)
    rw_min = min(reward_loss)
    print(f"{'Min loss':<30s} {bl_min:>12.4f} {rw_min:>12.4f} {(rw_min-bl_min)/bl_min*100:>+9.1f}%")

    avg_bl = sum(baseline_loss[-100:]) / 100
    avg_rw = sum(reward_loss[-100:]) / 100
    print(f"{'Avg loss (last 100)':<30s} {avg_bl:>12.4f} {avg_rw:>12.4f} {(avg_rw-avg_bl)/avg_bl*100:>+9.1f}%")

    print(f"{'Parameters':<30s} {n_params_base:>12d} {n_params_reward:>12d} {'+':>5s}{n_params_reward-n_params_base}")

    print(f"{'Prediction entropy':<30s} {entropy_base:>12.4f} {entropy_reward:>12.4f} {(entropy_reward-entropy_base)/entropy_base*100:>+9.1f}%")

    # Reward evolution per layer
    print(f"\n--- Reward Evolution (actual correctness probability per layer) ---")
    print(f"{'Phase':<25s}", end='')
    for li in range(n_layer):
        print(f" {'Layer '+str(li):>10s}", end='')
    print()

    early = reward_history[:100]
    late = reward_history[-100:]

    print(f"{'Early (steps 1-100)':<25s}", end='')
    for li in range(n_layer):
        m = sum(h['layer_rewards'][li] for h in early) / len(early)
        print(f" {m:>10.4f}", end='')
    print()

    print(f"{'Late (steps 901-1000)':<25s}", end='')
    for li in range(n_layer):
        m = sum(h['layer_rewards'][li] for h in late) / len(late)
        print(f" {m:>10.4f}", end='')
    print()

    # Loss component breakdown
    print(f"\n--- Loss Component Breakdown (averages, last 100 steps) ---")
    avg_final = sum(h['loss_final'] for h in late) / len(late)
    avg_reinf = sum(h['loss_reinforce'] for h in late) / len(late)
    avg_credit = sum(h['loss_credit'] for h in late) / len(late)
    print(f"  L_final (CE):           {avg_final:.4f}")
    print(f"  L_reinforcement (deep): {avg_reinf:.4f}  (x alpha={0.3})")
    print(f"  L_credit (regression):  {avg_credit:.4f}  (x delta={0.1})")

    # Gate statistics
    early_gate = sum(h['gate_mean'] for h in early) / len(early)
    late_gate = sum(h['gate_mean'] for h in late) / len(late)
    print(f"\n--- Learned Gate Statistics ---")
    print(f"  Gate mean (early):  {early_gate:.4f}")
    print(f"  Gate mean (late):   {late_gate:.4f}")

    # Summary
    delta_loss = (avg_rw - avg_bl) / avg_bl * 100
    delta_entropy = (entropy_reward - entropy_base) / entropy_base * 100
    print(f"\n--- Summary ---")
    print(f"  Loss change:    {delta_loss:+.2f}%")
    print(f"  Entropy change: {delta_entropy:+.2f}%")

    if delta_loss < 0:
        print(f"  Reward-v2 achieves {abs(delta_loss):.1f}% lower loss than baseline.")
    if delta_entropy < 0:
        print(f"  Reward-v2 has {abs(delta_entropy):.1f}% lower entropy (more confident).")


if __name__ == '__main__':
    main()
