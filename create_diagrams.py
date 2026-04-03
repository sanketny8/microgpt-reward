"""Generate architecture diagrams for README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colors ──
C_BG       = '#ffffff'
C_BLUE     = '#3b82f6'
C_BLUE_L   = '#dbeafe'
C_PURPLE   = '#7c3aed'
C_PURPLE_L = '#ede9fe'
C_GREEN    = '#059669'
C_GREEN_L  = '#a7f3d0'
C_RED      = '#dc2626'
C_RED_L    = '#fecaca'
C_ORANGE   = '#ea580c'
C_ORANGE_L = '#fed7aa'
C_GRAY     = '#64748b'
C_GRAY_L   = '#f1f5f9'
C_DARK     = '#1e293b'
C_YELLOW_L = '#fef3c7'


def draw_box(ax, x, y, w, h, label, fc, ec, fontsize=9, bold=False, text_color='#1e293b'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color=C_GRAY, style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_dashed_arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='dashed'))


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 1: Standard GPT vs Reward-Gated GPT (side by side)
# ═══════════════════════════════════════════════════════════════

def create_comparison_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_aspect('equal')

    # ── Left: Standard GPT ──
    ax1.set_title('Standard GPT', fontsize=16, fontweight='bold', color=C_DARK, pad=15)

    bw, bh = 0.6, 0.055
    cx = 0.2

    # Input
    draw_box(ax1, cx, 0.90, bw, bh, 'Token + Position Embedding', C_BLUE_L, C_BLUE, 9, True)
    draw_arrow(ax1, cx+bw/2, 0.90, cx+bw/2, 0.87)

    draw_box(ax1, cx, 0.815, bw, bh, 'RMSNorm', C_GRAY_L, C_GRAY, 9)
    draw_arrow(ax1, cx+bw/2, 0.815, cx+bw/2, 0.785)

    # Attention
    draw_box(ax1, cx, 0.73, bw, bh, 'Q, K, V Projections', C_BLUE_L, C_BLUE, 9)
    draw_arrow(ax1, cx+bw/2, 0.73, cx+bw/2, 0.70)

    draw_box(ax1, cx, 0.645, bw, bh, 'Multi-Head Attention', C_BLUE_L, C_BLUE, 9, True)
    draw_arrow(ax1, cx+bw/2, 0.645, cx+bw/2, 0.615)

    draw_box(ax1, cx, 0.56, bw, bh, 'Output Projection (W_o)', C_BLUE_L, C_BLUE, 9)
    draw_arrow(ax1, cx+bw/2, 0.56, cx+bw/2, 0.53)

    # Residual add
    draw_box(ax1, cx, 0.475, bw, bh, 'x = x_res + x_attn', C_GRAY_L, C_GRAY, 9, True)
    # Residual skip line
    draw_dashed_arrow(ax1, cx-0.03, 0.87, cx-0.03, 0.505, C_GRAY)
    ax1.annotate('', xy=(cx, 0.505), xytext=(cx-0.03, 0.505),
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2, linestyle='dashed'))
    ax1.text(cx-0.06, 0.69, 'residual', fontsize=7, color=C_GRAY, rotation=90, ha='center')

    draw_arrow(ax1, cx+bw/2, 0.475, cx+bw/2, 0.445)

    # MLP
    draw_box(ax1, cx, 0.39, bw, bh, 'RMSNorm', C_GRAY_L, C_GRAY, 9)
    draw_arrow(ax1, cx+bw/2, 0.39, cx+bw/2, 0.36)

    draw_box(ax1, cx, 0.305, bw, bh, 'FC1 → ReLU → FC2', C_YELLOW_L, C_ORANGE, 9, True)
    draw_arrow(ax1, cx+bw/2, 0.305, cx+bw/2, 0.275)

    # Residual add
    draw_box(ax1, cx, 0.22, bw, bh, 'x = x_res + x_mlp', C_GRAY_L, C_GRAY, 9, True)
    # Residual skip
    draw_dashed_arrow(ax1, cx-0.03, 0.445, cx-0.03, 0.25, C_GRAY)
    ax1.annotate('', xy=(cx, 0.25), xytext=(cx-0.03, 0.25),
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2, linestyle='dashed'))

    draw_arrow(ax1, cx+bw/2, 0.22, cx+bw/2, 0.19)

    # Output
    draw_box(ax1, cx, 0.135, bw, bh, 'Output → Next Layer / LM Head', C_BLUE_L, C_BLUE, 9)

    # Border
    rect = mpatches.FancyBboxPatch((0.05, 0.08), 0.9, 0.90,
            boxstyle="round,pad=0.02", facecolor='none', edgecolor=C_GRAY, lw=2, linestyle='--')
    ax1.add_patch(rect)

    # ── Right: Reward-Gated GPT v2 ──
    ax2.set_title('Reward-Gated GPT v2', fontsize=16, fontweight='bold', color=C_PURPLE, pad=15)

    cx = 0.15
    bw2 = 0.5

    # Input
    draw_box(ax2, cx, 0.92, bw2, 0.045, 'Token + Position Embedding', C_BLUE_L, C_BLUE, 8, True)
    draw_arrow(ax2, cx+bw2/2, 0.92, cx+bw2/2, 0.895)

    draw_box(ax2, cx, 0.85, bw2, 0.045, 'RMSNorm', C_GRAY_L, C_GRAY, 8)
    draw_arrow(ax2, cx+bw2/2, 0.85, cx+bw2/2, 0.825)

    # Temperature modulation
    draw_box(ax2, 0.70, 0.85, 0.25, 0.045, 'prev_reward', C_RED_L, C_RED, 7, True)
    draw_arrow(ax2, 0.70, 0.872, cx+bw2+0.01, 0.78, C_RED, lw=1.2)
    ax2.text(0.72, 0.815, 'temp = 1/(1+(r-0.5)×s)', fontsize=6, color=C_RED, style='italic')

    # Attention
    draw_box(ax2, cx, 0.78, bw2, 0.045, 'Q, K, V Projections', C_BLUE_L, C_BLUE, 8)
    draw_arrow(ax2, cx+bw2/2, 0.78, cx+bw2/2, 0.755)

    draw_box(ax2, cx, 0.71, bw2, 0.045, 'Multi-Head Attention (temp-modulated)', C_BLUE_L, C_BLUE, 8, True)
    draw_arrow(ax2, cx+bw2/2, 0.71, cx+bw2/2, 0.685)

    draw_box(ax2, cx, 0.64, bw2, 0.045, 'Output Projection (W_o)', C_BLUE_L, C_BLUE, 8)
    draw_arrow(ax2, cx+bw2/2, 0.64, cx+bw2/2, 0.615)

    # Reward gate attn
    draw_box(ax2, cx, 0.57, bw2, 0.045, 'r_attn = sigmoid(W_r_attn @ x_attn)', C_PURPLE_L, C_PURPLE, 8, True)
    draw_arrow(ax2, cx+bw2/2, 0.57, cx+bw2/2, 0.545)

    # Gated residual
    draw_box(ax2, cx, 0.50, bw2, 0.045, 'x = x_res + r_attn × x_attn', C_PURPLE_L, C_PURPLE, 8)
    # Residual skip
    draw_dashed_arrow(ax2, cx-0.03, 0.895, cx-0.03, 0.525, C_GRAY)
    ax2.annotate('', xy=(cx, 0.525), xytext=(cx-0.03, 0.525),
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2, linestyle='dashed'))

    draw_arrow(ax2, cx+bw2/2, 0.50, cx+bw2/2, 0.475)

    # MLP
    draw_box(ax2, cx, 0.43, bw2, 0.045, 'RMSNorm', C_GRAY_L, C_GRAY, 8)
    draw_arrow(ax2, cx+bw2/2, 0.43, cx+bw2/2, 0.405)

    draw_box(ax2, cx, 0.36, bw2, 0.045, 'FC1 → ReLU → FC2', C_YELLOW_L, C_ORANGE, 8, True)
    draw_arrow(ax2, cx+bw2/2, 0.36, cx+bw2/2, 0.335)

    # Reward gate MLP
    draw_box(ax2, cx, 0.29, bw2, 0.045, 'r_mlp = sigmoid(W_r_mlp @ x_mlp)', C_PURPLE_L, C_PURPLE, 8, True)
    draw_arrow(ax2, cx+bw2/2, 0.29, cx+bw2/2, 0.265)

    # Gated residual
    draw_box(ax2, cx, 0.22, bw2, 0.045, 'x = x_res + r_mlp × x_mlp', C_PURPLE_L, C_PURPLE, 8)
    # Residual skip
    draw_dashed_arrow(ax2, cx-0.03, 0.475, cx-0.03, 0.245, C_GRAY)
    ax2.annotate('', xy=(cx, 0.245), xytext=(cx-0.03, 0.245),
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2, linestyle='dashed'))

    draw_arrow(ax2, cx+bw2/2, 0.22, cx+bw2/2, 0.195)

    # Local prediction
    draw_box(ax2, cx, 0.15, bw2, 0.045, 'W_p @ x → local_logits', C_ORANGE_L, C_ORANGE, 8, True)
    draw_arrow(ax2, cx+bw2/2, 0.15, cx+bw2/2, 0.125)

    # Real reward
    draw_box(ax2, cx, 0.08, bw2, 0.045, 'actual_reward = p(target)', C_GREEN_L, C_GREEN, 8, True)

    # Arrow from reward back to top (next layer)
    draw_arrow(ax2, cx+bw2+0.02, 0.10, 0.82, 0.10, C_GREEN, lw=1.5)
    draw_arrow(ax2, 0.82, 0.10, 0.82, 0.872, C_GREEN, lw=1.5)

    ax2.text(0.85, 0.50, '→ next\nlayer', fontsize=7, color=C_GREEN, ha='center', fontweight='bold')

    # Border
    rect = mpatches.FancyBboxPatch((0.05, 0.03), 0.9, 0.95,
            boxstyle="round,pad=0.02", facecolor='none', edgecolor=C_PURPLE, lw=2, linestyle='--')
    ax2.add_patch(rect)

    # Legend for new components
    ax2.text(0.70, 0.72, 'NEW', fontsize=8, color=C_PURPLE, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', facecolor=C_PURPLE_L, edgecolor=C_PURPLE))
    ax2.text(0.70, 0.16, 'NEW', fontsize=8, color=C_ORANGE, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', facecolor=C_ORANGE_L, edgecolor=C_ORANGE))
    ax2.text(0.70, 0.09, 'NEW', fontsize=8, color=C_GREEN, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', facecolor=C_GREEN_L, edgecolor=C_GREEN))

    plt.tight_layout()
    plt.savefig('/Users/sanyayadhish/MyProfile/microgpt-reward/diagrams/architecture_comparison.png',
                dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print("Created: architecture_comparison.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 2: Four-Quadrant Loss System
# ═══════════════════════════════════════════════════════════════

def create_four_quadrant_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.96, 'Four-Quadrant Reinforcement Loss', fontsize=18,
            fontweight='bold', ha='center', color=C_DARK)
    ax.text(0.5, 0.92, 'Behavioral psychology mapped to transformer training',
            fontsize=10, ha='center', color=C_GRAY, style='italic')

    # ── Top-left: Positive Reinforcement ──
    draw_box(ax, 0.03, 0.60, 0.44, 0.27, '', C_GREEN_L, C_GREEN, 9)
    ax.text(0.25, 0.84, 'POSITIVE REINFORCEMENT', fontsize=11, fontweight='bold',
            ha='center', color=C_GREEN)
    ax.text(0.25, 0.80, 'Add stimulus → Increase behavior', fontsize=8,
            ha='center', color=C_GRAY, style='italic')

    draw_box(ax, 0.06, 0.63, 0.38, 0.14, '', 'white', C_GREEN, 8)
    ax.text(0.25, 0.745, 'Deep Supervision', fontsize=10, fontweight='bold',
            ha='center', color=C_DARK)
    ax.text(0.25, 0.71, 'Each layer predicts next token via W_p', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.25, 0.675, 'L = alpha × sum(w_i × CE_layer_i)', fontsize=8,
            ha='center', color=C_DARK, family='monospace')
    ax.text(0.25, 0.645, 'w_i = (i+1)/n_layers  |  alpha = 0.3', fontsize=7,
            ha='center', color=C_GRAY)

    # ── Top-right: Negative Reinforcement ──
    draw_box(ax, 0.53, 0.60, 0.44, 0.27, '', C_BLUE_L, C_BLUE, 9)
    ax.text(0.75, 0.84, 'NEGATIVE REINFORCEMENT', fontsize=11, fontweight='bold',
            ha='center', color=C_BLUE)
    ax.text(0.75, 0.80, 'Remove stimulus → Increase behavior', fontsize=8,
            ha='center', color=C_GRAY, style='italic')

    draw_box(ax, 0.56, 0.63, 0.38, 0.14, '', 'white', C_BLUE, 8)
    ax.text(0.75, 0.745, 'Reward Improvement', fontsize=10, fontweight='bold',
            ha='center', color=C_DARK)
    ax.text(0.75, 0.71, 'When layer improves over previous,', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.75, 0.675, 'credit penalty is released', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.75, 0.645, 'relu(r_prev - r_curr) → 0 (no penalty)', fontsize=7,
            ha='center', color=C_DARK, family='monospace')

    # ── Bottom-left: Positive Punishment ──
    draw_box(ax, 0.03, 0.27, 0.44, 0.27, '', C_ORANGE_L, C_ORANGE, 9)
    ax.text(0.25, 0.51, 'POSITIVE PUNISHMENT', fontsize=11, fontweight='bold',
            ha='center', color=C_ORANGE)
    ax.text(0.25, 0.47, 'Add stimulus → Decrease behavior', fontsize=8,
            ha='center', color=C_GRAY, style='italic')

    draw_box(ax, 0.06, 0.30, 0.38, 0.14, '', 'white', C_ORANGE, 8)
    ax.text(0.25, 0.415, 'Reward-Gated Suppression', fontsize=10, fontweight='bold',
            ha='center', color=C_DARK)
    ax.text(0.25, 0.38, 'Sigmoid gates dampen layer outputs', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.25, 0.345, 'r = sigmoid(W_r @ x)  →  gates at 0.24', fontsize=8,
            ha='center', color=C_DARK, family='monospace')
    ax.text(0.25, 0.315, '76% of contribution suppressed', fontsize=7,
            ha='center', color=C_GRAY)

    # ── Bottom-right: Negative Punishment ──
    draw_box(ax, 0.53, 0.27, 0.44, 0.27, '', C_RED_L, C_RED, 9)
    ax.text(0.75, 0.51, 'NEGATIVE PUNISHMENT', fontsize=11, fontweight='bold',
            ha='center', color=C_RED)
    ax.text(0.75, 0.47, 'Remove stimulus → Decrease behavior', fontsize=8,
            ha='center', color=C_GRAY, style='italic')

    draw_box(ax, 0.56, 0.30, 0.38, 0.14, '', 'white', C_RED, 8)
    ax.text(0.75, 0.415, 'Credit Assignment', fontsize=10, fontweight='bold',
            ha='center', color=C_DARK)
    ax.text(0.75, 0.38, 'Penalize layers that regress', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.75, 0.345, 'L = delta × relu(r_prev - r_curr) × CE', fontsize=8,
            ha='center', color=C_DARK, family='monospace')
    ax.text(0.75, 0.315, 'delta = 0.1', fontsize=7, ha='center', color=C_GRAY)

    # ── Total Loss at bottom ──
    draw_box(ax, 0.15, 0.08, 0.70, 0.12, '', C_DARK, C_DARK, 10)
    ax.text(0.5, 0.155, 'Total Loss', fontsize=12, fontweight='bold',
            ha='center', color='white')
    ax.text(0.5, 0.115, 'L = L_final  +  alpha × L_reinforce  +  delta × L_credit  +  beta × L_entropy',
            fontsize=9, ha='center', color='#94a3b8', family='monospace')

    # Arrows from quadrants to total
    draw_arrow(ax, 0.25, 0.60, 0.30, 0.20, C_GREEN, lw=2)
    draw_arrow(ax, 0.75, 0.27, 0.70, 0.20, C_RED, lw=2)
    draw_dashed_arrow(ax, 0.25, 0.27, 0.35, 0.20, C_ORANGE)
    draw_dashed_arrow(ax, 0.75, 0.60, 0.65, 0.20, C_BLUE)

    ax.text(0.16, 0.22, 'explicit\nloss term', fontsize=7, color=C_GREEN, ha='center')
    ax.text(0.84, 0.22, 'explicit\nloss term', fontsize=7, color=C_RED, ha='center')
    ax.text(0.16, 0.55, 'implicit via\ngated residuals', fontsize=7, color=C_ORANGE, ha='center', style='italic')
    ax.text(0.84, 0.55, 'implicit via\nreduced L_credit', fontsize=7, color=C_BLUE, ha='center', style='italic')

    plt.savefig('/Users/sanyayadhish/MyProfile/microgpt-reward/diagrams/four_quadrant_loss.png',
                dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print("Created: four_quadrant_loss.png")


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 3: Full 3-Layer Stack with reward chaining
# ═══════════════════════════════════════════════════════════════

def create_stack_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.97, 'Reward-Gated GPT v2 — Full 3-Layer Architecture',
            fontsize=16, fontweight='bold', ha='center', color=C_DARK)

    # ── Token Embedding ──
    draw_box(ax, 0.25, 0.88, 0.30, 0.05, 'Token + Position Embedding', C_BLUE_L, C_BLUE, 10, True)
    draw_arrow(ax, 0.40, 0.88, 0.40, 0.855)
    draw_box(ax, 0.25, 0.81, 0.30, 0.045, 'RMSNorm', C_GRAY_L, C_GRAY, 9)
    draw_arrow(ax, 0.40, 0.81, 0.40, 0.785)

    # ── Layer 0 ──
    draw_box(ax, 0.15, 0.68, 0.50, 0.105, '', C_PURPLE_L, C_PURPLE)
    ax.text(0.40, 0.77, 'Layer 0  —  Reward-Gated Block', fontsize=10,
            fontweight='bold', ha='center', color=C_PURPLE)
    ax.text(0.40, 0.74, 'Attn(temp) → r_attn gate → MLP → r_mlp gate', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.40, 0.71, 'W_p → local pred → reward₀', fontsize=8,
            ha='center', color=C_DARK, family='monospace')
    ax.text(0.40, 0.695, 'reward₀ = 0.160', fontsize=8, ha='center',
            color=C_GREEN, fontweight='bold')

    # Deep supervision arrow from Layer 0
    draw_arrow(ax, 0.65, 0.73, 0.78, 0.73, C_GREEN, lw=1.5)
    draw_box(ax, 0.78, 0.71, 0.18, 0.04, 'L_reinforce₀', C_GREEN_L, C_GREEN, 8)
    ax.text(0.87, 0.695, 'w₀ = 1/3', fontsize=7, ha='center', color=C_GRAY)

    draw_arrow(ax, 0.40, 0.68, 0.40, 0.655)

    # ── Layer 1 ──
    draw_box(ax, 0.15, 0.55, 0.50, 0.105, '', C_PURPLE_L, C_PURPLE)
    ax.text(0.40, 0.64, 'Layer 1  —  Reward-Gated Block', fontsize=10,
            fontweight='bold', ha='center', color=C_PURPLE)
    ax.text(0.40, 0.61, 'Attn(temp←reward₀) → gate → MLP → gate', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.40, 0.58, 'W_p → local pred → reward₁', fontsize=8,
            ha='center', color=C_DARK, family='monospace')
    ax.text(0.40, 0.565, 'reward₁ = 0.200  (+25%)', fontsize=8, ha='center',
            color=C_GREEN, fontweight='bold')

    # Deep supervision arrow from Layer 1
    draw_arrow(ax, 0.65, 0.60, 0.78, 0.60, C_GREEN, lw=1.5)
    draw_box(ax, 0.78, 0.58, 0.18, 0.04, 'L_reinforce₁', C_GREEN_L, C_GREEN, 8)
    ax.text(0.87, 0.565, 'w₁ = 2/3', fontsize=7, ha='center', color=C_GRAY)

    # Reward chain arrow
    draw_arrow(ax, 0.12, 0.695, 0.12, 0.615, C_GREEN, lw=2)
    ax.text(0.09, 0.66, 'reward\nchain', fontsize=7, color=C_GREEN, ha='center', fontweight='bold')

    draw_arrow(ax, 0.40, 0.55, 0.40, 0.525)

    # ── Layer 2 ──
    draw_box(ax, 0.15, 0.42, 0.50, 0.105, '', C_PURPLE_L, C_PURPLE)
    ax.text(0.40, 0.51, 'Layer 2  —  Reward-Gated Block', fontsize=10,
            fontweight='bold', ha='center', color=C_PURPLE)
    ax.text(0.40, 0.48, 'Attn(temp←reward₁) → gate → MLP → gate', fontsize=8,
            ha='center', color=C_GRAY)
    ax.text(0.40, 0.45, 'W_p → local pred → reward₂', fontsize=8,
            ha='center', color=C_DARK, family='monospace')
    ax.text(0.40, 0.435, 'reward₂ = 0.219  (+37% vs L0)', fontsize=8, ha='center',
            color=C_GREEN, fontweight='bold')

    # Deep supervision arrow from Layer 2
    draw_arrow(ax, 0.65, 0.47, 0.78, 0.47, C_GREEN, lw=1.5)
    draw_box(ax, 0.78, 0.45, 0.18, 0.04, 'L_reinforce₂', C_GREEN_L, C_GREEN, 8)
    ax.text(0.87, 0.435, 'w₂ = 3/3', fontsize=7, ha='center', color=C_GRAY)

    # Reward chain arrow
    draw_arrow(ax, 0.12, 0.565, 0.12, 0.485, C_GREEN, lw=2)

    draw_arrow(ax, 0.40, 0.42, 0.40, 0.395)

    # ── LM Head ──
    draw_box(ax, 0.25, 0.33, 0.30, 0.065, 'LM Head\n(full precision)', C_BLUE_L, C_BLUE, 10, True)
    draw_arrow(ax, 0.40, 0.33, 0.40, 0.305)

    # ── Final Loss ──
    draw_box(ax, 0.20, 0.24, 0.40, 0.065, 'L_final  (standard CE)', C_DARK, C_DARK, 10, True, 'white')

    # ── Credit assignment arrows ──
    # Between L0 and L1
    ax.annotate('', xy=(0.78, 0.665), xytext=(0.78, 0.635),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.5))
    ax.text(0.78, 0.648, 'credit\ncheck', fontsize=6, color=C_RED, ha='center')

    # Between L1 and L2
    ax.annotate('', xy=(0.78, 0.535), xytext=(0.78, 0.505),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.5))
    ax.text(0.78, 0.518, 'credit\ncheck', fontsize=6, color=C_RED, ha='center')

    # ── Legend ──
    draw_box(ax, 0.05, 0.04, 0.90, 0.16, '', C_GRAY_L, C_GRAY)
    ax.text(0.5, 0.175, 'Legend', fontsize=10, fontweight='bold', ha='center', color=C_DARK)

    draw_box(ax, 0.08, 0.10, 0.12, 0.03, '', C_PURPLE_L, C_PURPLE, 7)
    ax.text(0.22, 0.115, 'Reward-gated layer', fontsize=8, color=C_DARK, va='center')

    draw_box(ax, 0.08, 0.06, 0.12, 0.03, '', C_GREEN_L, C_GREEN, 7)
    ax.text(0.22, 0.075, 'Deep supervision loss', fontsize=8, color=C_DARK, va='center')

    ax.plot([0.42, 0.48], [0.115, 0.115], color=C_GREEN, lw=2)
    ax.annotate('', xy=(0.48, 0.115), xytext=(0.46, 0.115),
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=2))
    ax.text(0.50, 0.115, 'Reward chain (layer→layer)', fontsize=8, color=C_DARK, va='center')

    ax.annotate('', xy=(0.48, 0.075), xytext=(0.42, 0.075),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.5))
    ax.text(0.50, 0.075, 'Credit assignment check', fontsize=8, color=C_DARK, va='center')

    plt.savefig('/Users/sanyayadhish/MyProfile/microgpt-reward/diagrams/full_stack.png',
                dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print("Created: full_stack.png")


if __name__ == '__main__':
    import os
    os.makedirs('/Users/sanyayadhish/MyProfile/microgpt-reward/diagrams', exist_ok=True)
    create_comparison_diagram()
    create_four_quadrant_diagram()
    create_stack_diagram()
    print("\nAll diagrams saved to diagrams/")
