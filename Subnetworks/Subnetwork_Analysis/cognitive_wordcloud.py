"""
Generate word clouds for top cognitive domains in FPN-A and FPN-B subnetworks.
Reads from cognitive_domain_fpn_profiles_composite_score.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
import glob
from PIL import ImageFont

# ========== CONFIGURATION ==========
INPUT_CSV = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/cognitive_atlas/cognitive_domain_fpn_profiles_composite_score.csv'
OUTPUT_DIR = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/cognitive_atlas'
TOP_N = 25

# Color palettes
# FPN-A: greenish/teal gradient (light to dark)
FPNA_COLORS = ['#E0F2F1', '#B2DFDB', '#80CBC4', '#4DB6AC', '#26A69A', '#009688', '#00897B', '#00796B', '#00695C', '#004D40']
# FPN-B: blueish gradient (light to dark)
FPNB_COLORS = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']

print("="*60)
print("COGNITIVE DOMAIN WORD CLOUD GENERATOR")
print("="*60)
print("[0/4] Using WordCloud default font (DroidSansMono)")

# ========== 1. LOAD DATA ==========
print(f"\n[1/4] Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"  ✓ Loaded {len(df)} cognitive domains")

# ========== 2. PREPARE DATA FOR WORD CLOUDS ==========
print("\n[2/4] Preparing word cloud data...")

# Split into FPN-A (positive mean_diff) and FPN-B (negative mean_diff)
fpna_domains = df[df['mean_diff_a_minus_b'] > 0].copy()
fpnb_domains = df[df['mean_diff_a_minus_b'] < 0].copy()

# Get top N by rank_score
top_fpna = fpna_domains.nlargest(TOP_N, 'rank_score')
top_fpnb = fpnb_domains.nlargest(TOP_N, 'rank_score')

print(f"  ✓ Selected top {len(top_fpna)} FPN-A domains")
print(f"  ✓ Selected top {len(top_fpnb)} FPN-B domains")

def format_domain_name(domain_str):
    """Convert domain_name_with_underscores to Title Case With Spaces"""
    return domain_str.replace('_', ' ').replace('-', ' ').title()

def create_word_frequencies(domains_df):
    """
    Create frequency dictionary for WordCloud.
    Frequency (size) = rank_score
    Returns: dict {word: frequency}, color_dict {word: abs_cohens_d}
    """
    word_freq = {}
    color_values = {}
    
    for _, row in domains_df.iterrows():
        formatted_name = format_domain_name(row['cognitive_domain'])
        word_freq[formatted_name] = row['rank_score']
        color_values[formatted_name] = row['abs_cohens_d']
    
    return word_freq, color_values

fpna_freq, fpna_colors = create_word_frequencies(top_fpna)
fpnb_freq, fpnb_colors = create_word_frequencies(top_fpnb)

print(f"  ✓ Created frequency dictionaries")

# ========== 3. CREATE CUSTOM COLOR FUNCTIONS ==========
print("\n[3/4] Setting up color gradients...")

def create_color_func(color_palette, color_values):
    """
    Create a color function for WordCloud based on abs_cohens_d values.
    Higher abs_cohens_d = darker color
    """
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("custom", color_palette, N=256)
    
    # Normalize color values to [0, 1]
    min_val = min(color_values.values())
    max_val = max(color_values.values())
    
    def color_func(word, **kwargs):
        # Get the abs_cohens_d for this word
        cohens_d = color_values.get(word, 0)
        # Normalize to [0, 1]
        normalized = (cohens_d - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        # Get color from colormap
        rgba = cmap(normalized)
        # Convert to hex
        return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    
    return color_func

fpna_color_func = create_color_func(FPNA_COLORS, fpna_colors)
fpnb_color_func = create_color_func(FPNB_COLORS, fpnb_colors)

print(f"  ✓ FPN-A color range: {min(fpna_colors.values()):.3f} - {max(fpna_colors.values()):.3f} |Cohen's d|")
print(f"  ✓ FPN-B color range: {min(fpnb_colors.values()):.3f} - {max(fpnb_colors.values()):.3f} |Cohen's d|")

# ========== 4. GENERATE WORD CLOUDS ==========
print("\n[4/4] Generating word clouds...")

def generate_wordcloud(word_freq, color_func, title, output_path):
    """Generate and save a word cloud"""
    try:
        # Create word cloud WITHOUT specifying font_path (use default)
        wc = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            color_func=color_func,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=120,
            prefer_horizontal=0.7,
            margin=10
        ).generate_from_frequencies(word_freq)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"  ✗ Error generating word cloud: {e}")
        print(f"  → Attempting fallback method...")
        
        # Fallback: Create a simple bar chart instead
        create_fallback_visualization(word_freq, color_func, title, output_path)

def create_fallback_visualization(word_freq, color_func, title, output_path):
    """Create a bar chart visualization as fallback"""
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    words = [w[0] for w in sorted_words]
    freqs = [w[1] for w in sorted_words]
    
    # Get colors for each word
    colors = [color_func(word) for word in words]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(words))
    ax.barh(y_pos, freqs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fallback_path = output_path.replace('.png', '_barchart.png')
    plt.savefig(fallback_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved fallback visualization: {fallback_path}")

# Generate FPN-A word cloud
fpna_output = os.path.join(OUTPUT_DIR, 'fpna_cognitive_domains_wordcloud.png')
generate_wordcloud(
    fpna_freq,
    fpna_color_func,
    f"Top {TOP_N} Cognitive Domains - FPN-A Dominant\n(size = composite score, darkness = |Cohen's d|)",
    fpna_output
)

# Generate FPN-B word cloud
fpnb_output = os.path.join(OUTPUT_DIR, 'fpnb_cognitive_domains_wordcloud.png')
generate_wordcloud(
    fpnb_freq,
    fpnb_color_func,
    f"Top {TOP_N} Cognitive Domains - FPN-B Dominant\n(size = composite score, darkness = |Cohen's d|)",
    fpnb_output
)

print("\n" + "="*60)
print("WORD CLOUD GENERATION COMPLETE!")
print("="*60)
print(f"Output files:")
print(f"  - {fpna_output}")
print(f"  - {fpnb_output}")