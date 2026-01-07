# Mondrian Boogie Woogie Art Generator

A Python tool that recreates Piet Mondrian's "Broadway Boogie Woogie" style by analyzing a reference image, extracting its dominant color palette, and remapping every pixel to the nearest palette color. Optionally adds Mondrian-inspired colored tiles on light regions and along detected grid lanes to evoke the rhythmic, jazz-influenced composition of the original artwork.

## Features

- **Palette Extraction**: Automatically extracts dominant colors from reference images using median-cut quantization
- **Pixel-Perfect Remapping**: Maps every pixel to the nearest palette color without dithering for crisp, geometric results
- **Boogie Tiles**: Adds clustered micro-tiles on light background regions to create rhythmic patterns
- **Lane Detection**: Identifies grid-like structures and places tile runs along detected lanes
- **Diamond Masking**: Automatically detects and preserves the central diamond composition
- **Color Weighting**: Biases tile placement toward specific colors (yellow, red, blue, black)
- **Deterministic Mode**: Use seeds for reproducible results or random generation
- **Light Region Masking**: Export a binary mask of light/white/gray regions for inspection or reuse

## Prerequisites

- Python 3.13+ (virtualenv already present at `venv/`)
- Pillow (bundled inside `venv`)

## Installation

If not using the bundled virtualenv:

```bash
pip install Pillow
```

## Basic Usage

### With Boogie Tiles (Deterministic Generation)

Use a seed for **reproducible tile placement** (same output every time):

```bash
python mondrian_boogie_woogie.py \
  --input art.JPG \
  --output recreated.png \
  --add-boogie-tiles \
  --tile-size 150 \
  --tile-density 0.38 \
  --boogie-mask boogie_mask.png \
  --tile-seed 12345 \
  --light-mask light_mask.png
```

**Note**: With `--tile-seed`, the same seed produces identical tile patterns for consistent, reproducible results.

### With Lane Tiles

Add tile runs along detected grid lines:

```bash
python mondrian_boogie_woogie.py \
  --input art.JPG \
  --output recreated.png \
  --add-lane-tiles \
  --tile-size 10 \
  --lane-density 0.20 \
  --lane-run-length 5 \
  --lane-mask lane_mask.png
```

### Palette Extraction

| Option | Description | Default |
|--------|-------------|---------|
| `--max-colors N` | Number of dominant colors to extract | 12 |
| `--max-dim PX` | Resize longest side to this dimension | None |

### Boogie Tiles

| Option | Description | Default |
|--------|-------------|---------|
| `--add-boogie-tiles` | Enable clustered tile placement | False |
| `--tile-size PX` | Side length of tiles in pixels | 10 |
| `--tile-density FLOAT` | Probability of placing tiles (0-1) | 0.08 |
| `--tile-brightness-threshold N` | Min brightness for light background | 220 |
| `--tile-gray-range N` | Max channel spread for gray detection | 25 |
| `--tile-seed N` | Seed for deterministic placement (omit for random) | Random |
| `--boogie-mask PATH` | Save mask of painted boogie tiles | None |

### Lane Tiles

| Option | Description | Default |
|--------|-------------|---------|
| `--add-lane-tiles` | Enable lane-following tile runs | False |
| `--lane-density FLOAT` | Probability of starting lane run (0-1) | 0.20 |
| `--lane-run-length N` | Nominal tiles per run (±1 variance) | 5 |
| `--lane-mask PATH` | Save lane detection + placement mask | None |

### Color Weighting

Control the probability distribution for tile colors:

| Option | Description | Default |
|--------|-------------|---------|
| `--yellow-weight FLOAT` | Bias weight for yellow tiles | 3.0 |
| `--red-weight FLOAT` | Bias weight for red tiles | 2.0 |
| `--blue-weight FLOAT` | Bias weight for blue tiles | 1.0 |
| `--black-weight FLOAT` | Bias weight for black/dark tiles | 0.5 |

### Diamond Masking

| Option | Description | Default |
|--------|-------------|---------|
| `--disable-diamond-mask` | Allow tiles outside detected diamond | False |
| `--diamond-white-threshold N` | Brightness for diamond boundary detection | 245 |

### Light Region Mask

Export a binary mask of light / white / gray regions using the same thresholds that gate boogie/lane tiles:

| Option | Description | Default |
|--------|-------------|---------|
| `--light-mask PATH` | Save mask of pixels that are both bright and near-gray | None |
| `--tile-brightness-threshold N` | Min brightness for light detection | 220 |
| `--tile-gray-range N` | Max channel spread for gray detection | 25 |

## How It Works

### 1. Palette Extraction
The script uses Pillow's median-cut quantization to identify the most dominant colors in your reference image, typically extracting 12 colors that represent the primary hues, blacks, and off-whites.

### 2. Pixel Remapping
Every pixel is mapped to its nearest palette color using Euclidean distance in RGB space, with no dithering. This creates clean, flat color blocks reminiscent of Mondrian's style.

### 3. Diamond Detection (Optional)
Automatically identifies the central painted region by detecting non-white content and fitting a rotated square (diamond) around it. Tiles are confined to this region unless disabled.

### 4. Tile Placement

**Boogie Tiles**: Scans the image on a grid and places short clusters (1-3 tiles) of colored squares on light background regions. Clusters can be horizontal or vertical, creating rhythmic patterns.

**Lane Tiles**: Detects color discontinuities (edges) to approximate grid lines, then places runs of tiles along these detected lanes in straight sequences.

### 5. Color Biasing
Tiles are not selected uniformly from the palette. Colors closer to yellow, red, blue, or black targets receive higher probability weights, which you can customize via command-line flags.

## Random vs Deterministic Tile Generation

### Random Generation (Default)
When **no seed** is provided, tile placement is completely random:
- Each run produces different tile patterns
- Great for exploring variations and finding interesting compositions
- Use when you want artistic freedom and variety

### Deterministic Generation (With Seed)
When a **seed** is specified via `--tile-seed`:
- Same seed always produces identical results
- Perfect for reproducibility in production or testing
- Allows you to "lock in" a pattern you like
- Anyone with the same seed and parameters gets the same output

**Example comparison:**
```bash
# Random - different each time
python mondrian_boogie_woogie.py --input art.JPG --output random.png --add-boogie-tiles

# Deterministic - always the same
python mondrian_boogie_woogie.py --input art.JPG --output fixed.png --add-boogie-tiles --tile-seed 12345
```

## Output Files

- **Main Output**: The recreated artwork with optional tile augmentation
- **Palette Strip**: Horizontal visualization showing extracted colors in order
- **Boogie Mask**: Binary mask showing where boogie tiles were placed (white = painted)
- **Lane Mask**: Visualization of detected lanes and placed lane tiles
- **Light Region Mask**: Binary mask of bright, near-gray areas in the source

## Tips

- Start with default parameters and adjust incrementally
- Use `--tile-seed` when you want reproducible results for testing or production
- Omit `--tile-seed` for random exploration and creative experimentation
- Higher `--tile-density` creates busier, more energetic compositions
- Larger `--tile-size` produces bolder, more visible tile patterns
- Combine both tile modes for rich, layered effects
- Save masks to understand where tiles are being placed
- Adjust color weights to match your artistic vision

## Technical Notes

- The script handles very large images via `Image.MAX_IMAGE_PIXELS = None`
- Use `--max-dim` to resize before processing for faster execution
- Tile placement respects the diamond boundary by default for authentic composition
- Random number generation uses a single seeded RNG for full determinism when `--tile-seed` is provided
- Color distance calculations use Euclidean RGB space
