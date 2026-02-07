# ComfyUI Feedback Sampler
**Work in Progress** 

A custom ComfyUI sampler for creating **Deforum-style zoom animations** through iterative feedback loop.

![Demo](demo.gif)

> **Example use with SDXLTurbo model** 

## Features

- **Feedback Loop Animation** - Iterative sampling with zoom, pan, and rotate
- **ControlNet Sequence Support** - Automatically detects batched hint images and applies per-frame ControlNet conditioning with automatic looping
- **LAB Color Coherence** - Deforum-inspired color matching prevents chromatic drift
- **Anti-Blur Sharpening** - Maintains detail at low denoise values
- **Noise Injection** - Prevents stagnation (Gaussian/Perlin)
- **Contrast Boost** - Keeps colors vibrant
- **Batch Output** - All frames as sequence for video creation

**Inspired by [Deforum Stable Diffusion](https://colab.research.google.com/github/deforum-art/deforum-stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb)**

## Compatibility

Works with:
- Stable Diffusion 1.5
- SDXL
- Flux models
- Basically any model thats compatible with native Ksampler
- Turbo/lighting models are recommended for faster iteration

## Installation

1. Clone into `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/pizurny/Comfyui-FeedbackSampler
```

2. Install dependencies (optional, for sharpening and Perlin noise):
```bash
pip install -r requirements.txt
```

Or for portable/embedded Python installation (run in `ComfyUI_windows_portable` folder):
```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Comfyui-FeedbackSampler\requirements.txt
```

3. Restart ComfyUI

Find node at: **Add Node → sampling → custom → Feedback Sampler**

## Quick Start

1. Connect model, CLIP conditioning, latent, and **VAE** (required)
2. Set parameters:
   - `zoom_value`: 0.005 (zoom speed, + = in, - = out)
   - `iterations`: 30 (frame count)
   - `feedback_denoise`: 0.3 (lower = more coherence)
   - `color_coherence`: LAB (prevents color drift)
3. Connect `all_latents` → VAE Decode → Video Combine

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zoom_value` | 0.01 | Zoom speed (-0.5 to 0.5) |
| `translate_x` | 0.0 | Horizontal pan in pixels/frame (-100 to 100). Positive = camera pans right |
| `translate_y` | 0.0 | Vertical pan in pixels/frame (-100 to 100). Positive = camera pans down |
| `angle` | 0.0 | Rotation in degrees/frame (-180 to 180). Positive = counterclockwise |
| `border_mode` | zeros | Fill for out-of-bounds areas (zeros/border/reflection) |
| `iterations` | 5 | Number of frames |
| `feedback_denoise` | 0.3 | Strength per frame (0-1) |
| `color_coherence` | LAB | Color matching (None/LAB/RGB/HSV) |
| `noise_amount` | 0.02 | Prevents stagnation (0-1) |
| `sharpen_amount` | 0.1 | Anti-blur strength (0-1) |

## Recommended Settings

**Smooth Animation:**
```
feedback_denoise: 0.25-0.35
zoom_value: 0.001-0.005
color_coherence: LAB
sharpen_amount: 0.15-0.25
```

**Fast Zoom:**
```
feedback_denoise: 0.35-0.45
zoom_value: 0.01-0.02
color_coherence: LAB
sharpen_amount: 0.1-0.2
```

**Pan Flythrough:**
```
zoom_value: 0.005
translate_x: 5-15
angle: 0
border_mode: reflection
feedback_denoise: 0.3-0.4
```

**Spiral Zoom:**
```
zoom_value: 0.02
angle: 1-3
translate_x: 0
border_mode: border
feedback_denoise: 0.3
```

**Slow Rotate:**
```
zoom_value: 0
angle: 0.5-2
border_mode: reflection
feedback_denoise: 0.25-0.35
```

## ControlNet Sequences

The Feedback Sampler automatically detects batched ControlNet hint images and applies them per-frame:

1. Load your ControlNet image sequence (e.g. using Load Image Batch)
2. Connect the batched images to your ControlNet Apply node as usual
3. Connect the ControlNet conditioning to the Feedback Sampler's positive input
4. The sampler will use one hint image per iteration, looping automatically if iterations > frames

This enables guided animations where each frame follows a different ControlNet reference — useful for motion-guided zoom animations or style sequences.

## Troubleshooting

- **Black frames:** Connect VAE to node
- **Blurry:** Increase `sharpen_amount` to 0.2+
- **Color bleeding:** Use `color_coherence: LAB`
- **Too much change:** Lower `feedback_denoise` to 0.2-0.3


## License

MIT License - See [LICENSE](LICENSE)

---

**Example workflows included:**   
`Feedback_Sampler_Flux.json`  
`Feedback_Sampler_SD1.5.json`  
`Feedback_Sampler_SD1.5_custom_startframe.json`  
`Feedback_Sampler_SDXLturbo.json`  
