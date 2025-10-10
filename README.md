# ComfyUI Feedback Sampler

A custom ComfyUI node that creates **Deforum-style zoom animations** through iterative feedback loops with advanced anti-blur and color coherence technology.

![Version](https://img.shields.io/badge/version-1.3.0-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)

## ✨ Features

- 🔄 **Feedback Loop Sampling** - Each frame becomes input for the next
- 🔍 **Zoom In/Out Animation** - Smooth camera movement with configurable speed
- 🎨 **LAB Color Coherence** - Prevents color bleeding and chromatic artifacts (Deforum-inspired)
- ⚡ **Anti-Blur Sharpening** - Maintains detail at low denoise values using unsharp masking
- 🎲 **Noise Injection** - Prevents stagnation and adds detail variety
- 🌈 **Contrast Boost** - Keeps colors vibrant across iterations
- 🎯 **Seed Control** - Fixed, incremental, or random for different effects
- 📹 **Batch Output** - All frames exported as a batch for easy video creation

## 📦 Installation

### Method 1: Manual Install

1. Navigate to your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes/
```

2. Create the node folder:
```bash
mkdir comfyui-feedback-sampler
cd comfyui-feedback-sampler
```

3. Copy these files:
   - `FeedbackSampler.py`
   - `__init__.py`

4. Install dependencies:
```bash
pip install scipy
```

5. Restart ComfyUI

### Method 2: Git Clone
```bash
cd ComfyUI/custom_nodes/
git clone <your-repo-url> comfyui-feedback-sampler
pip install scipy
```

## 🎮 Usage

### Basic Setup

1. Add **Feedback Sampler (Zoom + Anti-Blur)** node to your workflow
2. Connect:
   - Model → model
   - CLIP conditioning → positive/negative
   - Empty Latent or image latent → latent_image
   - **VAE → vae** (REQUIRED for color coherence)
3. Connect outputs:
   - `all_latents` → VAE Decode → Video output
   - `final_latent` → For the last frame only

### Node Location
```
Add Node → sampling → custom → Feedback Sampler (Zoom + Anti-Blur)
```

## ⚙️ Parameters

### Core Sampling
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **steps** | 20 | 1-10000 | Sampling steps per frame |
| **cfg** | 8.0 | 0-100 | Classifier-free guidance scale |
| **denoise** | 1.0 | 0-1 | First frame denoising strength |
| **feedback_denoise** | 0.3 | 0-1 | Subsequent frames denoising |

### Animation
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **zoom_value** | 0.01 | -0.5 to 0.5 | Zoom speed (+ = in, - = out, 0 = none) |
| **iterations** | 5 | 1-100 | Number of frames to generate |

### Color & Quality
| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **color_coherence** | LAB | None/LAB/RGB/HSV | Color matching mode (LAB recommended) |
| **noise_amount** | 0.02 | 0-0.1 | Noise injection to prevent stagnation |
| **sharpen_amount** | 0.1 | 0-1 | Unsharp masking strength (anti-blur) |
| **contrast_boost** | 1.0 | 0.8-1.5 | Contrast multiplier (prevents washed-out colors) |
| **seed_variation** | fixed | fixed/increment/random | How seed changes per frame |

## 🎯 Recommended Settings

### Smooth, Coherent Animation (Low Blur)
```yaml
feedback_denoise: 0.25-0.35   # Low for coherence
zoom_value: 0.001-0.005       # Subtle zoom
noise_amount: 0.02-0.03       # Moderate detail variety
sharpen_amount: 0.15-0.25     # Recover sharpness
contrast_boost: 1.05-1.10     # Keep colors vibrant
color_coherence: LAB          # Prevent color drift
seed_variation: fixed         # Maximum stability
```

### Fast Zoom Animation
```yaml
feedback_denoise: 0.35-0.45   # Higher for border generation
zoom_value: 0.01-0.02         # Faster zoom
noise_amount: 0.03-0.05       # More variety needed
sharpen_amount: 0.1-0.2       # Moderate sharpening
contrast_boost: 1.1-1.15      # Strong contrast
color_coherence: LAB          # Essential for stability
seed_variation: increment     # Smooth variation
```

### Creative/Experimental
```yaml
feedback_denoise: 0.5-0.7     # High creativity
zoom_value: 0.005-0.015       # Moderate zoom
noise_amount: 0.05-0.08       # High variety
sharpen_amount: 0.05-0.15     # Light sharpening
contrast_boost: 1.0-1.05      # Natural contrast
color_coherence: HSV/RGB      # Different color effects
seed_variation: random        # Maximum diversity
```

## 🔧 Key Features Explained

### LAB Color Coherence
Based on Deforum's color matching system. Prevents the chromatic artifacts and color bleeding that naturally occur in feedback loops. LAB color space is most perceptually uniform and prevents the "toxic color" effect that appears after 10-15 iterations.

**When to use:**
- ✅ Always recommended for animations longer than 5 frames
- ✅ Essential when using low denoise values
- ⚠️ Requires VAE to be connected

### Anti-Blur Sharpening
Unsharp masking that recovers detail lost during VAE encode/decode cycles. This is the **critical feature** that allows low denoise values (0.25-0.35) without blur.

**How it works:**
1. Creates slightly blurred version
2. Subtracts from original
3. Amplifies the difference (edges/details)

### Noise Injection
Adds controlled noise before generation to prevent the feedback loop from getting "stuck" in repetitive patterns. Correlated inversely with denoise strength.

## 🐛 Troubleshooting

### Black Frames After First Frame
- **Cause:** VAE not connected
- **Fix:** Connect a VAE to the optional `vae` input

### Blurry Animation
- **Solution:** Increase `sharpen_amount` to 0.15-0.25
- Also try increasing `noise_amount` to 0.03-0.05

### Too Much Change Between Frames
- **Solution:** Lower `feedback_denoise` to 0.2-0.3
- Set `seed_variation` to "fixed"
- Increase `color_coherence` strength

### Color Bleeding / Toxic Colors
- **Solution:** Set `color_coherence` to "LAB"
- Ensure VAE is connected
- Try `contrast_boost` of 1.05-1.10

### Sharpening Not Working
- **Cause:** scipy not installed
- **Fix:** `pip install scipy`

## 🎬 Example Workflow

```
Empty Latent Image (512x512)
    ↓
Feedback Sampler
    ├─ model: checkpoint model
    ├─ positive: CLIP conditioning
    ├─ negative: CLIP conditioning
    ├─ vae: VAE model ⚠️ REQUIRED
    ├─ zoom_value: 0.005
    ├─ iterations: 30
    ├─ feedback_denoise: 0.3
    ├─ color_coherence: LAB
    └─ sharpen_amount: 0.2
    ↓
all_latents output → VAE Decode (batch)
    ↓
Save Image / Video Combine
```

## 📝 Technical Details

- **Version:** 1.3.0
- **Developed with:** Claude Sonnet 4.5
- **Inspired by:** Deforum Stable Diffusion
- **Color Space:** LAB (L*a*b* / CIELAB)
- **Sharpening Method:** Unsharp masking with Gaussian blur

## 🙏 Credits

- Inspired by [Deforum Stable Diffusion](https://github.com/deforum-art/sd-webui-deforum)
- LAB color coherence based on Deforum's histogram matching implementation
- Anti-blur techniques adapted from Deforum's animation pipeline

## 📄 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Issues and pull requests welcome! This is an educational project developed through conversation.

---

**Need help?** Open an issue with:
- ComfyUI version
- Your node settings
- Console output
- Example of the problem