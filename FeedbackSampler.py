import torch
import torch.nn.functional as F
from comfy.samplers import KSampler
import comfy.sample
import comfy.samplers
import comfy.utils
import nodes
import latent_preview
import numpy as np
from PIL import Image

# Try to import scipy for sharpening and noise
try:
    from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Sharpening and Perlin noise will be disabled. Install with: pip install scipy")


class FeedbackSampler:
    """
    A sampler that feeds finished latent back into itself with zoom functionality.
    Creates deforum-style zooming animations through iterative feedback loops.
    Includes LAB color matching to prevent color bleeding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # === Standard KSampler Parameters ===
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # === Animation Parameters ===
                "zoom_value": ("FLOAT", {"default": 0.01, "min": -0.5, "max": 0.5, "step": 0.0001, "round": 0.0001}),
                "iterations": ("INT", {"default": 5, "min": 1, "max": 100}),
                "feedback_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_variation": (["fixed", "increment", "random"], {"default": "fixed"}),
                
                # === Color & Quality Enhancement ===
                "color_coherence": (["None", "LAB", "RGB", "HSV"], {"default": "LAB"}),
                "noise_amount": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "noise_type": (["gaussian", "perlin"], {"default": "perlin"}),
                "sharpen_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_boost": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 1.5, "step": 0.01}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("final_latent", "all_latents")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"
    
    def match_color_histogram(self, source, reference, mode="LAB"):
        """
        Match color histogram of source image to reference image.
        This is the critical function that prevents color bleeding.
        
        Args:
            source: Image to adjust (numpy array HxWx3, values 0-255)
            reference: Target color distribution (numpy array HxWx3, values 0-255)
            mode: Color space for matching ("LAB", "RGB", "HSV")
        
        Returns:
            Color-matched image (numpy array HxWx3, values 0-255)
        """
        if mode == "None":
            return source
        
        # Ensure uint8 type
        source = source.astype(np.uint8)
        reference = reference.astype(np.uint8)
        
        if mode == "LAB":
            # Convert to LAB color space (most perceptually uniform)
            # LAB separates lightness from color, best for preventing color drift
            source_lab = self.rgb_to_lab(source)
            reference_lab = self.rgb_to_lab(reference)
            
            # Match histogram for each channel
            matched_lab = np.zeros_like(source_lab)
            for i in range(3):
                matched_lab[:, :, i] = self.match_histograms(
                    source_lab[:, :, i], 
                    reference_lab[:, :, i]
                )
            
            # Convert back to RGB
            result = self.lab_to_rgb(matched_lab)
            
        elif mode == "HSV":
            # HSV mode - good for maintaining hue consistency
            source_hsv = self.rgb_to_hsv(source)
            reference_hsv = self.rgb_to_hsv(reference)
            
            matched_hsv = np.zeros_like(source_hsv)
            for i in range(3):
                matched_hsv[:, :, i] = self.match_histograms(
                    source_hsv[:, :, i],
                    reference_hsv[:, :, i]
                )
            
            result = self.hsv_to_rgb(matched_hsv)
            
        else:  # RGB
            # Direct RGB matching
            result = np.zeros_like(source)
            for i in range(3):
                result[:, :, i] = self.match_histograms(
                    source[:, :, i],
                    reference[:, :, i]
                )
        
        return result.astype(np.uint8)
    
    def match_histograms(self, source, reference):
        """
        Match histogram of source channel to reference channel.
        Uses cumulative distribution function (CDF) matching.
        """
        # Calculate histograms
        source_values, source_counts = np.unique(source.ravel(), return_counts=True)
        reference_values, reference_counts = np.unique(reference.ravel(), return_counts=True)
        
        # Calculate CDFs
        source_cdf = np.cumsum(source_counts).astype(np.float64)
        source_cdf /= source_cdf[-1]
        
        reference_cdf = np.cumsum(reference_counts).astype(np.float64)
        reference_cdf /= reference_cdf[-1]
        
        # Interpolate to find mapping
        interp_values = np.interp(source_cdf, reference_cdf, reference_values)
        
        # Build lookup table
        lookup = np.zeros(256, dtype=reference.dtype)
        for i, val in enumerate(source_values):
            lookup[val] = interp_values[i]
        
        # Apply lookup table
        return lookup[source]
    
    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space"""
        # Normalize to 0-1
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Apply gamma correction
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(mask, 
                              np.power((rgb_norm + 0.055) / 1.055, 2.4),
                              rgb_norm / 12.92)
        
        # RGB to XYZ
        xyz = np.zeros_like(rgb_linear)
        xyz[:, :, 0] = rgb_linear[:, :, 0] * 0.4124564 + rgb_linear[:, :, 1] * 0.3575761 + rgb_linear[:, :, 2] * 0.1804375
        xyz[:, :, 1] = rgb_linear[:, :, 0] * 0.2126729 + rgb_linear[:, :, 1] * 0.7151522 + rgb_linear[:, :, 2] * 0.0721750
        xyz[:, :, 2] = rgb_linear[:, :, 0] * 0.0193339 + rgb_linear[:, :, 1] * 0.1191920 + rgb_linear[:, :, 2] * 0.9503041
        
        # Normalize by D65 white point
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883
        
        # XYZ to LAB
        mask = xyz > 0.008856
        f = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))
        
        lab = np.zeros_like(xyz)
        lab[:, :, 0] = (116 * f[:, :, 1]) - 16  # L
        lab[:, :, 1] = 500 * (f[:, :, 0] - f[:, :, 1])  # a
        lab[:, :, 2] = 200 * (f[:, :, 1] - f[:, :, 2])  # b
        
        # Scale to 0-255 for histogram matching
        lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0  # L: 0-100 -> 0-255
        lab[:, :, 1] = (lab[:, :, 1] + 128.0)  # a: -128-127 -> 0-255
        lab[:, :, 2] = (lab[:, :, 2] + 128.0)  # b: -128-127 -> 0-255
        
        return np.clip(lab, 0, 255).astype(np.uint8)
    
    def lab_to_rgb(self, lab):
        """Convert LAB back to RGB with proper bounds checking"""
        # Unscale from 0-255
        lab_float = lab.astype(np.float32)
        lab_float[:, :, 0] = lab_float[:, :, 0] * 100.0 / 255.0  # L: 0-255 -> 0-100
        lab_float[:, :, 1] = lab_float[:, :, 1] - 128.0  # a: 0-255 -> -128-127
        lab_float[:, :, 2] = lab_float[:, :, 2] - 128.0  # b: 0-255 -> -128-127
        
        # LAB to XYZ
        fy = (lab_float[:, :, 0] + 16) / 116
        fx = lab_float[:, :, 1] / 500 + fy
        fz = fy - lab_float[:, :, 2] / 200
        
        # Ensure positive values before power operations
        fx = np.maximum(fx, 0.0)
        fy = np.maximum(fy, 0.0)
        fz = np.maximum(fz, 0.0)
        
        mask_x = fx > 0.2068966
        mask_y = fy > 0.2068966
        mask_z = fz > 0.2068966
        
        xyz = np.zeros_like(lab_float)
        xyz[:, :, 0] = np.where(mask_x, np.power(fx, 3), (fx - 16/116) / 7.787)
        xyz[:, :, 1] = np.where(mask_y, np.power(fy, 3), (fy - 16/116) / 7.787)
        xyz[:, :, 2] = np.where(mask_z, np.power(fz, 3), (fz - 16/116) / 7.787)
        
        # Clip XYZ to valid range
        xyz = np.clip(xyz, 0.0, 1.0)
        
        # Denormalize by D65 white point
        xyz[:, :, 0] *= 0.95047
        xyz[:, :, 1] *= 1.00000
        xyz[:, :, 2] *= 1.08883
        
        # XYZ to RGB
        rgb_linear = np.zeros_like(xyz)
        rgb_linear[:, :, 0] = xyz[:, :, 0] *  3.2404542 + xyz[:, :, 1] * -1.5371385 + xyz[:, :, 2] * -0.4985314
        rgb_linear[:, :, 1] = xyz[:, :, 0] * -0.9692660 + xyz[:, :, 1] *  1.8760108 + xyz[:, :, 2] *  0.0415560
        rgb_linear[:, :, 2] = xyz[:, :, 0] *  0.0556434 + xyz[:, :, 1] * -0.2040259 + xyz[:, :, 2] *  1.0572252
        
        # Clip to valid range before gamma correction (CRITICAL!)
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        
        # Apply gamma correction - now safe from negative values
        mask = rgb_linear > 0.0031308
        rgb = np.where(mask,
                      1.055 * np.power(rgb_linear, 1/2.4) - 0.055,
                      12.92 * rgb_linear)
        
        # Final clip and convert
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Safety check for NaN or Inf
        if np.any(np.isnan(rgb)) or np.any(np.isinf(rgb)):
            print("  WARNING: Invalid values detected in LAB->RGB conversion, using fallback")
            return np.zeros_like(rgb, dtype=np.uint8) + 128  # Return gray as fallback
        
        return rgb
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        deltac = maxc - minc
        s = np.where(maxc != 0, deltac / maxc, 0)
        
        rc = np.where(deltac != 0, (maxc - r) / deltac, 0)
        gc = np.where(deltac != 0, (maxc - g) / deltac, 0)
        bc = np.where(deltac != 0, (maxc - b) / deltac, 0)
        
        h = np.zeros_like(r)
        h = np.where((r == maxc), bc - gc, h)
        h = np.where((g == maxc), 2.0 + rc - bc, h)
        h = np.where((b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        
        hsv = np.stack([h, s, v], axis=2)
        return (hsv * 255).astype(np.uint8)
    
    def hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        hsv_norm = hsv.astype(np.float32) / 255.0
        h, s, v = hsv_norm[:, :, 0], hsv_norm[:, :, 1], hsv_norm[:, :, 2]
        
        i = (h * 6.0).astype(np.int32)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        rgb = np.zeros((*h.shape, 3), dtype=np.float32)
        
        mask = (i == 0)
        rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
        mask = (i == 1)
        rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
        mask = (i == 2)
        rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
        mask = (i == 3)
        rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
        mask = (i == 4)
        rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
        mask = (i == 5)
        rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)
        
        return (rgb * 255).astype(np.uint8)
    
    def latent_to_image(self, latent, vae):
        """Convert latent to RGB image for color matching"""
        # VAE decode expects dict with "samples" key
        latent_dict = {"samples": latent}
        decoded = vae.decode(latent_dict["samples"])
        
        # ComfyUI VAE outputs (B, H, W, C) already! 
        # Just take first batch element and convert to numpy
        img = decoded[0].cpu().numpy()  # Already (H, W, C)
        
        # Convert from 0-1 to 0-255
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        
        return img
    
    def image_to_latent(self, image, vae):
        """Convert RGB image back to latent"""
        # Convert from numpy (H, W, C) 0-255 to tensor (B, H, W, C) 0-1
        # ComfyUI VAE expects (B, H, W, C) format!
        img = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add batch dim: (H, W, C) -> (B, H, W, C)
        
        # Move to correct device
        if hasattr(vae, 'device'):
            img_tensor = img_tensor.to(vae.device)
        elif hasattr(vae, 'first_stage_model'):
            img_tensor = img_tensor.to(next(vae.first_stage_model.parameters()).device)
        
        # Encode to latent
        latent = vae.encode(img_tensor)
        return latent
    
    def generate_perlin_noise(self, shape, scale=10, octaves=4):
        """
        Generate Perlin-like noise for organic texture - SIMPLIFIED for performance.
        Creates structured noise instead of random static.
        
        Args:
            shape: (H, W, C) for the noise
            scale: Lower = larger features (default 10)
            octaves: More = more detail layers (default 4)
        """
        if not SCIPY_AVAILABLE:
            print("    [Perlin noise unavailable without scipy, using Gaussian]", flush=True)
            return np.random.randn(*shape).astype(np.float32) * 0.5 + 0.5
        
        H, W, C = shape
        noise = np.zeros(shape, dtype=np.float32)
        
        print(f"    [Generating Perlin noise {H}x{W}x{C}...]", end=" ", flush=True)
        
        for c in range(C):
            channel_noise = np.zeros((H, W), dtype=np.float32)
            
            for octave in range(octaves):
                freq = 2 ** octave
                amp = 1.0 / (2 ** octave)
                
                # Generate random base at lower resolution
                grid_size = max(4, scale // freq)
                grid_h = H // grid_size + 2
                grid_w = W // grid_size + 2
                
                # Generate random values at grid points
                grid_noise = np.random.randn(grid_h, grid_w).astype(np.float32) * amp
                
                # Upsample using bilinear interpolation (much faster than per-pixel)
                upsampled = scipy_zoom(grid_noise, (H / grid_h, W / grid_w), order=1)
                
                # Crop to exact size
                upsampled = upsampled[:H, :W]
                
                channel_noise += upsampled
            
            noise[:, :, c] = channel_noise
        
        # Normalize to 0-1 range
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        print("Done!", flush=True)
        return noise
    
    def apply_noise_pixel(self, image, amount, noise_type="gaussian"):
        """
        Add noise in pixel space (after color coherence).
        This is critical - adding noise BEFORE color coherence gets removed by histogram matching!
        
        Args:
            image: numpy array (H, W, C), values 0-255
            amount: noise strength (0-1)
            noise_type: "gaussian" or "perlin"
        """
        if amount <= 0:
            return image
        
        img_float = image.astype(np.float32)
        
        if noise_type == "perlin":
            # Generate Perlin noise (organic, structured)
            noise = self.generate_perlin_noise(image.shape, scale=8, octaves=4)
            # Scale to -1 to 1 range
            noise = (noise - 0.5) * 2.0
            # Scale by amount (treat amount as intensity)
            noise_scaled = noise * (amount * 30.0)  # 30 is max intensity
        else:
            # Gaussian noise (random static)
            noise = np.random.randn(*image.shape).astype(np.float32)
            noise_scaled = noise * (amount * 15.0)  # 15 is max intensity
        
        # Add noise to image
        noisy = img_float + noise_scaled
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_noise(self, latent, amount):
        """
        Add controlled noise to latent to prevent stagnation.
        This helps maintain detail at low denoise values.
        """
        if amount <= 0:
            return latent
        
        # Generate noise with same shape as latent
        noise = torch.randn_like(latent) * amount
        return latent + noise
    
    def apply_sharpening(self, image, amount):
        """
        Apply unsharp masking to recover detail lost in VAE encode/decode.
        This is critical for maintaining sharpness at low denoise values.
        
        Args:
            image: numpy array (H, W, C), values 0-255
            amount: sharpening strength (0 = no sharpening, 1 = maximum)
        """
        if amount <= 0 or not SCIPY_AVAILABLE:
            return image
        
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Create Gaussian blur (the "blurred" version)
        blurred = gaussian_filter(img_float, sigma=1.0)
        
        # Unsharp mask: original + amount * (original - blurred)
        sharpened = img_float + amount * (img_float - blurred)
        
        # Clip and convert back
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def apply_contrast(self, image, boost):
        """
        Apply contrast adjustment to prevent washed-out colors.
        
        Args:
            image: numpy array (H, W, C), values 0-255
            boost: contrast multiplier (1.0 = no change, >1.0 = more contrast)
        """
        if boost == 1.0:
            return image
        
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Apply contrast around midpoint (127.5)
        midpoint = 127.5
        contrasted = (img_float - midpoint) * boost + midpoint
        
        # Clip and convert back
        return np.clip(contrasted, 0, 255).astype(np.uint8)
    
    def zoom_latent(self, latent, zoom_factor):
        """
        Apply zoom transformation to latent tensor.
        Positive zoom_factor = zoom in (scale up)
        Negative zoom_factor = zoom out (scale down)
        Zero = no change
        """
        if zoom_factor == 0:
            return latent
        
        # Calculate scale factor (1 + zoom means zoom in, 1 - zoom means zoom out)
        scale = 1.0 + zoom_factor
        
        # Get original dimensions
        batch, channels, height, width = latent.shape
        
        # Calculate new dimensions for zoom
        if zoom_factor > 0:  # Zoom in - sample from center
            new_height = int(height / scale)
            new_width = int(width / scale)
            
            # Calculate crop coordinates (center crop)
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            
            # Crop center region
            cropped = latent[:, :, top:top+new_height, left:left+new_width]
            
            # Scale back to original size
            zoomed = F.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)
            
        else:  # Zoom out - scale down and pad
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Scale down
            scaled = F.interpolate(latent, size=(new_height, new_width), mode='bilinear', align_corners=False)
            
            # Create padded tensor
            pad_top = (height - new_height) // 2
            pad_left = (width - new_width) // 2
            pad_bottom = height - new_height - pad_top
            pad_right = width - new_width - pad_left
            
            # Pad to original size (padding with zeros)
            zoomed = F.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        return zoomed
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
               latent_image, denoise, zoom_value, iterations, feedback_denoise, seed_variation,
               color_coherence, noise_amount, noise_type="perlin", sharpen_amount=0.1, contrast_boost=1.0, vae=None):
        """
        Main sampling function with feedback loop, zoom, and color coherence.
        """
        import random
        
        # Check if VAE is available for color coherence
        if color_coherence != "None" and vae is None:
            print("WARNING: Color coherence requested but no VAE provided. Disabling color coherence.")
            color_coherence = "None"
        
        # Store all latents for output
        all_latents = []
        color_reference = None  # Store first frame for color matching
        
        # Get initial latent
        current_latent = latent_image["samples"].clone()
        latent_format = latent_image.copy()
        
        # First iteration with full denoise
        print(f"FeedbackSampler v1.4.3: Starting iteration 1/{iterations} with denoise={denoise}")
        latent_format["samples"] = current_latent
        
        # Sample first iteration
        result = nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_format, denoise=denoise
        )
        
        current_latent = result[0]["samples"]
        all_latents.append(current_latent.clone())
        
        # Store first frame as color reference
        if color_coherence != "None" and vae is not None:
            color_reference = self.latent_to_image(current_latent, vae)
            print(f"FeedbackSampler: Stored Frame 0 as color reference ({color_coherence} mode)")
        
        # Feedback loop iterations
        for i in range(1, iterations):
            # Determine seed for this iteration
            if seed_variation == "fixed":
                iteration_seed = seed
            elif seed_variation == "increment":
                iteration_seed = seed + i
            else:  # random
                iteration_seed = random.randint(0, 0xffffffffffffffff)
            
            print(f"FeedbackSampler: Iteration {i+1}/{iterations} | zoom={zoom_value} | denoise={feedback_denoise} | seed={iteration_seed} | noise={noise_amount}({noise_type}) | sharpen={sharpen_amount}")
            
            # Apply zoom transformation
            zoomed_latent = self.zoom_latent(current_latent, zoom_value)
            
            # CRITICAL: Apply color coherence + enhancements BEFORE generation
            if color_coherence != "None" and vae is not None and color_reference is not None:
                try:
                    print(f"  [1/6] Decoding latent to image...", end=" ", flush=True)
                    # Decode zoomed latent to image
                    current_image = self.latent_to_image(zoomed_latent, vae)
                    print(f"OK ({current_image.shape})", flush=True)
                    
                    print(f"  [2/6] Matching colors ({color_coherence})...", end=" ", flush=True)
                    # Match colors to reference frame
                    matched_image = self.match_color_histogram(current_image, color_reference, color_coherence)
                    print(f"OK", flush=True)
                    
                    # Apply contrast boost to prevent washed-out colors
                    if contrast_boost != 1.0:
                        print(f"  [3/6] Applying contrast boost ({contrast_boost})...", end=" ", flush=True)
                        matched_image = self.apply_contrast(matched_image, contrast_boost)
                        print(f"OK", flush=True)
                    
                    # Apply sharpening to recover detail (CRITICAL for low denoise)
                    if sharpen_amount > 0:
                        print(f"  [4/6] Applying sharpening ({sharpen_amount})...", end=" ", flush=True)
                        matched_image = self.apply_sharpening(matched_image, sharpen_amount)
                        print(f"OK", flush=True)
                    
                    # ADD NOISE IN PIXEL SPACE (after color matching, so it doesn't get removed!)
                    # This is critical for adding detail to flat/solid color areas
                    if noise_amount > 0:
                        print(f"  [5/6] Adding {noise_type} noise ({noise_amount})...", flush=True)
                        matched_image = self.apply_noise_pixel(matched_image, noise_amount, noise_type)
                        print(f"  [5/6] Noise added OK", flush=True)
                    
                    print(f"  [6/6] Encoding image to latent...", end=" ", flush=True)
                    # Encode back to latent
                    matched_latent = self.image_to_latent(matched_image, vae)
                    print(f"OK ({matched_latent.shape})", flush=True)
                    
                    zoomed_latent = matched_latent
                    print(f"  ✓ All enhancements applied successfully", flush=True)
                except Exception as e:
                    import traceback
                    print(f"\n  ✗ ERROR in color/enhancement pipeline: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    print(f"  Continuing without color correction for this frame...", flush=True)
            elif noise_amount > 0:
                # If no color coherence but noise requested, add in latent space as fallback
                zoomed_latent = self.apply_noise(zoomed_latent, noise_amount)
            
            # Prepare for next sampling
            latent_format["samples"] = zoomed_latent
            
            # Sample with feedback denoise value
            result = nodes.common_ksampler(
                model, iteration_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_format, denoise=feedback_denoise
            )
            
            current_latent = result[0]["samples"]
            all_latents.append(current_latent.clone())
        
        # Stack all latents for batch output
        all_latents_stacked = torch.cat(all_latents, dim=0)
        
        # Return final latent and all latents as batch
        final_output = {"samples": current_latent}
        all_output = {"samples": all_latents_stacked}
        
        return (final_output, all_output)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FeedbackSampler": FeedbackSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeedbackSampler": "Feedback Sampler"
}