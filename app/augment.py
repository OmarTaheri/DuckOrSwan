"""GPU data augmentation layer (Keras 3, backend-agnostic ops).

The built-in RandomRotation / RandomZoom layers were the bottleneck of the first training
run (~90 s per epoch: the GPU waited for the CPU). This layer does the same random
geometry with keras.ops on whatever device the batch is on (the GPU during training),
and adds colour jitter + random grayscale so the CNN cannot rely on colour alone
("white bird = swan") and has to learn shape.

At inference (training=False) it returns the input unchanged, so the web app
behaves exactly like a model without augmentation.
"""
import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="duckswan")
class GPUAugment(keras.layers.Layer):
    def __init__(self, flip=True, rotation_deg=20.0, zoom=0.15, translate=0.1,
                 color=True, brightness=0.15, contrast=0.25, saturation=0.35,
                 grayscale_p=0.2, seed=1337, **kwargs):
        super().__init__(**kwargs)
        self.flip = flip
        self.rotation_deg = rotation_deg
        self.zoom = zoom
        self.translate = translate
        self.color = color
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.grayscale_p = grayscale_p
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def _u(self, shape, lo, hi):
        return keras.random.uniform(shape, lo, hi, seed=self.seed_generator)

    def call(self, x, training=None):
        if not training:
            return x
        b = ops.shape(x)[0]
        h, w = ops.shape(x)[1], ops.shape(x)[2]

        # 1) random horizontal flip
        if self.flip:
            mask = self._u((b, 1, 1, 1), 0.0, 1.0) < 0.5
            x = ops.where(mask, ops.flip(x, axis=2), x)

        # 2) random rotation + zoom + translation as one affine transform per image.
        #    The transform maps each output pixel (x, y) to the input pixel it samples.
        r = self.rotation_deg * math.pi / 180.0
        theta = self._u((b,), -r, r)
        s = self._u((b,), 1.0 - self.zoom, 1.0 + self.zoom)
        tx = self._u((b,), -self.translate, self.translate)
        ty = self._u((b,), -self.translate, self.translate)
        cos, sin = ops.cos(theta) * s, ops.sin(theta) * s
        if keras.backend.backend() == "torch":
            # fast path: native GPU kernels (normalised coordinates, square images)
            import torch.nn.functional as F

            mat = ops.stack([ops.stack([cos, -sin, 2 * tx], 1), ops.stack([sin, cos, 2 * ty], 1)], 1)
            nchw = x.permute(0, 3, 1, 2)
            grid = F.affine_grid(mat, list(nchw.shape), align_corners=False)
            x = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="reflection",
                              align_corners=False).permute(0, 2, 3, 1)
        else:
            tx, ty = tx * float(w), ty * float(h)
            cx, cy = (float(w) - 1.0) / 2.0, (float(h) - 1.0) / 2.0
            zeros = ops.zeros_like(cos)
            transform = ops.stack([
                cos, -sin, cx - cos * cx + sin * cy + tx,
                sin, cos, cy - sin * cx - cos * cy + ty,
                zeros, zeros,
            ], axis=1)
            x = ops.image.affine_transform(x, transform, interpolation="bilinear", fill_mode="reflect")

        # 3) colour jitter + random grayscale (fights the "white = swan" colour shortcut)
        if self.color:
            x = x + self._u((b, 1, 1, 1), -self.brightness, self.brightness)
            mean = ops.mean(x, axis=(1, 2, 3), keepdims=True)
            x = (x - mean) * self._u((b, 1, 1, 1), 1.0 - self.contrast, 1.0 + self.contrast) + mean
            gray = ops.sum(x * ops.convert_to_tensor([0.299, 0.587, 0.114], dtype=x.dtype),
                           axis=-1, keepdims=True)
            sat = self._u((b, 1, 1, 1), 1.0 - self.saturation, 1.0 + self.saturation)
            sat = ops.where(self._u((b, 1, 1, 1), 0.0, 1.0) < self.grayscale_p, 0.0, sat)
            x = gray + (x - gray) * sat
            x = ops.clip(x, 0.0, 1.0)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(flip=self.flip, rotation_deg=self.rotation_deg, zoom=self.zoom,
                   translate=self.translate, color=self.color, brightness=self.brightness,
                   contrast=self.contrast, saturation=self.saturation,
                   grayscale_p=self.grayscale_p, seed=self.seed)
        return cfg
