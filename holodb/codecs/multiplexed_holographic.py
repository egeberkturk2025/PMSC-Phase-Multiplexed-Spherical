"""
multiplexed_holographic.py — Phase-Multiplexed Spherical Codec
===============================================================
HoloDB'nin 5. ve en gelişmiş codec'i.

Fiziksel Analoji:
    İç içe dönen küreler (planetaryum modeli):
    - Yarıçap R  = Genlik (tüm görseller ORTAK)
    - Dönüş hızı ω = Frekans (tüm görseller ORTAK)
    - Başlangıç açısı φ = Faz (her görsel ÖZEL)

Ana Sınıf:
    MultiplexedHolographicCodec

Temel Metotlar:
    encode_batch(images, use_alpha, differential)
        N görsel → MultiplexedPayload (ortak taban + faz slotları)

    decode_single(payload, idx)
        Tek görsel decode, bit-perfect garantili

    encode_lossy_only(images)
        Delta yamasız ultra-hızlı mod (~12x sıkıştırma)

    add_image(payload, image)
        Mevcut payload'a incremental ekleme

    rebase(payload)
        Tüm görselleri yeniden decode edip ortak tabanı optimize et

Veri Yapıları:
    MultiplexedPayload  — Ortak taban + görsel listesi
    ImageSlot           — Tek görselin faz + delta verileri

Bağımlılıklar:
    numpy, zlib (stdlib)

Bölüm: wiki/18_faz_coklamali_kure_motoru.md
Sprint Geçmişi: A (core) → B (format) → C (alpha+diff) → D (incr) → E (entegrasyon)

Telif Hakkı (c) 2026 Ege Berk Türk — Tüm Hakları Saklıdır.
Ticari kullanımı kesinlikle yasaktır.
"""

import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ────────────────────────── Zigzag Helpers ──────────────────────────


def zigzag_encode(values: np.ndarray, max_val: int) -> np.ndarray:
    """
    Signed int → unsigned zigzag mapping.
    0→0, -1→1, 1→2, -2→3, 2→4 ...

    Dynamically selects uint8 or uint16 to prevent overflow.
    """
    v = values.astype(np.int16)
    encoded = np.where(v >= 0, 2 * v, -2 * v - 1)
    if max_val <= 127:
        return encoded.astype(np.uint8)
    else:
        return encoded.astype(np.uint16)


def zigzag_decode(values: np.ndarray) -> np.ndarray:
    """Unsigned zigzag → signed int."""
    v = values.astype(np.int16)
    return np.where(v % 2 == 0, v // 2, -(v + 1) // 2).astype(np.int16)


# ────────────────────────── Payload Dataclass ──────────────────────────


@dataclass
class ImageSlot:
    """Per-image payload inside a multiplexed hologram."""
    phase_comp: bytes          # zlib-compressed quantized phase key (uint8 × K)
    patch_comp: bytes          # zlib-compressed zigzag delta patch
    max_delta: int             # maximum absolute delta value
    delta_is_uint8: bool       # True if zigzag values fit in uint8
    zero_pct: float            # percentage of zero-valued deltas
    alpha: Optional[np.ndarray] = None  # float16 × K (Per-image amplitude scaling vector)
    differential: bool = False  # True if phase is stored differentially


@dataclass
class MultiplexedPayload:
    """Full multiplexed hologram payload."""
    h: int
    w: int
    keep_ratio: float
    n_kept: int                          # number of kept frequencies
    shared_amp_comp: bytes               # zlib-compressed shared amplitude (float32 × K)
    shared_idx_comp: bytes               # zlib-compressed delta-encoded indices (uint32 × K)
    sorted_idx: np.ndarray               # raw sorted indices for in-memory decode
    images: List[ImageSlot] = field(default_factory=list)
    lossy: bool = False

    @property
    def n_images(self) -> int:
        return len(self.images)

    @n_images.setter
    def n_images(self, value):
        pass


# ────────────────────────── Codec Class ──────────────────────────


class MultiplexedHolographicCodec:
    """
    Phase-Multiplexed Spherical Codec.

    Stores N images in a single shared frequency-domain base.
    Each image is identified by its unique phase key (starting angles of spheres).
    Bit-perfect reconstruction is guaranteed via zigzag-encoded delta patches.

    Parameters
    ----------
    keep_ratio : float
        Fraction of top-K frequency bins to keep (default 0.02 = 2%).
    lossy : bool
        If True, skip delta patches for ultra-high compression (not bit-perfect).
    """

    def __init__(self, keep_ratio: float = 0.02, lossy: bool = False):
        self.keep_ratio = keep_ratio
        self.lossy = lossy

    # ─────────────── Auto keep_ratio ───────────────

    @staticmethod
    def auto_keep_ratio(images: List[np.ndarray]) -> float:
        """
        Automatically select keep_ratio based on inter-image phase similarity.

        Phase correlation accurately measures content similarity:
        - Similar images (video frames): phase_corr ≈ 0.9 → aggressive pruning (0.01)
        - Different images (random noise): phase_corr ≈ 0.0 → conservative pruning (0.05)

        Note: Amplitude correlation is unreliable because even random noise images
        share similar amplitude statistics (~0.93 correlation).
        """
        if len(images) < 2:
            return 0.02

        # Compute phase spectra
        phases = []
        for img in images:
            freq = np.fft.fft2(img.astype(np.float64))
            phases.append(np.angle(np.fft.fftshift(freq)).flatten())

        # Average pairwise phase correlation
        correlations = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                corr = np.corrcoef(phases[i], phases[j])[0, 1]
                correlations.append(corr)

        avg_corr = float(np.mean(correlations))

        if avg_corr > 0.8:
            return 0.01
        elif avg_corr > 0.3:
            return 0.02
        else:
            return 0.05

    # ─────────────── Batch Encode ───────────────

    def encode_batch(
        self,
        images: List[np.ndarray],
        use_alpha: bool = True,
        differential: bool = False
    ) -> MultiplexedPayload:
        """
        Encode N grayscale images into a single multiplexed hologram.

        Parameters
        ----------
        images : list of np.ndarray
            List of 2D uint8 arrays, all must have the same (h, w) shape.
        use_alpha : bool
            Enable per-image amplitude scaling vector (reduces delta patch size).
        differential : bool
            Enable differential phase key encoding (for similar sequential images).

        Returns
        -------
        MultiplexedPayload
        """
        if not images:
            raise ValueError("images list must not be empty")

        h, w = images[0].shape
        for idx, img in enumerate(images):
            if img.shape != (h, w):
                raise ValueError(
                    f"Image {idx} shape {img.shape} differs from first image shape ({h}, {w})"
                )

        # 1. FFT all images
        amp_list = []
        phase_list = []

        for img in images:
            freq = np.fft.fft2(img.astype(np.float64))
            fs = np.fft.fftshift(freq)
            amp_list.append(np.abs(fs))
            phase_list.append(np.angle(fs))

        # 2. Shared amplitude = mean of all amplitudes
        shared_amp = np.mean(amp_list, axis=0)

        # 3. Top-K sparsification
        flat_amp = shared_amp.flatten()
        n_keep = max(1, int(len(flat_amp) * self.keep_ratio))
        top_idx = np.argpartition(flat_amp, -n_keep)[-n_keep:]

        mask_2d = np.zeros_like(shared_amp, dtype=bool)
        mask_2d.flat[top_idx] = True

        shared_amp_vals = shared_amp[mask_2d].astype(np.float32)
        sorted_idx = np.sort(top_idx).astype(np.uint32)
        delta_idx = np.diff(sorted_idx, prepend=0).astype(np.uint32)

        # Compress shared base
        comp_shared_amp = zlib.compress(shared_amp_vals.tobytes(), 9)
        comp_shared_idx = zlib.compress(delta_idx.tobytes(), 9)

        # 4. Per-image phase key + delta patch
        image_slots = []
        prev_phase_deq = None

        for i in range(len(images)):
            img_phase = phase_list[i][mask_2d]
            
            # Per-Image Amplitude Scaling Vector
            alpha_k = None
            if use_alpha:
                img_amp_vals = amp_list[i][mask_2d].astype(np.float32)
                # Avoid division by zero warning
                safe_shared = np.where(shared_amp_vals > 1e-10, shared_amp_vals, 1.0)
                alpha_k = np.where(
                    shared_amp_vals > 1e-10,
                    img_amp_vals / safe_shared,
                    1.0
                ).astype(np.float16)

            # Phase key encoding (differential or absolute)
            is_diff = False
            if differential and i > 0 and prev_phase_deq is not None:
                # Phase difference: phi_k - prev_reconstructed_phase
                phase_diff = img_phase - prev_phase_deq
                # Normalize to [-π, π]
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                phase_q = np.uint8(
                    np.clip((phase_diff + np.pi) / (2 * np.pi) * 255, 0, 255)
                )
                
                # De-quantize phase for local reconstruction & feedback tracking
                phase_diff_deq = phase_q.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi
                phase_deq = (phase_diff_deq + prev_phase_deq + np.pi) % (2 * np.pi) - np.pi
                prev_phase_deq = phase_deq
                is_diff = True
            else:
                phase_q = np.uint8(
                    np.clip((img_phase + np.pi) / (2 * np.pi) * 255, 0, 255)
                )
                phase_deq = phase_q.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi
                prev_phase_deq = phase_deq
                
            comp_phase = zlib.compress(phase_q.tobytes(), 9)

            if self.lossy:
                # Lossy mode: no delta patch
                image_slots.append(ImageSlot(
                    phase_comp=comp_phase,
                    patch_comp=b'',
                    max_delta=0,
                    delta_is_uint8=True,
                    zero_pct=0.0,
                    alpha=alpha_k,
                    differential=is_diff
                ))
            else:
                # Bit-perfect mode: compute delta patch
                # Reconstruct frequency spectrum using effective amplitude (scaled if alpha exists)
                eff_amp_vals = shared_amp_vals
                if alpha_k is not None:
                    eff_amp_vals = shared_amp_vals * alpha_k.astype(np.float32)

                reconstructed_freq = np.zeros(h * w, dtype=np.complex128)
                reconstructed_freq[sorted_idx] = eff_amp_vals * np.exp(1j * phase_deq)

                recon_freq_2d = np.fft.ifftshift(reconstructed_freq.reshape(h, w))
                recon = np.clip(
                    np.round(np.fft.ifft2(recon_freq_2d).real), 0, 255
                ).astype(np.uint8)

                delta = images[i].astype(np.int16) - recon.astype(np.int16)
                max_delta = int(np.max(np.abs(delta)))

                zigzag = zigzag_encode(delta.flatten(), max_delta)
                comp_patch = zlib.compress(zigzag.tobytes(), 9)

                image_slots.append(ImageSlot(
                    phase_comp=comp_phase,
                    patch_comp=comp_patch,
                    max_delta=max_delta,
                    delta_is_uint8=(max_delta <= 127),
                    zero_pct=float(np.sum(delta == 0) / delta.size * 100),
                    alpha=alpha_k,
                    differential=is_diff
                ))

        return MultiplexedPayload(
            h=h,
            w=w,
            keep_ratio=self.keep_ratio,
            n_kept=n_keep,
            shared_amp_comp=comp_shared_amp,
            shared_idx_comp=comp_shared_idx,
            sorted_idx=sorted_idx,
            images=image_slots,
            lossy=self.lossy,
        )

    # ─────────────── Single Decode ───────────────

    @staticmethod
    def decode_single(payload: MultiplexedPayload, idx: int) -> np.ndarray:
        """
        Decode a single image from a multiplexed payload.

        Parameters
        ----------
        payload : MultiplexedPayload
        idx : int
            Index of the image to decode.

        Returns
        -------
        np.ndarray
            Reconstructed 2D uint8 image.
        """
        if idx < 0 or idx >= len(payload.images):
            raise IndexError(
                f"Image index {idx} out of range [0, {len(payload.images)})"
            )

        h, w = payload.h, payload.w
        sorted_idx = payload.sorted_idx

        # Decompress shared amplitude
        amp_raw = zlib.decompress(payload.shared_amp_comp)
        shared_amp_vals = np.frombuffer(amp_raw, dtype=np.float32)

        # Accumulate phase: resolve differential phase key if differential is enabled
        phase_deq = None
        for i in range(idx + 1):
            slot_i = payload.images[i]
            phase_raw_i = zlib.decompress(slot_i.phase_comp)
            phase_q_i = np.frombuffer(phase_raw_i, dtype=np.uint8)
            phase_val_i = phase_q_i.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi
            
            if i == 0:
                phase_deq = phase_val_i
            else:
                if slot_i.differential:
                    phase_deq = (phase_deq + phase_val_i + np.pi) % (2 * np.pi) - np.pi
                else:
                    phase_deq = phase_val_i

        # Apply Per-Image Amplitude Scaling Vector (if present)
        img_slot = payload.images[idx]
        eff_amp_vals = shared_amp_vals
        if img_slot.alpha is not None:
            eff_amp_vals = shared_amp_vals * img_slot.alpha.astype(np.float32)

        # Reconstruct frequency spectrum
        reconstructed_freq = np.zeros(h * w, dtype=np.complex128)
        reconstructed_freq[sorted_idx] = eff_amp_vals * np.exp(1j * phase_deq)

        # iFFT → lossy reconstruction
        recon_freq_2d = np.fft.ifftshift(reconstructed_freq.reshape(h, w))
        recon = np.clip(
            np.round(np.fft.ifft2(recon_freq_2d).real), 0, 255
        ).astype(np.uint8)

        if payload.lossy or not img_slot.patch_comp:
            return recon

        # Apply zigzag delta patch for bit-perfect result
        patch_raw = zlib.decompress(img_slot.patch_comp)
        dtype = np.uint8 if img_slot.delta_is_uint8 else np.uint16
        zigzag = np.frombuffer(patch_raw, dtype=dtype)
        delta = zigzag_decode(zigzag).reshape(h, w)

        return np.clip(recon.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # ─────────────── Single Image Convenience ───────────────

    def encode_single(self, image: np.ndarray) -> MultiplexedPayload:
        """
        Convenience method: encode a single image as a multiplexed payload
        with batch size 1.
        """
        return self.encode_batch([image])

    # ─────────────── Metrics ───────────────

    @staticmethod
    def total_stored_bytes(payload: MultiplexedPayload) -> int:
        """Calculate total stored bytes for the entire multiplexed payload."""
        shared_size = len(payload.shared_amp_comp) + len(payload.shared_idx_comp)
        per_image_size = 0
        for slot in payload.images:
            per_image_size += len(slot.phase_comp) + len(slot.patch_comp)
            if slot.alpha is not None:
                # Add size of compressed alpha
                alpha_comp = zlib.compress(slot.alpha.tobytes(), 9)
                per_image_size += len(alpha_comp)
        header_overhead = 20  # h, w, keep_ratio, n_kept, n_images
        return shared_size + per_image_size + header_overhead

    # ─────────────── Lossy-Only Fast Mode ───────────────

    def encode_lossy_only(
        self,
        images: List[np.ndarray],
        use_alpha: bool = True,
    ) -> MultiplexedPayload:
        """
        Delta yaması olmadan kodla. Ultra-hızlı, yüksek sıkıştırma.
        Kullanım: thumbnail, önizleme, RAG vektör araması.

        Tüm ImageSlot'larda patch_comp = b'' olur.
        decode_single çağrıldığında delta uygulanmaz, lossy recon döndürülür.
        """
        old_lossy = self.lossy
        self.lossy = True
        try:
            return self.encode_batch(images, use_alpha=use_alpha, differential=False)
        finally:
            self.lossy = old_lossy

    # ─────────────── Incremental Encoding ───────────────

    def add_image(
        self,
        payload: MultiplexedPayload,
        new_image: np.ndarray,
        use_alpha: bool = True,
        differential: bool = False,
    ) -> MultiplexedPayload:
        """
        Mevcut payload'a yeni görsel ekle.
        Ortak tabanı YENIDEN HESAPLAMAZ — sadece yeni slot ekler.

        Uyarı: Ortak taban sabit kalır, yeni görsel buna göre encode edilir.
        Her 10 görselde bir rebase() çağrısı önerilir.

        Parameters
        ----------
        payload : MultiplexedPayload
            Mevcut payload (mutate edilmez).
        new_image : np.ndarray
            Eklenecek yeni 2D uint8 görsel.
        use_alpha : bool
            Per-image amplitude scaling kullan.
        differential : bool
            Diferansiyel faz kodlama kullan (son görsel referans).

        Returns
        -------
        MultiplexedPayload
            Yeni slot eklenmiş yeni payload (deep copy).
        """
        from copy import deepcopy

        h, w = payload.h, payload.w
        if new_image.shape != (h, w):
            raise ValueError(
                f"Image shape {new_image.shape} differs from payload shape ({h}, {w})"
            )

        # Decompress shared base
        shared_amp_vals = np.frombuffer(
            zlib.decompress(payload.shared_amp_comp), dtype=np.float32
        ).copy()
        sorted_idx = payload.sorted_idx

        # FFT of new image
        freq = np.fft.fftshift(np.fft.fft2(new_image.astype(np.float64)))
        img_phase = np.angle(freq).flatten()[sorted_idx]
        img_amp_vals = np.abs(freq).flatten()[sorted_idx].astype(np.float32)

        # Alpha scaling
        alpha_k = None
        if use_alpha:
            safe_shared = np.where(shared_amp_vals > 1e-10, shared_amp_vals, 1.0)
            alpha_k = np.where(
                shared_amp_vals > 1e-10,
                img_amp_vals / safe_shared,
                1.0,
            ).astype(np.float16)

        # Phase encoding
        is_diff = False
        if differential and len(payload.images) > 0:
            last_slot = payload.images[-1]
            last_phase_q = np.frombuffer(
                zlib.decompress(last_slot.phase_comp), dtype=np.uint8
            )
            last_phase = last_phase_q.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi

            # If last slot was differential, we need the accumulated phase
            if last_slot.differential and len(payload.images) >= 2:
                # Rebuild accumulated phase up to last slot
                last_phase = self._rebuild_accumulated_phase(payload, len(payload.images) - 1)

            phase_diff = (img_phase - last_phase + np.pi) % (2 * np.pi) - np.pi
            phase_q = np.uint8(
                np.clip((phase_diff + np.pi) / (2 * np.pi) * 255, 0, 255)
            )
            # Dequantize for reconstruction
            phase_diff_deq = phase_q.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi
            phase_deq = (phase_diff_deq + last_phase + np.pi) % (2 * np.pi) - np.pi
            is_diff = True
        else:
            phase_q = np.uint8(
                np.clip((img_phase + np.pi) / (2 * np.pi) * 255, 0, 255)
            )
            phase_deq = phase_q.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi

        phase_comp = zlib.compress(phase_q.tobytes(), 9)

        # Lossy reconstruction
        eff_amp = shared_amp_vals
        if alpha_k is not None:
            eff_amp = shared_amp_vals * alpha_k.astype(np.float32)

        rec_freq = np.zeros(h * w, dtype=np.complex128)
        rec_freq[sorted_idx] = eff_amp * np.exp(1j * phase_deq)
        recon = np.clip(
            np.round(np.fft.ifft2(np.fft.ifftshift(rec_freq.reshape(h, w))).real),
            0, 255,
        ).astype(np.uint8)

        # Delta patch
        delta = new_image.astype(np.int16) - recon.astype(np.int16)
        max_d = int(np.max(np.abs(delta)))
        zz = zigzag_encode(delta.flatten(), max_d)
        patch_comp = zlib.compress(zz.tobytes(), 9)

        new_slot = ImageSlot(
            phase_comp=phase_comp,
            patch_comp=patch_comp,
            max_delta=max_d,
            delta_is_uint8=(max_d <= 127),
            zero_pct=float(np.sum(delta == 0) / delta.size * 100),
            alpha=alpha_k,
            differential=is_diff,
        )

        # Deep copy to avoid mutating original
        new_payload = deepcopy(payload)
        new_payload.images.append(new_slot)
        return new_payload

    def _rebuild_accumulated_phase(
        self, payload: MultiplexedPayload, target_idx: int
    ) -> np.ndarray:
        """Reconstruct accumulated absolute phase at target_idx via sequential decode."""
        phase_deq = None
        for i in range(target_idx + 1):
            slot = payload.images[i]
            phase_q = np.frombuffer(
                zlib.decompress(slot.phase_comp), dtype=np.uint8
            )
            phase_val = phase_q.astype(np.float32) / 255.0 * (2 * np.pi) - np.pi

            if slot.differential and phase_deq is not None:
                phase_deq = (phase_val + phase_deq + np.pi) % (2 * np.pi) - np.pi
            else:
                phase_deq = phase_val
        return phase_deq

    def rebase(
        self,
        payload: MultiplexedPayload,
        use_alpha: bool = True,
    ) -> MultiplexedPayload:
        """
        Tüm görselleri yeniden decode edip ortak tabanı yeniden hesapla.
        Her ~10 görselde bir çağrılması önerilir.
        Bit-perfect garantisi korunur.

        Parameters
        ----------
        payload : MultiplexedPayload
            Mevcut payload.
        use_alpha : bool
            Yeniden encode sırasında alpha scaling kullan.

        Returns
        -------
        MultiplexedPayload
            Yeni ortak tabanla yeniden encode edilmiş payload.
        """
        # 1. Decode all images
        images = [
            self.decode_single(payload, i)
            for i in range(payload.n_images)
        ]
        # 2. Re-encode with fresh shared base
        return self.encode_batch(
            images,
            use_alpha=use_alpha,
            differential=False,  # Reset differential after rebase
        )
