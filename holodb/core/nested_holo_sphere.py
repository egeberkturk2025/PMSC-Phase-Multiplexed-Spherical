# holodb/core/nested_holo_sphere.py
# NestedHolographicSphere - 5D veri konteyneri.
# Oyun motoru metaforunda 'world container' karsiligi.
# Telif Hakki (c) 2026 Ege Berk Turk - Tum Haklari Saklidir.

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SPHERE_FORMAT_VERSION = "holo-sphere-v1"
SCHEMA_VERSION = "1.0"
ENGINE_VERSION = "1.0"


@dataclass
class LayerConfig:
    """Layer konfigurasyonu."""
    name: str
    codec_type: str = "direct_voxel"
    dimensions: Tuple[int, int, int] = (64, 64, 64)
    noise_type: Optional[str] = None
    noise_level: float = 0.0
    repair_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "codec_type": self.codec_type,
            "dimensions": list(self.dimensions),
            "noise_type": self.noise_type,
            "noise_level": self.noise_level,
            "repair_method": self.repair_method,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LayerConfig:
        return cls(
            name=d["name"],
            codec_type=d.get("codec_type", "direct_voxel"),
            dimensions=tuple(d.get("dimensions", [64, 64, 64])),
            noise_type=d.get("noise_type"),
            noise_level=d.get("noise_level", 0.0),
            repair_method=d.get("repair_method"),
        )


class LayerData:
    """Sphere icindeki bir layer'in veri yapisi."""

    def __init__(
        self,
        name: str,
        config: LayerConfig,
        data: Optional[bytes] = None,
        index: int = 0,
    ) -> None:
        self.name = name
        self.config = config
        self.data: Optional[bytes] = data
        self.original_data_hash: Optional[str] = None
        self.created_at: Optional[str] = None
        self.index: int = index

    def set_data(self, data: bytes) -> None:
        self.data = data
        self.original_data_hash = hashlib.sha256(data).hexdigest()
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        import base64
        return {
            "name": self.name,
            "index": self.index,
            "config": self.config.to_dict(),
            "data_b64": base64.b64encode(self.data).decode() if self.data else None,
            "original_data_hash": self.original_data_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LayerData:
        import base64
        layer = cls(
            name=d["name"],
            config=LayerConfig.from_dict(d["config"]),
            index=d.get("index", 0),
        )
        if d.get("data_b64"):
            layer.data = base64.b64decode(d["data_b64"])
        layer.original_data_hash = d.get("original_data_hash")
        layer.created_at = d.get("created_at")
        return layer


class NestedHolographicSphere:
    """
    Ic ice holografik kure - 5D veri konteyneri.

    Oyun motoru metaforu:
        Outer Sphere -> Docker gibi container
        Inner Spheres -> Layerlar / data chunks
        Her layer    -> Ayri codec, noise, repair konfigurasyonu

    Kullanim:
        sphere = NestedHolographicSphere("my_sphere")
        sphere.add_layer("metin", b"Hello World")
        data = sphere.reconstruct("metin")
        sphere.export_sphere("dosya.sphere")
    """

    SPHERE_FORMAT_VERSION = SPHERE_FORMAT_VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ENGINE_VERSION = ENGINE_VERSION

    def __init__(
        self,
        name: str,
        dimensions: Tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self.name = name
        self.dimensions = dimensions
        self.layers: Dict[str, LayerData] = {}
        self.sphere_id = str(uuid.uuid4())[:8]
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format_version": self.SPHERE_FORMAT_VERSION,
            "total_layers": 0,
        }

    # ------------------------------------------------------------------
    # Layer yonetimi
    # ------------------------------------------------------------------
    def add_layer(
        self,
        name: str,
        data: bytes,
        codec: str = "direct_voxel",
        dimensions: Optional[Tuple[int, int, int]] = None,
    ) -> str:
        """Veriyi encode edip sphere'e yeni layer olarak ekle."""
        if name in self.layers:
            raise ValueError(f"Layer '{name}' zaten mevcut")
        config = LayerConfig(
            name=name,
            codec_type=codec,
            dimensions=dimensions or self.dimensions,
        )
        layer = LayerData(name=name, config=config, index=len(self.layers))
        layer.set_data(data)
        self.layers[name] = layer
        self.metadata["total_layers"] = len(self.layers)
        return name

    def reconstruct(self, layer_name: str) -> bytes:
        """Layer verisini geri al."""
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' bulunamadi")
        layer = self.layers[layer_name]
        if layer.data is None:
            raise ValueError(f"Layer '{layer_name}' bos")
        return layer.data

    def get_layer(self, layer_name: str) -> LayerData:
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' bulunamadi")
        return self.layers[layer_name]

    def remove_layer(self, layer_name: str) -> None:
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' bulunamadi")
        del self.layers[layer_name]
        self.metadata["total_layers"] = len(self.layers)

    def list_layers(self) -> List[str]:
        return list(self.layers.keys())

    def inject_noise(
        self,
        layer_name: str,
        level: float = 0.1,
        profile: str = "gaussian",
        seed: Optional[int] = None,
    ) -> None:
        """Layer verisine noise ekle (test/repair deneyleri icin)."""
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' bulunamadi")
        layer = self.layers[layer_name]
        if layer.data is None:
            return
        rng = np.random.default_rng(seed)
        arr = np.frombuffer(layer.data, dtype=np.uint8).copy()
        noise = rng.integers(0, int(255 * level) + 1, size=len(arr), dtype=np.uint8)
        arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        layer.data = arr.tobytes()
        layer.config.noise_type = profile
        layer.config.noise_level = level

    def verify_layer_integrity(self, layer_name: str, expected_data: bytes) -> bool:
        """Layer verisini dogrula."""
        if layer_name not in self.layers:
            return False
        layer = self.layers[layer_name]
        if layer.original_data_hash is None:
            return False
        return hashlib.sha256(expected_data).hexdigest() == layer.original_data_hash

    # ------------------------------------------------------------------
    # Serializasyon
    # ------------------------------------------------------------------
    def generate_integrity_hash(self) -> str:
        hasher = hashlib.sha256()
        for name, layer in sorted(self.layers.items()):
            hasher.update(name.encode())
            if layer.original_data_hash:
                hasher.update(layer.original_data_hash.encode())
        return hasher.hexdigest()[:16]

    def export_sphere(self, path: str) -> None:
        """'.sphere' formatinda diske yaz."""
        path_obj = Path(path)
        layers_data = []
        for idx, (layer_name, layer) in enumerate(sorted(self.layers.items())):
            layer.index = idx
            layers_data.append(layer.to_dict())
        sphere_dict = {
            "format": self.SPHERE_FORMAT_VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "engine_version": self.ENGINE_VERSION,
            "created_at": self.metadata.get("created_at"),
            "layer_count": len(self.layers),
            "integrity_hash": self.generate_integrity_hash(),
            "name": self.name,
            "dimensions": list(self.dimensions),
            "metadata": self.metadata,
            "layers": layers_data,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        path_obj.write_text(json.dumps(sphere_dict, indent=2), encoding="utf-8")

    def load_sphere(self, path: str) -> None:
        """'.sphere' dosyasindan yukle."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Sphere dosyasi bulunamadi: {path}")
        sphere_dict = json.loads(path_obj.read_text(encoding="utf-8"))
        fmt = sphere_dict.get("format")
        if fmt != self.SPHERE_FORMAT_VERSION:
            raise ValueError(f"Desteklenmeyen format: {fmt}")
        self.name = sphere_dict["name"]
        self.dimensions = tuple(sphere_dict["dimensions"])
        self.metadata = sphere_dict.get("metadata", {})
        self.layers.clear()
        for layer_data in sorted(sphere_dict.get("layers", []), key=lambda x: x.get("index", 0)):
            layer = LayerData.from_dict(layer_data)
            self.layers[layer.name] = layer

    def inspect(self) -> Dict[str, Any]:
        """Sphere hakkinda standartlastirilmis rapor."""
        layer_reports = []
        for name, layer in self.layers.items():
            layer_reports.append({
                "name": name,
                "codec": layer.config.codec_type,
                "size_bytes": len(layer.data) if layer.data else 0,
                "noise_level": layer.config.noise_level,
                "integrity_verified": layer.original_data_hash is not None,
            })
        return {
            "name": self.name,
            "sphere_id": self.sphere_id,
            "dimensions": self.dimensions,
            "total_layers": len(self.layers),
            "layers": layer_reports,
            "integrity_status": "verified" if all(
                l.original_data_hash for l in self.layers.values()
            ) else "partial",
        }

    def merge(self, other: NestedHolographicSphere) -> None:
        """Baska bir sphere'i bu sphere ile birlestir."""
        for layer_name, layer in other.layers.items():
            new_name = layer_name if layer_name not in self.layers else f"{layer_name}_merged_{len(self.layers)}"
            self.layers[new_name] = layer
        self.metadata["total_layers"] = len(self.layers)

    def clear(self) -> None:
        """Tum layerlari temizle."""
        self.layers.clear()
        self.metadata["total_layers"] = 0


def create_sphere(
    name: str,
    dimensions: Tuple[int, int, int] = (64, 64, 64),
) -> NestedHolographicSphere:
    """Kolaylik factory fonksiyonu."""
    return NestedHolographicSphere(name=name, dimensions=dimensions)


def load_sphere(path: str) -> NestedHolographicSphere:
    """'.sphere' dosyasindan sphere yukle."""
    sphere = NestedHolographicSphere(name="temp")
    sphere.load_sphere(path)
    return sphere
