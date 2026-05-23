# holodb/core/codecs.py
# HoloDB Codec Registry - Plugin tabanli codec sistemi.
# Telif Hakki (c) 2026 Ege Berk Turk - Tum Haklari Saklidir.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class CodecRegistry:
    """
    Plugin tabanli codec sistemi.

    Ornek:
        registry = CodecRegistry()
        registry.register("spherical", SphericalEmbeddingCodec)
        codec = registry.get_codec("spherical")
    """

    def __init__(self) -> None:
        self.codecs: Dict[str, Type] = {}
        self.instances: Dict[str, object] = {}
        self.singleton: Dict[str, bool] = {}

    def register(
        self,
        name: str,
        codec_class: Type,
        singleton: bool = True,
    ) -> None:
        """Yeni codec register et."""
        if name in self.codecs:
            logger.warning(f"Codec '{name}' zaten registerlu, yeniden yaziliyor")
        self.codecs[name] = codec_class
        self.singleton[name] = singleton
        if singleton:
            try:
                self.instances[name] = codec_class()
                logger.info(f"Codec '{name}' singleton olarak register edildi")
            except Exception as e:
                logger.error(f"Codec instance olusturulamadi '{name}': {e}")
        else:
            logger.info(f"Codec '{name}' non-singleton olarak register edildi")

    def get_codec(self, name: str) -> object:
        """Codec instance domdur."""
        if name in self.instances:
            return self.instances[name]
        if name not in self.codecs:
            raise KeyError(f"Codec '{name}' register edilmemis")
        return self.codecs[name]()

    def list_codecs(self) -> List[str]:
        """Mevcut codecleri listele."""
        return list(self.codecs.keys())

    def unregister(self, name: str) -> bool:
        """Codeci registryden kaldir."""
        if name in self.codecs:
            del self.codecs[name]
            self.instances.pop(name, None)
            self.singleton.pop(name, None)
            logger.info(f"Codec '{name}' unregister edildi")
            return True
        return False

    def get_codec_info(self, name: str) -> Dict[str, Any]:
        """Codec hakkinda bilgi al."""
        if name not in self.codecs:
            raise KeyError(f"Codec '{name}' bulunamadi")
        codec_class = self.codecs[name]
        return {
            "name": name,
            "class": codec_class.__name__,
            "module": codec_class.__module__,
            "singleton": self.singleton.get(name, False),
            "instantiated": name in self.instances,
        }

    def is_registered(self, name: str) -> bool:
        """Codec register edilmis mi kontrol et."""
        return name in self.codecs


# Global registry singleton
_global_registry: Optional[CodecRegistry] = None


def get_registry() -> CodecRegistry:
    """Global registry instance al."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CodecRegistry()
        _register_default_codecs()
    return _global_registry


def _register_default_codecs() -> None:
    """Default codecleri registryye kaydet."""
    global _global_registry

    # DirectVoxel
    try:
        from holodb.codecs.direct_voxel import DirectVoxelCodec  # type: ignore
        _global_registry.register("direct_voxel", DirectVoxelCodec)
    except ImportError as e:
        logger.warning(f"DirectVoxelCodec import hatasi: {e}")

    # SphericalEmbeddingCodec
    try:
        from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec
        _global_registry.register("spherical_embedding", SphericalEmbeddingCodec)
    except ImportError as e:
        logger.warning(f"SphericalEmbeddingCodec import hatasi: {e}")

    # MultiplexedHolographic
    try:
        from holodb.codecs.multiplexed_holographic import MultiplexedHolographicCodec
        _global_registry.register("multiplexed_holographic", MultiplexedHolographicCodec)
    except ImportError as e:
        logger.warning(f"MultiplexedHolographicCodec import hatasi: {e}")

    # EmbeddingCodec
    try:
        from holodb.codecs.embedding_holographic import EmbeddingCodec  # type: ignore
        _global_registry.register("embedding", EmbeddingCodec)
    except ImportError as e:
        logger.warning(f"EmbeddingCodec import hatasi: {e}")


def register_codec(name: str, codec_class: Type, singleton: bool = True) -> None:
    """Global registry uzerinden codec register et."""
    get_registry().register(name, codec_class, singleton)


def get_codec(name: str) -> object:
    """Global registry uzerinden codec al."""
    return get_registry().get_codec(name)


def list_all_codecs() -> List[str]:
    """Tum codecleri listele."""
    return get_registry().list_codecs()
