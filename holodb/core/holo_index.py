# holodb/core/holo_index.py
# HoloIndex: rootseed + path -> deterministik adreslenebilir depolama.
# Tek seed ile binlerce dosya hedefi: rootseed, path -> omega, t -> bytes
# Telif Hakki (c) 2026 Ege Berk Turk - Tum Haklari Saklidir.

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from holodb.core.seed_engine import SeedEngine


class AddressableStore(Protocol):
    """HoloIndex'in gerektirdigi depolama arayuzu."""

    def write(self, omega: int, t: int, data: bytes) -> None:
        ...

    def read(self, omega: int, t: int) -> Optional[bytes]:
        ...


class InMemoryStore:
    """Gelistirme ve test icin bellekte calisan store."""

    def __init__(self) -> None:
        self._store: dict = {}

    def write(self, omega: int, t: int, data: bytes) -> None:
        self._store[(omega, t)] = data

    def read(self, omega: int, t: int) -> Optional[bytes]:
        return self._store.get((omega, t))

    def __len__(self) -> int:
        return len(self._store)

    def total_bytes(self) -> int:
        return sum(len(v) for v in self._store.values())


class HoloIndex:
    """
    Deterministik adreslenebilir depolama.

    Tek bir root_seed ile hiyerarsik yol uzayini adresler:
        root_seed + "documents/report.pdf/page3" -> (omega=0, t=int)

    Ayni seed her zaman ayni adresi uretir (cross-platform stabil).

    Ornek:
        store = InMemoryStore()
        idx = HoloIndex(root_seed=42, store=store)
        idx.store("docs/hello.txt", b"Hello World")
        data = idx.retrieve("docs/hello.txt")  # -> b"Hello World"
    """

    def __init__(self, root_seed: int, store: AddressableStore) -> None:
        self.root_seed = int(root_seed)
        self.store = store

    def addr(self, path: str) -> Tuple[int, int]:
        """Yol stringini (omega, t) adres cifti'ne donustur."""
        t = SeedEngine.derive_from_path(self.root_seed, path)
        return 0, t

    def store_data(self, path: str, data: bytes) -> None:
        """Veriyi deterministik adrese yaz."""
        omega, t = self.addr(path)
        self.store.write(omega, t, data)

    # Eski API uyumlulugu icin alias
    def store(self, path: str, data: bytes) -> None:  # type: ignore[override]
        self.store_data(path, data)

    def retrieve(self, path: str) -> Optional[bytes]:
        """Deterministik adresteki veriyi oku."""
        omega, t = self.addr(path)
        return self.store.read(omega, t)

    def exists(self, path: str) -> bool:
        """Adres dolu mu kontrol et."""
        return self.retrieve(path) is not None

    def address_of(self, path: str) -> dict:
        """Debug: adres bilgisini goster."""
        omega, t = self.addr(path)
        return {"root_seed": self.root_seed, "path": path, "omega": omega, "t": t}
