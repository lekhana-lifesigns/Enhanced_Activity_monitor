"""System health model for edge device metrics - matches edge cache schema."""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class SystemHealth(Base):
    """
    System health metrics from edge devices.
    Stores CPU, RAM, temperature, disk, and inference performance.
    Schema aligned with edge storage/db.py system_health table.
    """

    __tablename__ = "system_health"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Timestamp (edge uses 'ts' as REAL, we use datetime)
    ts: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, index=True, nullable=False, default=datetime.utcnow
    )

    # Device identification
    device_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)

    # System metrics (matches edge schema)
    cpu_percent: Mapped[float] = mapped_column(Float, nullable=True)
    ram_percent: Mapped[float] = mapped_column(Float, nullable=True)
    temp_celsius: Mapped[float] = mapped_column(Float, nullable=True)
    disk_percent: Mapped[float] = mapped_column(Float, nullable=True)

    # Inference performance (matches edge schema)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=True)
    fps: Mapped[float] = mapped_column(Float, nullable=True)

    # Additional backend fields
    gpu_percent: Mapped[float] = mapped_column(Float, nullable=True)
    gpu_memory_percent: Mapped[float] = mapped_column(Float, nullable=True)
    network_rx_bytes: Mapped[int] = mapped_column(nullable=True)
    network_tx_bytes: Mapped[int] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_health_device_ts", "device_id", "ts"),
        Index("idx_health_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<SystemHealth {self.device_id} cpu={self.cpu_percent}% ram={self.ram_percent}%>"
