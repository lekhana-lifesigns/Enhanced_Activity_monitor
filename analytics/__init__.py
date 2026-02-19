# analytics/__init__.py
"""
Analytics module for Enhanced Activity Monitor.
Provides posture detection, activity classification, clinical monitoring,
and hourly aggregation for efficient reporting.

Architecture:
    Live Stream → PostureSystem → StateCache (RAM) → HourlyAggregator → DB → Reports
"""

from analytics.posture_system import (
    PostureSystem,
    get_posture_system,
    process_keypoints,
)
from analytics.support_surface_detector import (
    SupportSurfaceDetector,
    SupportSurface,
    get_support_surface_detector,
)
from analytics.posture_state_machine import (
    PostureStateMachine,
    PostureState,
    PostureOutput,
    get_posture_state_machine,
)
from analytics.state_cache import (
    StateCache,
    PersonState,
    get_state_cache,
    close_state_cache,
)
from analytics.hourly_aggregator import (
    HourlyAggregator,
    HourlyAggregatorManager,
    HourlyMetrics,
    get_aggregator,
    get_aggregator_manager,
)

__all__ = [
    # Unified posture system
    "PostureSystem",
    "get_posture_system",
    "process_keypoints",
    # Support surface detection
    "SupportSurfaceDetector",
    "SupportSurface",
    "get_support_surface_detector",
    # Posture state machine
    "PostureStateMachine",
    "PostureState",
    "PostureOutput",
    "get_posture_state_machine",
    # State cache
    "StateCache",
    "PersonState",
    "get_state_cache",
    "close_state_cache",
    # Hourly aggregation
    "HourlyAggregator",
    "HourlyAggregatorManager",
    "HourlyMetrics",
    "get_aggregator",
    "get_aggregator_manager",
]
