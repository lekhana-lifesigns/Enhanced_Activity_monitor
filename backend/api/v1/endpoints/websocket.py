"""WebSocket endpoints for real-time streaming."""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from jose import jwt, JWTError

from backend.core.config import settings

log = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time streaming.

    Supports:
    - Device-specific streams (events from a single device)
    - Facility-wide streams (all events from a facility)
    - Alert streams (all alerts)
    """

    def __init__(self):
        # device_id -> set of connections
        self.device_connections: Dict[str, Set[WebSocket]] = {}
        # facility_id -> set of connections
        self.facility_connections: Dict[str, Set[WebSocket]] = {}
        # All alert subscribers
        self.alert_connections: Set[WebSocket] = set()
        # All connections
        self.all_connections: Set[WebSocket] = set()

    async def connect(
        self,
        websocket: WebSocket,
        device_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        subscribe_alerts: bool = False,
    ):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.all_connections.add(websocket)

        if device_id:
            if device_id not in self.device_connections:
                self.device_connections[device_id] = set()
            self.device_connections[device_id].add(websocket)
            log.info(f"WebSocket connected to device: {device_id}")

        if facility_id:
            if facility_id not in self.facility_connections:
                self.facility_connections[facility_id] = set()
            self.facility_connections[facility_id].add(websocket)
            log.info(f"WebSocket connected to facility: {facility_id}")

        if subscribe_alerts:
            self.alert_connections.add(websocket)
            log.info("WebSocket subscribed to alerts")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.all_connections.discard(websocket)
        self.alert_connections.discard(websocket)

        for connections in self.device_connections.values():
            connections.discard(websocket)

        for connections in self.facility_connections.values():
            connections.discard(websocket)

        log.info("WebSocket disconnected")

    async def broadcast_event(self, device_id: str, event_data: dict):
        """Broadcast event to relevant subscribers."""
        message = json.dumps({
            "type": "event",
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": event_data,
        })

        # Send to device-specific subscribers
        if device_id in self.device_connections:
            await self._send_to_connections(
                self.device_connections[device_id],
                message,
            )

        # Send to facility subscribers
        facility_id = event_data.get("facility_id")
        if facility_id and facility_id in self.facility_connections:
            await self._send_to_connections(
                self.facility_connections[facility_id],
                message,
            )

    async def broadcast_alert(self, alert_data: dict):
        """Broadcast alert to all alert subscribers."""
        message = json.dumps({
            "type": "alert",
            "timestamp": datetime.utcnow().isoformat(),
            "data": alert_data,
        })

        await self._send_to_connections(self.alert_connections, message)

    async def _send_to_connections(self, connections: Set[WebSocket], message: str):
        """Send message to a set of connections."""
        disconnected = set()

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                log.debug(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    async def send_heartbeat(self):
        """Send heartbeat to all connections."""
        message = json.dumps({
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
        })

        await self._send_to_connections(self.all_connections, message)

    @property
    def connection_count(self) -> int:
        """Get total connection count."""
        return len(self.all_connections)


# Global connection manager
manager = ConnectionManager()


def verify_ws_token(token: str) -> dict:
    """Verify JWT token for WebSocket authentication."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload
    except JWTError:
        return None


@router.websocket("/live/{device_id}")
async def websocket_device_stream(
    websocket: WebSocket,
    device_id: str,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time device event stream.

    Connect with: ws://host/api/v1/ws/live/{device_id}?token={jwt_token}

    Messages received:
    - type: "event" - Activity event from device
    - type: "heartbeat" - Keep-alive ping

    Messages can be sent:
    - type: "ping" - Request heartbeat
    """
    # Verify token
    if token:
        payload = verify_ws_token(token)
        if not payload:
            await websocket.close(code=4001, reason="Invalid token")
            return
    else:
        # In development, allow without token
        if settings.ENVIRONMENT == "production":
            await websocket.close(code=4001, reason="Token required")
            return

    await manager.connect(websocket, device_id=device_id)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            try:
                # Wait for messages from client
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.WS_HEARTBEAT_INTERVAL,
                )

                # Handle ping
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/alerts")
async def websocket_alerts_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    facility_id: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time alert stream.

    Connect with: ws://host/api/v1/ws/alerts?token={jwt_token}

    Optionally filter by facility: ?facility_id={facility_id}
    """
    # Verify token
    if token:
        payload = verify_ws_token(token)
        if not payload:
            await websocket.close(code=4001, reason="Invalid token")
            return
    else:
        if settings.ENVIRONMENT == "production":
            await websocket.close(code=4001, reason="Token required")
            return

    await manager.connect(
        websocket,
        facility_id=facility_id,
        subscribe_alerts=True,
    )

    try:
        await websocket.send_json({
            "type": "connected",
            "stream": "alerts",
            "facility_id": facility_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.WS_HEARTBEAT_INTERVAL,
                )

                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "total_connections": manager.connection_count,
        "device_subscriptions": len(manager.device_connections),
        "facility_subscriptions": len(manager.facility_connections),
        "alert_subscribers": len(manager.alert_connections),
    }
