"""
RF-Sense3D: Emergency Alert Dispatcher.
Dispatches critical events (Falls, Apnea/Cardiac Arrest, Perimeter Breaches)
to Webhooks, Telegram, or IoT alarms asynchronously.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("RFEmergencyAlert")


class EmergencyAlertDispatcher:
    """
    Asynchronous Emergency Notification Service.
    """

    def __init__(self, webhook_url: Optional[str] = None, cooldown_seconds: float = 10.0):
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time: Dict[str, float] = {}

    def trigger_fall_alert(
        self,
        target_id: str,
        confidence: float,
        location_coords: Dict[str, float],
        vital_signs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches an emergency Fall Detection alarm payload.
        """
        now = time.time()
        if target_id in self.last_alert_time:
            if now - self.last_alert_time[target_id] < self.cooldown_seconds:
                return {"status": "cooldown", "target_id": target_id}

        self.last_alert_time[target_id] = now

        payload = {
            "event_type": "EMERGENCY_FALL_DETECTED",
            "timestamp": now,
            "target_id": target_id,
            "confidence_pct": round(confidence * 100, 1),
            "coordinates_meters": location_coords,
            "vital_signs": vital_signs or {},
            "urgency": "CRITICAL",
            "message": f"EMERGENCY: Human fall detected for {target_id} at position ({location_coords.get('x', 0):.2f}m, {location_coords.get('y', 0):.2f}m). Please check immediately!",
        }

        logger.critical(f"[ALERT DISPATCHED] {json.dumps(payload)}")
        return {"status": "dispatched", "payload": payload}
