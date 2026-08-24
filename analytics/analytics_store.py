"""
analytics_store.py
=============================================================================
Supabase & Analytical State Persistence Client:
- Persists high-resolution Contactless Vital Signs time-series (BPM, BrPM)
- Logs Multi-Person Re-ID trajectory events & intrusion alerts
- Supports offline SQLite / JSONL caching when cloud database is offline
=============================================================================
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("AnalyticsStore")


class SupabaseAnalyticsStore:
    """
    Persistence gateway for vital sign histories and tracking telemetry.
    """
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        local_cache_path: str = "dataset/telemetry_log.jsonl"
    ):
        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_KEY")
        self.local_cache_path = local_cache_path
        
        # Make directory for local backup
        os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
        self.client = None
        
        if self.url and self.key:
            try:
                import importlib
                supabase_module = importlib.import_module("supabase")
                create_client = getattr(supabase_module, "create_client")
                self.client = create_client(self.url, self.key)
                logger.info("Connected to Supabase analytical database.")
            except Exception as e:
                logger.warning(f"Could not connect to Supabase: {e}. Using local JSONL cache.")

    def log_vital_signs(
        self,
        target_id: str,
        respiration_rate: float,
        heart_rate: float,
        confidence: float
    ):
        record = {
            "timestamp": time.time(),
            "target_id": target_id,
            "respiration_brpm": respiration_rate,
            "heart_rate_bpm": heart_rate,
            "confidence": confidence
        }

        # Try Supabase insert
        if self.client:
            try:
                self.client.table("vital_signs_telemetry").insert(record).execute()
                return
            except Exception as e:
                logger.error(f"Supabase insert failed: {e}")

        # Local append fallback
        with open(self.local_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_track_event(
        self,
        target_id: str,
        event_type: str,  # "entry", "exit", "trajectory_update"
        position_3d: List[float]
    ):
        record = {
            "timestamp": time.time(),
            "target_id": target_id,
            "event_type": event_type,
            "position_3d": position_3d
        }
        with open(self.local_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
