# pipeline/patient/rate_limiter.py
"""
Rate Limiting for Face Verification
Prevents brute-force attacks and excessive verification attempts
"""
import time
import logging
from collections import deque
from typing import Dict, Optional, Tuple

log = logging.getLogger("rate_limiter")


class RateLimiter:
    """
    Rate limiter for face verification attempts.
    Tracks attempts per patient ID and enforces limits.
    """
    
    def __init__(self, 
                 max_attempts: int = 5,
                 window_seconds: int = 60,
                 lockout_seconds: int = 300):
        """
        Args:
            max_attempts: Maximum attempts allowed in window
            window_seconds: Time window for rate limiting
            lockout_seconds: Lockout duration after exceeding limit
        """
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        
        # Track attempts per patient ID
        # patient_id -> deque of timestamps
        self.attempts: Dict[str, deque] = {}
        
        # Track lockouts
        # patient_id -> lockout_until_timestamp
        self.lockouts: Dict[str, float] = {}
    
    def check(self, patient_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if verification attempt is allowed.
        
        Args:
            patient_id: Patient ID
        
        Returns:
            (allowed: bool, reason: str or None)
        """
        now = time.time()
        
        # Check if locked out
        if patient_id in self.lockouts:
            lockout_until = self.lockouts[patient_id]
            if now < lockout_until:
                remaining = int(lockout_until - now)
                return False, f"Rate limit exceeded. Locked out for {remaining} seconds."
            else:
                # Lockout expired
                del self.lockouts[patient_id]
        
        # Initialize attempts deque if needed
        if patient_id not in self.attempts:
            self.attempts[patient_id] = deque()
        
        # Remove old attempts outside window
        attempts_deque = self.attempts[patient_id]
        while attempts_deque and attempts_deque[0] < now - self.window_seconds:
            attempts_deque.popleft()
        
        # Check if limit exceeded
        if len(attempts_deque) >= self.max_attempts:
            # Lockout
            self.lockouts[patient_id] = now + self.lockout_seconds
            log.warning("Rate limit exceeded for patient %s. Locking out for %d seconds.", 
                       patient_id, self.lockout_seconds)
            return False, f"Rate limit exceeded ({self.max_attempts} attempts in {self.window_seconds}s). Locked out for {self.lockout_seconds}s."
        
        # Record attempt
        attempts_deque.append(now)
        
        remaining = self.max_attempts - len(attempts_deque)
        return True, None
    
    def record_success(self, patient_id: str):
        """
        Record successful verification (reset attempts).
        
        Args:
            patient_id: Patient ID
        """
        if patient_id in self.attempts:
            self.attempts[patient_id].clear()
        if patient_id in self.lockouts:
            del self.lockouts[patient_id]
        log.debug("Reset rate limit for patient %s after successful verification", patient_id)
    
    def get_remaining_attempts(self, patient_id: str) -> int:
        """Get remaining attempts for patient."""
        if patient_id not in self.attempts:
            return self.max_attempts
        
        now = time.time()
        attempts_deque = self.attempts[patient_id]
        
        # Remove old attempts
        while attempts_deque and attempts_deque[0] < now - self.window_seconds:
            attempts_deque.popleft()
        
        return max(0, self.max_attempts - len(attempts_deque))
    
    def is_locked_out(self, patient_id: str) -> bool:
        """Check if patient is currently locked out."""
        if patient_id not in self.lockouts:
            return False
        
        now = time.time()
        if now >= self.lockouts[patient_id]:
            del self.lockouts[patient_id]
            return False
        
        return True


def create_rate_limiter(config: dict = None) -> RateLimiter:
    """Factory function to create rate limiter."""
    if config is None:
        config = {}
    
    return RateLimiter(
        max_attempts=config.get("max_attempts", 5),
        window_seconds=config.get("window_seconds", 60),
        lockout_seconds=config.get("lockout_seconds", 300)
    )

