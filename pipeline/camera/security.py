# pipeline/camera/security.py
"""
Camera Security Module
Implements access control, encryption, and audit logging for clinical-grade security
"""
import os
import json
import time
import logging
import hashlib
import hmac
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from pathlib import Path

log = logging.getLogger("camera_security")

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    log.warning("cryptography not available. Install with: pip install cryptography")


class CameraSecurity:
    """
    Camera security module for clinical-grade access control and data protection.
    Features:
    - Access control (authentication/authorization)
    - Frame encryption (optional)
    - Audit logging
    - Secure storage
    - HIPAA compliance measures
    """
    
    def __init__(self, 
                 config_path: str = "config/security.yaml",
                 audit_log_path: str = "logs/security_audit.log",
                 enable_encryption: bool = True):
        """
        Args:
            config_path: Path to security configuration
            audit_log_path: Path to audit log file
            enable_encryption: Enable frame encryption
        """
        self.config_path = config_path
        self.audit_log_path = audit_log_path
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        
        # Access control
        self.authorized_users = {}
        self.active_sessions = {}
        
        # Encryption
        self.encryption_key = None
        self.cipher = None
        
        # Audit log
        self.audit_log_file = None
        self._init_audit_log()
        
        # Load configuration
        self._load_config()
        
        # Initialize encryption if enabled
        if self.enable_encryption:
            self._init_encryption()
        
        log.info("Camera security module initialized (encryption: %s)", self.enable_encryption)
    
    def _init_audit_log(self):
        """Initialize audit log file."""
        try:
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
            self.audit_log_file = open(self.audit_log_path, 'a', encoding='utf-8')
            log.info("Audit log initialized: %s", self.audit_log_path)
        except Exception as e:
            log.error("Failed to initialize audit log: %s", e)
            self.audit_log_file = None
    
    def _load_config(self):
        """Load security configuration."""
        try:
            if os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.authorized_users = config.get("authorized_users", {})
            else:
                # Default configuration
                self.authorized_users = {
                    "admin": {
                        "password_hash": self._hash_password("admin123"),  # Change in production!
                        "role": "admin",
                        "permissions": ["read", "write", "admin"]
                    }
                }
                log.warning("Using default security config. Change passwords in production!")
        except Exception as e:
            log.error("Failed to load security config: %s", e)
            self.authorized_users = {}
    
    def _init_encryption(self):
        """Initialize encryption."""
        if not CRYPTO_AVAILABLE:
            return
        
        try:
            # Generate or load encryption key
            key_file = "storage/.encryption_key"
            
            if os.path.exists(key_file):
                # Load existing key
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_file), exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                # Set restrictive permissions (Unix only)
                try:
                    os.chmod(key_file, 0o600)
                except:
                    pass
            
            self.encryption_key = key
            self.cipher = Fernet(key)
            log.info("Encryption initialized")
            
        except Exception as e:
            log.error("Failed to initialize encryption: %s", e)
            self.enable_encryption = False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate user.
        
        Args:
            username: Username
            password: Password
        
        Returns:
            (success: bool, session_token: str or None)
        """
        if username not in self.authorized_users:
            self._audit_log("AUTH_FAIL", username, "User not found")
            return False, None
        
        user = self.authorized_users[username]
        password_hash = self._hash_password(password)
        
        if user["password_hash"] != password_hash:
            self._audit_log("AUTH_FAIL", username, "Invalid password")
            return False, None
        
        # Generate session token
        session_token = self._generate_session_token(username)
        self.active_sessions[session_token] = {
            "username": username,
            "role": user["role"],
            "permissions": user.get("permissions", []),
            "created_at": time.time(),
            "last_activity": time.time()
        }
        
        self._audit_log("AUTH_SUCCESS", username, "User authenticated")
        return True, session_token
    
    def _generate_session_token(self, username: str) -> str:
        """Generate secure session token."""
        timestamp = str(time.time())
        data = f"{username}:{timestamp}"
        token = hashlib.sha256(data.encode()).hexdigest()
        return token
    
    def authorize(self, session_token: str, permission: str) -> bool:
        """
        Check if session has permission.
        
        Args:
            session_token: Session token
            permission: Required permission ("read", "write", "admin")
        
        Returns:
            True if authorized
        """
        if session_token not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_token]
        session["last_activity"] = time.time()
        
        permissions = session.get("permissions", [])
        return permission in permissions or "admin" in permissions
    
    def encrypt_frame(self, frame_data: bytes) -> Optional[bytes]:
        """
        Encrypt frame data.
        
        Args:
            frame_data: Frame bytes
        
        Returns:
            Encrypted bytes or None if encryption disabled
        """
        if not self.enable_encryption or not self.cipher:
            return None
        
        try:
            encrypted = self.cipher.encrypt(frame_data)
            return encrypted
        except Exception as e:
            log.error("Frame encryption failed: %s", e)
            return None
    
    def decrypt_frame(self, encrypted_data: bytes) -> Optional[bytes]:
        """
        Decrypt frame data.
        
        Args:
            encrypted_data: Encrypted frame bytes
        
        Returns:
            Decrypted bytes or None if decryption fails
        """
        if not self.enable_encryption or not self.cipher:
            return None
        
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return decrypted
        except Exception as e:
            log.error("Frame decryption failed: %s", e)
            return None
    
    def _audit_log(self, event_type: str, username: str, details: str):
        """
        Log security event to audit log.
        
        Args:
            event_type: Event type (AUTH_SUCCESS, AUTH_FAIL, ACCESS_DENIED, etc.)
            username: Username
            details: Event details
        """
        if not self.audit_log_file:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "username": username,
                "details": details
            }
            
            self.audit_log_file.write(json.dumps(log_entry) + "\n")
            self.audit_log_file.flush()
            
        except Exception as e:
            log.error("Audit log write failed: %s", e)
    
    def log_access(self, session_token: str, resource: str, action: str):
        """
        Log resource access.
        
        Args:
            session_token: Session token
            resource: Resource accessed (e.g., "camera", "patient_data")
            action: Action performed (e.g., "read", "write")
        """
        if session_token in self.active_sessions:
            username = self.active_sessions[session_token]["username"]
            self._audit_log("ACCESS", username, f"{action} {resource}")
    
    def revoke_session(self, session_token: str):
        """Revoke session token."""
        if session_token in self.active_sessions:
            username = self.active_sessions[session_token]["username"]
            del self.active_sessions[session_token]
            self._audit_log("SESSION_REVOKED", username, "Session revoked")
    
    def cleanup_expired_sessions(self, max_age_seconds: int = 3600):
        """Remove expired sessions."""
        now = time.time()
        expired = []
        
        for token, session in self.active_sessions.items():
            age = now - session["last_activity"]
            if age > max_age_seconds:
                expired.append(token)
        
        for token in expired:
            username = self.active_sessions[token]["username"]
            del self.active_sessions[token]
            self._audit_log("SESSION_EXPIRED", username, "Session expired")
        
        if expired:
            log.info("Cleaned up %d expired sessions", len(expired))


def create_camera_security(config: dict = None) -> Optional[CameraSecurity]:
    """Factory function to create camera security module."""
    if config is None:
        config = {}
    
    try:
        security = CameraSecurity(
            config_path=config.get("config_path", "config/security.yaml"),
            audit_log_path=config.get("audit_log_path", "logs/security_audit.log"),
            enable_encryption=config.get("enable_encryption", True)
        )
        return security
    except Exception as e:
        log.warning("Failed to create camera security: %s", e)
        return None

