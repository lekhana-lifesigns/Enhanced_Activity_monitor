# security/secure_config.py
"""
Secure Configuration Management
Replaces hardcoded credentials with environment variables and secure key management.
"""

import os
import hashlib
import secrets
import logging
from typing import Optional, Dict
from pathlib import Path

log = logging.getLogger("secure_config")


class SecureConfigError(Exception):
 """Raised when secure configuration fails."""
 pass


class SecureCredentialManager:
 """
 Manages credentials securely using environment variables or secure vault.

 NEVER stores plaintext passwords in config files.
 """

 # Required environment variables
 REQUIRED_ENV_VARS = [
 "EAM_MQTT_USERNAME",
 "EAM_MQTT_PASSWORD",
 "EAM_DB_ENCRYPTION_KEY",
 "EAM_ADMIN_PASSWORD_HASH",
 ]

 def __init__(self, env_file: Optional[str] = None):
 """
 Initialize credential manager.

 Args:
 env_file: Optional path to .env file (for development only)
 """
 self._credentials: Dict[str, str] = {}
 self._load_from_environment()

 if env_file and Path(env_file).exists():
 self._load_from_env_file(env_file)
 log.warning("Loaded credentials from .env file - FOR DEVELOPMENT ONLY")

 def _load_from_environment(self):
 """Load credentials from environment variables."""
 for var in self.REQUIRED_ENV_VARS:
 value = os.environ.get(var)
 if value:
 self._credentials[var] = value

 def _load_from_env_file(self, env_file: str):
 """Load credentials from .env file (development only)."""
 try:
 with open(env_file, 'r') as f:
 for line in f:
 line = line.strip()
 if line and not line.startswith('#') and '=' in line:
 key, value = line.split('=', 1)
 self._credentials[key.strip()] = value.strip().strip('"\'')
 except Exception as e:
 log.error(f"Failed to load .env file: {e}")

 def get_credential(self, key: str) -> Optional[str]:
 """Get a credential value."""
 return self._credentials.get(key)

 def require_credential(self, key: str) -> str:
 """Get a credential or raise error if missing."""
 value = self.get_credential(key)
 if not value:
 raise SecureConfigError(
 f"Required credential '{key}' not found. "
 f"Set environment variable or use .env file."
 )
 return value

 def validate_required_credentials(self) -> bool:
 """Validate that all required credentials are present."""
 missing = []
 for var in self.REQUIRED_ENV_VARS:
 if var not in self._credentials:
 missing.append(var)

 if missing:
 log.critical(f"Missing required credentials: {missing}")
 return False
 return True

 @staticmethod
 def hash_password(password: str, salt: Optional[bytes] = None) -> tuple:
 """
 Hash password with random salt using PBKDF2.

 Returns:
 (hash_hex, salt_hex) tuple
 """
 if salt is None:
 salt = secrets.token_bytes(32) # 256-bit random salt

 # Use PBKDF2 with SHA-256, 100,000 iterations
 hash_bytes = hashlib.pbkdf2_hmac(
 'sha256',
 password.encode('utf-8'),
 salt,
 iterations=100000
 )

 return hash_bytes.hex(), salt.hex()

 @staticmethod
 def verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
 """Verify password against stored hash."""
 salt = bytes.fromhex(salt_hex)
 computed_hash, _ = SecureCredentialManager.hash_password(password, salt)
 return secrets.compare_digest(computed_hash, stored_hash)

 @staticmethod
 def generate_api_key() -> str:
 """Generate a secure API key."""
 return secrets.token_urlsafe(32)


class SecureMQTTConfig:
 """Secure MQTT configuration with TLS support."""

 def __init__(self, credential_manager: SecureCredentialManager):
 self.creds = credential_manager

 def get_config(self) -> Dict:
 """Get secure MQTT configuration."""
 return {
 "broker": os.environ.get("EAM_MQTT_BROKER", "localhost"),
 "port": int(os.environ.get("EAM_MQTT_PORT", "8883")), # TLS port
 "username": self.creds.require_credential("EAM_MQTT_USERNAME"),
 "password": self.creds.require_credential("EAM_MQTT_PASSWORD"),
 "use_tls": True,
 "tls_ca_certs": os.environ.get("EAM_MQTT_CA_CERT", "certs/ca.crt"),
 "tls_certfile": os.environ.get("EAM_MQTT_CLIENT_CERT", "certs/client.crt"),
 "tls_keyfile": os.environ.get("EAM_MQTT_CLIENT_KEY", "certs/client.key"),
 "tls_insecure": False,
 }


# Global instance (lazy initialization)
_credential_manager: Optional[SecureCredentialManager] = None


def get_credential_manager() -> SecureCredentialManager:
 """Get the global credential manager instance."""
 global _credential_manager
 if _credential_manager is None:
 env_file = os.environ.get("EAM_ENV_FILE", ".env")
 _credential_manager = SecureCredentialManager(env_file)
 return _credential_manager


def require_secure_mode():
 """Ensure running in secure mode with all required credentials."""
 manager = get_credential_manager()
 if not manager.validate_required_credentials():
 raise SecureConfigError(
 "System cannot start without required security credentials. "
 "See documentation for required environment variables."
 )
