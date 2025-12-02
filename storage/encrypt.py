# storage/encrypt.py
# Encryption utilities for sensitive patient data

import hashlib
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

log = logging.getLogger("encrypt")

# Default encryption key (should be set via environment variable in production)
DEFAULT_KEY = b"change_this_key_in_production_use_env_var"

def generate_key(password: bytes, salt: bytes = None) -> bytes:
    """
    Generate encryption key from password.
    
    Args:
        password: Password bytes
        salt: Salt bytes (optional, generates random if None)
    
    Returns:
        Encryption key (bytes)
    """
    if salt is None:
        import os
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key

def get_encryption_key():
    """
    Get encryption key from environment or use default.
    
    Returns:
        Encryption key (bytes)
    """
    import os
    key_env = os.getenv("EAC_ENCRYPTION_KEY")
    if key_env:
        return key_env.encode()
    return DEFAULT_KEY

class DataEncryptor:
    """
    Encrypt/decrypt sensitive patient data.
    """
    
    def __init__(self, key=None):
        """
        Initialize encryptor.
        
        Args:
            key: Encryption key (bytes). If None, uses default or env var.
        """
        if key is None:
            key = get_encryption_key()
        
        if isinstance(key, str):
            key = key.encode()
        
        # Generate Fernet key
        if len(key) < 32:
            # Derive key from password
            import os
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key))
        elif len(key) > 32:
            key = key[:32]
        
        # Ensure key is URL-safe base64
        try:
            self.fernet = Fernet(key)
        except Exception:
            # Generate new key if invalid
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            log.warning("Invalid encryption key, generated new key")
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt string data.
        
        Args:
            data: String to encrypt
        
        Returns:
            Encrypted string (base64)
        """
        if not data:
            return ""
        
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            log.exception("Encryption failed: %s", e)
            return data  # Return unencrypted on error
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt string data.
        
        Args:
            encrypted_data: Encrypted string (base64)
        
        Returns:
            Decrypted string
        """
        if not encrypted_data:
            return ""
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            log.exception("Decryption failed: %s", e)
            return encrypted_data  # Return as-is on error
    
    def encrypt_dict(self, data: dict) -> dict:
        """
        Encrypt sensitive fields in dictionary.
        
        Args:
            data: Dictionary to encrypt
        
        Returns:
            Dictionary with encrypted fields
        """
        # Fields to encrypt (patient identifiers, etc.)
        sensitive_fields = ['patient_id', 'device_id', 'kps', 'bbox']
        
        encrypted = data.copy()
        for field in sensitive_fields:
            if field in encrypted and encrypted[field]:
                if isinstance(encrypted[field], (dict, list)):
                    import json
                    encrypted[field] = self.encrypt(json.dumps(encrypted[field]))
                else:
                    encrypted[field] = self.encrypt(str(encrypted[field]))
        
        return encrypted
    
    def decrypt_dict(self, encrypted_data: dict) -> dict:
        """
        Decrypt sensitive fields in dictionary.
        
        Args:
            encrypted_data: Dictionary with encrypted fields
        
        Returns:
            Dictionary with decrypted fields
        """
        sensitive_fields = ['patient_id', 'device_id', 'kps', 'bbox']
        
        decrypted = encrypted_data.copy()
        for field in sensitive_fields:
            if field in decrypted and decrypted[field]:
                try:
                    decrypted_str = self.decrypt(decrypted[field])
                    import json
                    decrypted[field] = json.loads(decrypted_str)
                except Exception:
                    # If decryption fails, keep original
                    pass
        
        return decrypted

# Global encryptor instance
_encryptor = None

def get_encryptor():
    """Get or create global encryptor instance."""
    global _encryptor
    if _encryptor is None:
        _encryptor = DataEncryptor()
    return _encryptor

