# security/face_encryption.py
"""
HIPAA-compliant face image encryption for patient biometric data.
Encrypts face images at rest using AES-256-GCM.
"""
import os
import io
import logging
import hashlib
import secrets
from typing import Optional, Tuple
from pathlib import Path

log = logging.getLogger("face_encryption")

# Try to import cryptography
try:
 from cryptography.hazmat.primitives.ciphers.aead import AESGCM
 from cryptography.hazmat.primitives import hashes
 from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
 from cryptography.hazmat.backends import default_backend
 CRYPTO_AVAILABLE = True
except ImportError:
 CRYPTO_AVAILABLE = False
 log.warning("cryptography not available. Install with: pip install cryptography")

# Try to import cv2 and numpy
try:
 import cv2
 import numpy as np
 CV2_AVAILABLE = True
except ImportError:
 CV2_AVAILABLE = False
 log.warning("OpenCV not available for face image processing")


class FaceImageEncryptor:
 """
 Encrypts and decrypts face images using AES-256-GCM.

 Security features:
 - AES-256-GCM authenticated encryption
 - Unique nonce per encryption
 - Key derivation from master key using PBKDF2
 - Secure key storage recommendations
 """

 # File extension for encrypted images
 ENCRYPTED_EXTENSION = ".enc"

 # Nonce size (96 bits recommended for GCM)
 NONCE_SIZE = 12

 # Salt size for key derivation
 SALT_SIZE = 16

 # PBKDF2 iterations (high for security)
 KDF_ITERATIONS = 100000

 def __init__(self, master_key: Optional[bytes] = None, key_file: Optional[str] = None):
 """
 Initialize face image encryptor.

 Args:
 master_key: 32-byte master encryption key (if not provided, loaded from file or env)
 key_file: Path to key file (default: storage/.face_encryption_key)
 """
 if not CRYPTO_AVAILABLE:
 raise ImportError("cryptography library required for face encryption")

 self.key_file = key_file or "storage/.face_encryption_key"
 self._master_key = master_key or self._load_or_generate_key()

 log.info("FaceImageEncryptor initialized")

 def _load_or_generate_key(self) -> bytes:
 """Load master key from file/env or generate new one."""
 # Try environment variable first
 env_key = os.environ.get("EAM_FACE_ENCRYPTION_KEY")
 if env_key:
 try:
 return bytes.fromhex(env_key)
 except ValueError:
 log.warning("Invalid EAM_FACE_ENCRYPTION_KEY format, generating new key")

 # Try key file
 if os.path.exists(self.key_file):
 try:
 with open(self.key_file, "rb") as f:
 key = f.read()
 if len(key) == 32:
 log.info("Loaded face encryption key from: %s", self.key_file)
 return key
 except Exception as e:
 log.warning("Failed to load key file: %s", e)

 # Generate new key
 key = secrets.token_bytes(32)

 # Save key file with restricted permissions
 try:
 os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
 with open(self.key_file, "wb") as f:
 f.write(key)
 # Set restrictive permissions (owner read/write only)
 os.chmod(self.key_file, 0o600)
 log.info("Generated new face encryption key: %s", self.key_file)
 except Exception as e:
 log.warning("Could not save key file: %s. Key will be lost on restart.", e)

 return key

 def _derive_key(self, salt: bytes) -> bytes:
 """Derive encryption key from master key using PBKDF2."""
 kdf = PBKDF2HMAC(
 algorithm=hashes.SHA256(),
 length=32,
 salt=salt,
 iterations=self.KDF_ITERATIONS,
 backend=default_backend()
 )
 return kdf.derive(self._master_key)

 def encrypt_image(self, image: np.ndarray, patient_id: str) -> bytes:
 """
 Encrypt a face image.

 Args:
 image: Face image as numpy array (BGR format from OpenCV)
 patient_id: Patient ID (used as additional authenticated data)

 Returns:
 Encrypted image bytes (salt + nonce + ciphertext + tag)
 """
 if not CV2_AVAILABLE:
 raise ImportError("OpenCV required for face image encryption")

 # Encode image to PNG bytes
 success, img_bytes = cv2.imencode(".png", image)
 if not success:
 raise ValueError("Failed to encode image")

 plaintext = img_bytes.tobytes()

 # Generate unique salt and nonce
 salt = secrets.token_bytes(self.SALT_SIZE)
 nonce = secrets.token_bytes(self.NONCE_SIZE)

 # Derive key from master key
 key = self._derive_key(salt)

 # Encrypt with AES-GCM (authenticated encryption)
 aesgcm = AESGCM(key)

 # Use patient_id as additional authenticated data (AAD)
 # This ensures the ciphertext can only be decrypted for the correct patient
 aad = patient_id.encode("utf-8")

 ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

 # Return: salt || nonce || ciphertext (includes auth tag)
 return salt + nonce + ciphertext

 def decrypt_image(self, encrypted_data: bytes, patient_id: str) -> np.ndarray:
 """
 Decrypt a face image.

 Args:
 encrypted_data: Encrypted image bytes
 patient_id: Patient ID (must match original encryption)

 Returns:
 Decrypted face image as numpy array

 Raises:
 ValueError: If decryption fails (wrong key or tampered data)
 """
 if not CV2_AVAILABLE:
 raise ImportError("OpenCV required for face image decryption")

 if len(encrypted_data) < self.SALT_SIZE + self.NONCE_SIZE + 16:
 raise ValueError("Invalid encrypted data (too short)")

 # Extract components
 salt = encrypted_data[:self.SALT_SIZE]
 nonce = encrypted_data[self.SALT_SIZE:self.SALT_SIZE + self.NONCE_SIZE]
 ciphertext = encrypted_data[self.SALT_SIZE + self.NONCE_SIZE:]

 # Derive key
 key = self._derive_key(salt)

 # Decrypt with AES-GCM
 aesgcm = AESGCM(key)
 aad = patient_id.encode("utf-8")

 try:
 plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
 except Exception as e:
 raise ValueError(f"Decryption failed (wrong key or tampered data): {e}")

 # Decode image from PNG bytes
 img_array = np.frombuffer(plaintext, dtype=np.uint8)
 image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

 if image is None:
 raise ValueError("Failed to decode decrypted image")

 return image

 def encrypt_and_save(self, image: np.ndarray, patient_id: str, output_path: str) -> str:
 """
 Encrypt and save a face image to file.

 Args:
 image: Face image
 patient_id: Patient ID
 output_path: Output file path (will add .enc extension)

 Returns:
 Path to encrypted file
 """
 encrypted_data = self.encrypt_image(image, patient_id)

 # Ensure .enc extension
 if not output_path.endswith(self.ENCRYPTED_EXTENSION):
 output_path += self.ENCRYPTED_EXTENSION

 # Create directory if needed
 os.makedirs(os.path.dirname(output_path), exist_ok=True)

 # Write encrypted data
 with open(output_path, "wb") as f:
 f.write(encrypted_data)

 log.debug("Saved encrypted face image: %s", output_path)
 return output_path

 def load_and_decrypt(self, encrypted_path: str, patient_id: str) -> np.ndarray:
 """
 Load and decrypt a face image from file.

 Args:
 encrypted_path: Path to encrypted file
 patient_id: Patient ID

 Returns:
 Decrypted face image
 """
 with open(encrypted_path, "rb") as f:
 encrypted_data = f.read()

 return self.decrypt_image(encrypted_data, patient_id)

 def get_encrypted_path(self, original_path: str) -> str:
 """Get encrypted file path for an original image path."""
 return original_path + self.ENCRYPTED_EXTENSION

 def is_encrypted(self, file_path: str) -> bool:
 """Check if a file is encrypted (has .enc extension)."""
 return file_path.endswith(self.ENCRYPTED_EXTENSION)


class SecurePatientFaceStorage:
 """
 Secure storage for patient face images with encryption.
 Drop-in replacement for direct file storage.
 """

 def __init__(self, storage_dir: str = "storage/patient_faces_secure",
 encryptor: Optional[FaceImageEncryptor] = None):
 """
 Initialize secure face storage.

 Args:
 storage_dir: Directory for encrypted face images
 encryptor: FaceImageEncryptor instance (created if not provided)
 """
 self.storage_dir = storage_dir
 self.encryptor = encryptor or FaceImageEncryptor()

 os.makedirs(storage_dir, exist_ok=True)
 log.info("SecurePatientFaceStorage initialized: %s", storage_dir)

 def _get_face_path(self, patient_id: str) -> str:
 """Get encrypted file path for a patient."""
 # Hash patient_id to avoid PII in filename
 filename_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
 return os.path.join(self.storage_dir, f"{filename_hash}.enc")

 def save_face(self, patient_id: str, face_image: np.ndarray) -> str:
 """
 Save encrypted face image for patient.

 Args:
 patient_id: Patient ID
 face_image: Face image (BGR numpy array)

 Returns:
 Path to encrypted file
 """
 output_path = self._get_face_path(patient_id)

 # Remove .enc since encrypt_and_save adds it
 output_path_base = output_path[:-4] if output_path.endswith(".enc") else output_path

 return self.encryptor.encrypt_and_save(face_image, patient_id, output_path_base)

 def load_face(self, patient_id: str) -> Optional[np.ndarray]:
 """
 Load decrypted face image for patient.

 Args:
 patient_id: Patient ID

 Returns:
 Face image or None if not found
 """
 face_path = self._get_face_path(patient_id)

 if not os.path.exists(face_path):
 log.debug("No face image found for patient: %s", patient_id)
 return None

 try:
 return self.encryptor.load_and_decrypt(face_path, patient_id)
 except Exception as e:
 log.error("Failed to load face for patient %s: %s", patient_id, e)
 return None

 def has_face(self, patient_id: str) -> bool:
 """Check if encrypted face exists for patient."""
 return os.path.exists(self._get_face_path(patient_id))

 def delete_face(self, patient_id: str) -> bool:
 """
 Securely delete face image for patient.

 Args:
 patient_id: Patient ID

 Returns:
 True if deleted, False if not found
 """
 face_path = self._get_face_path(patient_id)

 if os.path.exists(face_path):
 try:
 # Overwrite with random data before deletion (secure delete)
 file_size = os.path.getsize(face_path)
 with open(face_path, "wb") as f:
 f.write(secrets.token_bytes(file_size))
 os.remove(face_path)
 log.info("Securely deleted face image for patient: %s", patient_id)
 return True
 except Exception as e:
 log.error("Failed to delete face for patient %s: %s", patient_id, e)
 return False

 return False

 def list_patients(self) -> list:
 """
 List all patients with stored face images.

 Note: Returns hashed filenames, not patient IDs (for privacy).
 """
 patients = []
 for filename in os.listdir(self.storage_dir):
 if filename.endswith(".enc"):
 patients.append(filename[:-4]) # Remove .enc
 return patients

 def get_storage_stats(self) -> dict:
 """Get storage statistics."""
 total_size = 0
 file_count = 0

 for filename in os.listdir(self.storage_dir):
 if filename.endswith(".enc"):
 file_path = os.path.join(self.storage_dir, filename)
 total_size += os.path.getsize(file_path)
 file_count += 1

 return {
 "storage_dir": self.storage_dir,
 "file_count": file_count,
 "total_size_bytes": total_size,
 "total_size_mb": round(total_size / (1024 * 1024), 2)
 }


def migrate_unencrypted_faces(source_dir: str, dest_storage: SecurePatientFaceStorage,
 patient_id_mapping: dict = None) -> dict:
 """
 Migrate unencrypted face images to secure encrypted storage.

 Args:
 source_dir: Directory with unencrypted face images
 dest_storage: SecurePatientFaceStorage instance
 patient_id_mapping: Optional dict mapping filename -> patient_id

 Returns:
 Migration statistics
 """
 if not CV2_AVAILABLE:
 raise ImportError("OpenCV required for face image migration")

 stats = {
 "migrated": 0,
 "failed": 0,
 "skipped": 0,
 "errors": []
 }

 for filename in os.listdir(source_dir):
 if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
 continue

 file_path = os.path.join(source_dir, filename)

 # Determine patient_id
 if patient_id_mapping and filename in patient_id_mapping:
 patient_id = patient_id_mapping[filename]
 else:
 # Use filename without extension as patient_id
 patient_id = os.path.splitext(filename)[0]

 try:
 # Load original image
 image = cv2.imread(file_path)
 if image is None:
 stats["failed"] += 1
 stats["errors"].append(f"Failed to load: {filename}")
 continue

 # Save encrypted
 dest_storage.save_face(patient_id, image)
 stats["migrated"] += 1

 log.info("Migrated face image: %s -> encrypted", filename)

 except Exception as e:
 stats["failed"] += 1
 stats["errors"].append(f"{filename}: {str(e)}")
 log.error("Failed to migrate %s: %s", filename, e)

 log.info("Migration complete: %d migrated, %d failed",
 stats["migrated"], stats["failed"])
 return stats
