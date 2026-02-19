# tests/test_security.py
"""
Security Tests for CI/CD Pipeline
Catches security vulnerabilities before deployment.
"""

import os
import re
import pytest
import yaml
import sqlite3
from pathlib import Path
from typing import List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestHardcodedCredentials:
 """Test for hardcoded credentials in codebase."""

 # Patterns that indicate hardcoded secrets
 SECRET_PATTERNS = [
 (r'password\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded password"),
 (r'api[_\s]?key\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded API key"),
 (r'secret\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded secret"),
 (r'token\s*[=:]\s*["\'][a-zA-Z0-9]{20,}["\']', "Hardcoded token"),
 (r'private[_\s]?key\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded private key"),
 # Allow hashes but flag if comment reveals plaintext
 (r'#.*(?:password|secret).*["\'][^"\']{3,}["\']', "Password in comment"),
 ]

 # Files to exclude from scanning
 EXCLUDE_PATTERNS = [
 r'\.git/',
 r'__pycache__/',
 r'\.pyc$',
 r'node_modules/',
 r'venv/',
 r'\.env\.example$', # Example files are OK
 r'test_security\.py$', # This file
 ]

 def _should_exclude(self, filepath: str) -> bool:
 """Check if file should be excluded from scanning."""
 for pattern in self.EXCLUDE_PATTERNS:
 if re.search(pattern, str(filepath)):
 return True
 return False

 def _scan_file(self, filepath: Path) -> List[Tuple[int, str, str]]:
 """Scan a file for hardcoded secrets."""
 findings = []

 try:
 content = filepath.read_text(encoding='utf-8', errors='ignore')
 lines = content.split('\n')

 for line_num, line in enumerate(lines, 1):
 for pattern, description in self.SECRET_PATTERNS:
 if re.search(pattern, line, re.IGNORECASE):
 # Skip if it's clearly a variable assignment from env
 if 'os.environ' in line or 'getenv' in line:
 continue
 findings.append((line_num, description, line.strip()[:100]))

 except Exception as e:
 pass # Skip unreadable files

 return findings

 def test_no_hardcoded_credentials_in_python(self):
 """Ensure no hardcoded credentials in Python files."""
 findings = []

 for filepath in PROJECT_ROOT.rglob("*.py"):
 if self._should_exclude(filepath):
 continue

 file_findings = self._scan_file(filepath)
 for line_num, desc, line in file_findings:
 findings.append(f"{filepath}:{line_num} - {desc}: {line}")

 assert len(findings) == 0, \
 f"Found hardcoded credentials:\n" + "\n".join(findings)

 def test_no_hardcoded_credentials_in_yaml(self):
 """Ensure no hardcoded credentials in YAML config files."""
 findings = []

 for filepath in PROJECT_ROOT.rglob("*.yaml"):
 if self._should_exclude(filepath):
 continue
 if 'example' in filepath.name.lower():
 continue

 file_findings = self._scan_file(filepath)
 for line_num, desc, line in file_findings:
 findings.append(f"{filepath}:{line_num} - {desc}: {line}")

 assert len(findings) == 0, \
 f"Found hardcoded credentials in YAML:\n" + "\n".join(findings)


class TestMQTTSecurity:
 """Test MQTT security configuration."""

 def test_mqtt_uses_tls_port(self):
 """Ensure MQTT is configured for TLS (port 8883)."""
 mqtt_config = PROJECT_ROOT / "config" / "mqtt.yaml"

 if mqtt_config.exists():
 with open(mqtt_config) as f:
 config = yaml.safe_load(f)

 port = config.get("port", 1883)

 # Warn if using non-TLS port
 if port == 1883:
 pytest.fail(
 "MQTT configured for non-TLS port 1883. "
 "Use port 8883 for TLS encryption."
 )

 def test_mqtt_has_authentication(self):
 """Ensure MQTT requires authentication."""
 mqtt_config = PROJECT_ROOT / "config" / "mqtt.yaml"

 if mqtt_config.exists():
 with open(mqtt_config) as f:
 config = yaml.safe_load(f)

 # Check for username/password or reference to env vars
 has_auth = (
 config.get("username") or
 config.get("password") or
 config.get("use_tls") or
 "${" in str(config) # Environment variable reference
 )

 if not has_auth:
 pytest.fail(
 "MQTT has no authentication configured. "
 "Add username/password or TLS client certificates."
 )


class TestDatabaseSecurity:
 """Test database security practices."""

 def test_no_sql_string_formatting(self):
 """Ensure no SQL injection vulnerabilities via string formatting."""
 vulnerable_patterns = [
 r'execute\([^)]*%\s*[^)]+\)', # execute("..." % something)
 r'execute\([^)]*\.format\(', # execute("...".format())
 r'execute\(f["\']', # execute(f"...")
 r'executemany\([^)]*%\s*[^)]+\)',
 ]

 findings = []

 for filepath in PROJECT_ROOT.rglob("*.py"):
 if self._should_exclude(filepath):
 continue

 try:
 content = filepath.read_text()
 lines = content.split('\n')

 for line_num, line in enumerate(lines, 1):
 for pattern in vulnerable_patterns:
 if re.search(pattern, line):
 findings.append(
 f"{filepath}:{line_num} - Potential SQL injection: {line.strip()[:80]}"
 )
 except:
 pass

 assert len(findings) == 0, \
 f"Found potential SQL injection:\n" + "\n".join(findings)

 def _should_exclude(self, filepath: Path) -> bool:
 """Check if file should be excluded."""
 exclude = ['test_security.py', '__pycache__', '.git', 'venv']
 return any(x in str(filepath) for x in exclude)

 def test_database_uses_parameterized_queries(self):
 """Verify database uses parameterized queries."""
 db_file = PROJECT_ROOT / "storage" / "db.py"

 if db_file.exists():
 content = db_file.read_text()

 # Count parameterized vs non-parameterized
 parameterized = len(re.findall(r'execute\([^)]*\?', content))
 string_format = len(re.findall(r'execute\([^)]*%', content))
 f_string = len(re.findall(r'execute\(f["\']', content))

 total_unsafe = string_format + f_string

 if total_unsafe > 0:
 pytest.fail(
 f"Found {total_unsafe} potentially unsafe SQL queries. "
 f"Use parameterized queries (?) instead."
 )


class TestEncryption:
 """Test encryption implementation."""

 def test_encryption_uses_random_salt(self):
 """Ensure encryption uses random (not deterministic) salt."""
 encrypt_file = PROJECT_ROOT / "storage" / "encrypt.py"

 if encrypt_file.exists():
 content = encrypt_file.read_text()

 # Check for deterministic salt
 if 'sha256(key)' in content and 'salt' in content:
 pytest.fail(
 "Encryption uses deterministic salt derived from key. "
 "Use os.urandom() or secrets.token_bytes() for salt."
 )

 def test_face_images_encrypted(self):
 """Ensure face images are not stored as plaintext."""
 face_recognition = PROJECT_ROOT / "pipeline" / "patient" / "face_recognition.py"

 if face_recognition.exists():
 content = face_recognition.read_text()

 # Check for unencrypted image writes
 if 'cv2.imwrite' in content:
 # Check if encryption is applied
 if 'encrypt' not in content.lower():
 pytest.fail(
 "Face images may be stored unencrypted. "
 "Encrypt biometric data before storage."
 )


class TestPHILogging:
 """Test that PHI is not logged in plaintext."""

 PHI_PATTERNS = [
 r'log\.\w+\([^)]*patient[_\s]?id[^)]*\)', # log.xxx(...patient_id...)
 r'log\.\w+\([^)]*%s.*patient', # log.xxx("...%s", patient...)
 r'logging\.\w+\([^)]*patient[_\s]?id',
 ]

 def test_no_phi_in_logs(self):
 """Ensure patient IDs are not logged directly."""
 findings = []

 for filepath in PROJECT_ROOT.rglob("*.py"):
 if 'test_' in filepath.name or 'hipaa_logging' in filepath.name:
 continue

 try:
 content = filepath.read_text()
 lines = content.split('\n')

 for line_num, line in enumerate(lines, 1):
 for pattern in self.PHI_PATTERNS:
 if re.search(pattern, line, re.IGNORECASE):
 # Check if using HIPAA logger
 if 'HIPAALogger' not in content:
 findings.append(
 f"{filepath}:{line_num} - PHI may be logged: {line.strip()[:60]}"
 )
 except:
 pass

 if findings:
 pytest.fail(
 f"Found potential PHI logging:\n" + "\n".join(findings[:10]) +
 f"\n... and {len(findings) - 10} more" if len(findings) > 10 else ""
 )


class TestErrorHandling:
 """Test that errors don't silently downgrade clinical alerts."""

 def test_no_silent_low_risk_on_error(self):
 """Ensure exceptions don't silently return LOW_RISK."""
 decision_engine = PROJECT_ROOT / "pipeline" / "pose" / "decision_engine.py"

 if decision_engine.exists():
 content = decision_engine.read_text()

 # Check for LOW_RISK in exception handlers
 in_except = False
 for line in content.split('\n'):
 if 'except' in line:
 in_except = True
 elif in_except and 'LOW_RISK' in line:
 pytest.fail(
 "Decision engine returns LOW_RISK on exception. "
 "This could mask critical alerts. Raise exception or return ERROR state."
 )
 elif in_except and ('return' in line or 'raise' in line):
 in_except = False

 def test_bare_except_clauses(self):
 """Find bare except clauses that swallow all errors."""
 findings = []

 for filepath in PROJECT_ROOT.rglob("*.py"):
 if '__pycache__' in str(filepath):
 continue

 try:
 content = filepath.read_text()
 lines = content.split('\n')

 for line_num, line in enumerate(lines, 1):
 # Match 'except:' but not 'except SomeError:'
 if re.match(r'\s*except\s*:\s*$', line):
 findings.append(f"{filepath}:{line_num}")

 except:
 pass

 if len(findings) > 5: # Allow some but not too many
 pytest.fail(
 f"Found {len(findings)} bare 'except:' clauses that swallow all errors:\n" +
 "\n".join(findings[:10])
 )


class TestThreadSafety:
 """Test thread safety of database operations."""

 def test_database_thread_safety(self):
 """Ensure database doesn't disable thread safety unsafely."""
 db_file = PROJECT_ROOT / "storage" / "db.py"

 if db_file.exists():
 content = db_file.read_text()

 # Check for disabled thread safety
 if 'check_same_thread=False' in content:
 # Check if connection pooling is used
 if 'pool' not in content.lower() and 'queue' not in content.lower():
 pytest.fail(
 "Database uses check_same_thread=False without connection pooling. "
 "This can cause race conditions and data corruption."
 )


# Run tests
if __name__ == "__main__":
 pytest.main([__file__, "-v"])
