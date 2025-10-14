"""
Service for managing "Remember Me" functionality with device-specific auto-login.
Uses SQLite for persistent token storage and device fingerprinting for security.
"""
import sqlite3
import secrets
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


class RememberMeService:
    """Manages remember me tokens for persistent login across sessions"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Store database in backend directory
            backend_dir = Path(__file__).parent.parent
            db_path = backend_dir / "remember_me.db"

        self.db_path = str(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS remember_me_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL,
                device_fingerprint TEXT NOT NULL,
                encrypted_password TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                last_used_at TIMESTAMP
            )
        ''')

        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_token_hash
            ON remember_me_tokens(token_hash)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_device_fingerprint
            ON remember_me_tokens(device_fingerprint)
        ''')

        conn.commit()
        conn.close()

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage"""
        return hashlib.sha256(token.encode()).hexdigest()

    def _simple_encrypt(self, text: str, key: str) -> str:
        """Simple XOR-based encryption (for demonstration; use proper encryption in production)"""
        key_bytes = key.encode()
        text_bytes = text.encode()
        encrypted = bytearray()

        for i, byte in enumerate(text_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        # Convert to hex string for storage
        return encrypted.hex()

    def _simple_decrypt(self, encrypted_hex: str, key: str) -> str:
        """Simple XOR-based decryption"""
        key_bytes = key.encode()
        encrypted_bytes = bytes.fromhex(encrypted_hex)
        decrypted = bytearray()

        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        return decrypted.decode()

    def create_token(self, username: str, password: str, device_fingerprint: str,
                     days_valid: int = 30) -> str:
        """
        Create a new remember me token for a user-device combination

        Args:
            username: User's username
            password: User's password (will be encrypted)
            device_fingerprint: Unique device identifier
            days_valid: Number of days the token should be valid

        Returns:
            The remember me token (to be stored in browser)
        """
        # Generate a secure random token
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(token)

        # Encrypt password using token as key
        encrypted_password = self._simple_encrypt(password, token)

        # Calculate expiration
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=days_valid)

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # First, remove any existing tokens for this user-device combination
        cursor.execute('''
            DELETE FROM remember_me_tokens
            WHERE username = ? AND device_fingerprint = ?
        ''', (username, device_fingerprint))

        # Insert new token
        cursor.execute('''
            INSERT INTO remember_me_tokens
            (token_hash, username, device_fingerprint, encrypted_password, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (token_hash, username, device_fingerprint, encrypted_password,
              created_at.isoformat(), expires_at.isoformat()))

        conn.commit()
        conn.close()

        return token

    def verify_token(self, token: str, device_fingerprint: str) -> Optional[Tuple[str, str]]:
        """
        Verify a remember me token and return credentials if valid

        Args:
            token: The remember me token
            device_fingerprint: Device identifier from the current request

        Returns:
            Tuple of (username, password) if valid, None otherwise
        """
        token_hash = self._hash_token(token)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT username, encrypted_password, device_fingerprint, expires_at
            FROM remember_me_tokens
            WHERE token_hash = ?
        ''', (token_hash,))

        result = cursor.fetchone()

        if not result:
            conn.close()
            return None

        username, encrypted_password, stored_fingerprint, expires_at_str = result

        # Verify device fingerprint matches
        if stored_fingerprint != device_fingerprint:
            conn.close()
            return None

        # Check if token is expired
        expires_at = datetime.fromisoformat(expires_at_str)
        if datetime.now() > expires_at:
            # Token expired, delete it
            cursor.execute('DELETE FROM remember_me_tokens WHERE token_hash = ?', (token_hash,))
            conn.commit()
            conn.close()
            return None

        # Update last used timestamp
        now_iso = datetime.now().isoformat()
        # Sliding expiration: extend expiry window on each successful verification
        new_expires_iso = (datetime.now() + timedelta(days=30)).isoformat()
        cursor.execute('''
            UPDATE remember_me_tokens
            SET last_used_at = ?, expires_at = ?
            WHERE token_hash = ?
        ''', (now_iso, new_expires_iso, token_hash))

        conn.commit()
        conn.close()

        # Decrypt password
        try:
            password = self._simple_decrypt(encrypted_password, token)
            return (username, password)
        except Exception:
            return None

    def remove_token(self, username: str, device_fingerprint: str = None):
        """
        Remove remember me token(s) for a user

        Args:
            username: User's username
            device_fingerprint: Optional device fingerprint. If provided, only removes
                              tokens for that device. Otherwise removes all tokens for user.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if device_fingerprint:
            cursor.execute('''
                DELETE FROM remember_me_tokens
                WHERE username = ? AND device_fingerprint = ?
            ''', (username, device_fingerprint))
        else:
            cursor.execute('''
                DELETE FROM remember_me_tokens
                WHERE username = ?
            ''', (username,))

        conn.commit()
        conn.close()

    def cleanup_expired_tokens(self):
        """Remove all expired tokens from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM remember_me_tokens
            WHERE expires_at < ?
        ''', (datetime.now().isoformat(),))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count
