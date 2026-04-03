import sqlite3
import bcrypt
import streamlit as st
import os
from datetime import datetime

class AuthManager:
    """Manages user authentication and role-based access control."""
    
    def __init__(self, db_path="auth.db"):
        """Initialize authentication manager."""
        self.db_path = db_path
        self.initialize_auth_database()
        self.create_default_admin()
    
    def initialize_auth_database(self):
        """Create authentication database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                active INTEGER DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_token TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_default_admin(self):
        """Create default admin user if none exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'Admin'")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            # Create default admin
            password_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, email)
                VALUES (?, ?, ?, ?)
            """, ("admin", password_hash, "Admin", "admin@hospital.com"))
            conn.commit()
        
        conn.close()
    
    def hash_password(self, password):
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    def verify_password(self, password, password_hash):
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash)
    
    def create_user(self, username, password, role, email=None):
        """Create new user account."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if username already exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False
            
            # Hash password and create user
            password_hash = self.hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, email)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, role, email))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error creating user: {e}")
            return False
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT username, password_hash, role, email, active
                FROM users
                WHERE username = ?
            """, (username,))
            
            user_data = cursor.fetchone()
            
            if user_data and user_data[4] == 1:  # Check if user is active
                if self.verify_password(password, user_data[1]):
                    # Update last login
                    cursor.execute("""
                        UPDATE users 
                        SET last_login = CURRENT_TIMESTAMP
                        WHERE username = ?
                    """, (username,))
                    conn.commit()
                    
                    return {
                        'username': user_data[0],
                        'role': user_data[2],
                        'email': user_data[3]
                    }
            
            return None
            
        except sqlite3.Error as e:
            print(f"Error authenticating user: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_role(self, username):
        """Get user's role."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error getting user role: {e}")
            return None
        finally:
            conn.close()
    
    def update_user_role(self, username, new_role):
        """Update user's role (Admin only)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE users 
                SET role = ?
                WHERE username = ?
            """, (new_role, username))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            print(f"Error updating user role: {e}")
            return False
        finally:
            conn.close()
    
    def deactivate_user(self, username):
        """Deactivate user account."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE users 
                SET active = 0
                WHERE username = ?
            """, (username,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            print(f"Error deactivating user: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_users(self):
        """Get all users (Admin only)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT username, role, email, created_at, last_login, active
                FROM users
                ORDER BY created_at DESC
            """)
            
            columns = ['username', 'role', 'email', 'created_at', 'last_login', 'active']
            users = []
            
            for row in cursor.fetchall():
                user_dict = dict(zip(columns, row))
                users.append(user_dict)
            
            return users
            
        except sqlite3.Error as e:
            print(f"Error getting users: {e}")
            return []
        finally:
            conn.close()
    
    def change_password(self, username, old_password, new_password):
        """Change user password."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Verify old password
            cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            if result and self.verify_password(old_password, result[0]):
                # Update with new password
                new_password_hash = self.hash_password(new_password)
                cursor.execute("""
                    UPDATE users 
                    SET password_hash = ?
                    WHERE username = ?
                """, (new_password_hash, username))
                
                conn.commit()
                return True
            
            return False
            
        except sqlite3.Error as e:
            print(f"Error changing password: {e}")
            return False
        finally:
            conn.close()
    
    def require_role(self, required_roles):
        """Decorator to require specific roles for functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if st.session_state.get('user_role') in required_roles:
                    return func(*args, **kwargs)
                else:
                    st.error(f"Access denied. Required role: {', '.join(required_roles)}")
                    return None
            return wrapper
        return decorator
