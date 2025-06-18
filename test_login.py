#!/usr/bin/env python3
"""
Script để test và debug login system
"""

from database import init_db, add_user, verify_user, get_all_users, reset_database, create_demo_user
import os

def test_database():
    print("🔍 Testing Database Functions...")
    print("=" * 50)
    
    # Initialize database
    print("1. Initializing database...")
    init_db()
    
    # Check if demo user exists
    print("\n2. Checking existing users...")
    users = get_all_users()
    print(f"Found {len(users)} users:")
    for user in users:
        print(f"   - ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")
    
    # Test demo login
    print("\n3. Testing demo user login...")
    user_id = verify_user('demo', 'demo123')
    if user_id:
        print(f"✅ Demo login successful! User ID: {user_id}")
    else:
        print("❌ Demo login failed!")
        print("Creating demo user...")
        if add_user('demo', 'demo123', 'demo@example.com'):
            print("✅ Demo user created successfully")
            user_id = verify_user('demo', 'demo123')
            print(f"✅ Demo login now works! User ID: {user_id}")
    
    # Test creating new user
    print("\n4. Testing new user creation...")
    if add_user('testuser', 'password123', 'test@example.com'):
        print("✅ Test user created successfully")
        test_user_id = verify_user('testuser', 'password123')
        if test_user_id:
            print(f"✅ Test user login successful! User ID: {test_user_id}")
        else:
            print("❌ Test user login failed!")
    else:
        print("❌ Test user creation failed (may already exist)")
    
    # Final user list
    print("\n5. Final user list...")
    users = get_all_users()
    print(f"Total users: {len(users)}")
    for user in users:
        print(f"   - ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")
    
    print("\n" + "=" * 50)
    print("🎯 Quick Test Credentials:")
    print("Username: demo")
    print("Password: demo123")
    print("=" * 50)

def reset_and_test():
    print("🗑️ Resetting database and testing from scratch...")
    reset_database()
    test_database()

if __name__ == "__main__":
    print("🚀 Login System Test Script")
    print("Choose an option:")
    print("1. Test current database")
    print("2. Reset database and test")
    print("3. Just create demo user")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_database()
    elif choice == "2":
        reset_and_test()
    elif choice == "3":
        create_demo_user()
    else:
        print("Invalid choice, running basic test...")
        test_database() 