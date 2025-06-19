#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để xem nội dung cơ sở dữ liệu SQLite
Database Viewer for Emotion Recognition System
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

def connect_db():
    """Kết nối tới cơ sở dữ liệu"""
    if not os.path.exists('emotion_app.db'):
        print("❌ Không tìm thấy file cơ sở dữ liệu: emotion_app.db")
        return None
    
    try:
        conn = sqlite3.connect('emotion_app.db')
        print("✅ Kết nối thành công tới cơ sở dữ liệu!")
        return conn
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return None

def show_tables(conn):
    """Hiển thị danh sách các bảng"""
    print("\n" + "="*50)
    print("📋 DANH SÁCH CÁC BẢNG TRONG CSDL")
    print("="*50)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table[0]}")
    
    return [table[0] for table in tables]

def show_table_structure(conn, table_name):
    """Hiển thị cấu trúc bảng"""
    print(f"\n" + "="*50)
    print(f"🏗️  CẤU TRÚC BẢNG: {table_name.upper()}")
    print("="*50)
    
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    print(f"{'STT':<4} {'Tên cột':<15} {'Kiểu DL':<12} {'NOT NULL':<10} {'Mặc định':<12} {'PK':<5}")
    print("-" * 70)
    
    for col in columns:
        cid, name, dtype, notnull, default, pk = col
        print(f"{cid+1:<4} {name:<15} {dtype:<12} {'Yes' if notnull else 'No':<10} {str(default) if default else 'NULL':<12} {'Yes' if pk else 'No':<5}")

def show_table_data(conn, table_name, limit=10):
    """Hiển thị dữ liệu trong bảng"""
    print(f"\n" + "="*50)
    print(f"📊 DỮ LIỆU BẢNG: {table_name.upper()} (Giới hạn {limit} dòng)")
    print("="*50)
    
    try:
        # Đếm tổng số dòng
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"📈 Tổng số dòng: {total_rows}")
        
        if total_rows == 0:
            print("⚠️  Bảng rỗng!")
            return
        
        # Lấy dữ liệu mẫu
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        print(f"\n📋 Dữ liệu mẫu ({min(limit, total_rows)} dòng đầu):")
        print(df.to_string(index=False))
        
        if total_rows > limit:
            print(f"\n... và {total_rows - limit} dòng khác")
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc dữ liệu: {e}")

def show_user_statistics(conn):
    """Hiển thị thống kê người dùng"""
    print(f"\n" + "="*50)
    print("👥 THỐNG KÊ NGƯỜI DÙNG")
    print("="*50)
    
    try:
        # Tổng số người dùng
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        print(f"👤 Tổng số người dùng: {total_users}")
        
        # Người dùng tạo gần đây
        cursor.execute("SELECT username, created_at FROM users ORDER BY created_at DESC LIMIT 5")
        recent_users = cursor.fetchall()
        print(f"\n🆕 Người dùng tạo gần đây:")
        for user in recent_users:
            print(f"   • {user[0]} - {user[1]}")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def show_emotion_statistics(conn):
    """Hiển thị thống kê cảm xúc"""
    print(f"\n" + "="*50)
    print("😊 THỐNG KÊ CẢM XÚC")
    print("="*50)
    
    try:
        # Tổng số phân tích
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emotion_logs")
        total_logs = cursor.fetchone()[0]
        print(f"📊 Tổng số lần phân tích: {total_logs}")
        
        if total_logs == 0:
            print("⚠️  Chưa có dữ liệu cảm xúc!")
            return
        
        # Thống kê theo cảm xúc
        cursor.execute("""
            SELECT emotion, COUNT(*) as count, 
                   ROUND(AVG(confidence), 3) as avg_confidence
            FROM emotion_logs 
            GROUP BY emotion 
            ORDER BY count DESC
        """)
        emotion_stats = cursor.fetchall()
        
        print(f"\n📈 Phân bố cảm xúc:")
        print(f"{'Cảm xúc':<12} {'Số lần':<8} {'Tỷ lệ':<8} {'Độ tin cậy TB':<15}")
        print("-" * 50)
        
        for emotion, count, avg_conf in emotion_stats:
            percentage = (count / total_logs) * 100
            print(f"{emotion:<12} {count:<8} {percentage:>5.1f}% {avg_conf:<15}")
        
        # Top users
        cursor.execute("""
            SELECT u.username, COUNT(el.id) as detection_count
            FROM users u
            LEFT JOIN emotion_logs el ON u.id = el.user_id
            GROUP BY u.id, u.username
            ORDER BY detection_count DESC
            LIMIT 5
        """)
        top_users = cursor.fetchall()
        
        print(f"\n🏆 Top người dùng hoạt động:")
        for user, count in top_users:
            print(f"   • {user}: {count} lần phân tích")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def main():
    """Hàm chính"""
    print("🎯 CÔNG CỤ XEM CƠ SỞ DỮ LIỆU - EMOTION RECOGNITION")
    print("=" * 60)
    
    # Kết nối database
    conn = connect_db()
    if not conn:
        return
    
    try:
        # Hiển thị danh sách bảng
        tables = show_tables(conn)
        
        # Hiển thị cấu trúc và dữ liệu từng bảng
        for table in tables:
            show_table_structure(conn, table)
            show_table_data(conn, table, limit=5)
        
        # Hiển thị thống kê
        show_user_statistics(conn)
        show_emotion_statistics(conn)
        
        print(f"\n" + "="*60)
        print("✅ Hoàn thành xem cơ sở dữ liệu!")
        print("💡 Mẹo: Bạn có thể dùng DB Browser for SQLite để xem GUI")
        print("📥 Tải tại: https://sqlitebrowser.org/")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 