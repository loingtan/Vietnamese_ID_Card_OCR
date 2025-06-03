"""
Script to view MongoDB data for Vietnamese ID Card OCR application.
"""

from pymongo import MongoClient
from datetime import datetime
import json
from pprint import pprint

# MongoDB connection settings
MONGODB_URL = "mongodb://localhost:27017"
MONGODB_DATABASE = "vnid_card_ocr"
COLLECTIONS = {
    "ocr_results": "Kết quả OCR",
    "user_sessions": "Phiên người dùng",
    "processing_metrics": "Metrics xử lý"
}

def connect_mongodb():
    """Connect to MongoDB."""
    try:
        client = MongoClient(MONGODB_URL)
        db = client[MONGODB_DATABASE]
        print(f"✅ Đã kết nối thành công đến MongoDB: {MONGODB_DATABASE}")
        return db
    except Exception as e:
        print(f"❌ Lỗi kết nối MongoDB: {e}")
        return None

def format_datetime(dt):
    """Format datetime for display."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt

def view_collection_data(db, collection_name):
    """View data in a collection."""
    collection = db[collection_name]
    count = collection.count_documents({})
    print(f"\n📊 {COLLECTIONS[collection_name]}: {count} bản ghi")
    
    if count > 0:
        print("\nMẫu dữ liệu:")
        for doc in collection.find().limit(5):
            # Convert ObjectId to string
            doc['_id'] = str(doc['_id'])
            
            # Format datetime fields
            for key, value in doc.items():
                if isinstance(value, datetime):
                    doc[key] = format_datetime(value)
            
            print("\n" + "="*50)
            pprint(doc)

def main():
    """Main function."""
    print("🔍 Xem dữ liệu MongoDB - Vietnamese ID Card OCR")
    print("="*50)
    
    # Connect to MongoDB
    db = connect_mongodb()
    if db is None:
        return
    
    # View data in each collection
    for collection_name in COLLECTIONS:
        view_collection_data(db, collection_name)
    
    print("\n✅ Hoàn thành!")

if __name__ == "__main__":
    main() 