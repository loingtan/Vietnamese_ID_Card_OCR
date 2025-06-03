"""
Script to clean up all data in MongoDB database.
"""

from src.database import MongoDBClient
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_database():
    """Clean up all data in MongoDB database."""
    try:
        # Initialize MongoDB client
        db_client = MongoDBClient()
        db_client.connect()
        
        # Get collections
        db = db_client._db
        collections = [
            db_client.config.MONGODB_COLLECTION_RESULTS,
            db_client.config.MONGODB_COLLECTION_SESSIONS,
            db_client.config.MONGODB_COLLECTION_METRICS
        ]
        
        # Delete all documents from each collection
        for collection_name in collections:
            collection = db[collection_name]
            result = collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} documents from {collection_name}")
            
        logger.info("Database cleanup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")
    finally:
        if db_client:
            db_client.disconnect()

if __name__ == "__main__":
    cleanup_database() 