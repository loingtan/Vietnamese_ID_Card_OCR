"""
MongoDB client for Vietnamese ID Card OCR application - Simplified version.
Only handles OCRResult data.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from src.config import get_config
from .models import OCRResult

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client for OCR application - simplified version."""

    def __init__(self):
        """Initialize MongoDB client."""
        self.config = get_config()
        self._client: Optional[MongoClient] = None
        self._async_client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[Database] = None
        self._async_db: Optional[AsyncIOMotorDatabase] = None
        self._connected = False

    def connect(self):
        """Establish synchronous connection to MongoDB."""
        try:
            self._client = MongoClient(self.config.MONGODB_URL)
            self._db = self._client[self.config.MONGODB_DATABASE]

            # Test connection
            self._client.admin.command('ping')
            self._connected = True
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            raise

    async def connect_async(self):
        """Establish asynchronous connection to MongoDB."""
        try:
            self._async_client = AsyncIOMotorClient(self.config.MONGODB_URL)
            self._async_db = self._async_client[self.config.MONGODB_DATABASE]

            # Test connection
            await self._async_client.admin.command('ping')
            self._connected = True
            logger.info("MongoDB async connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB async: {e}")
            self._connected = False
            raise

    def disconnect(self):
        """Close database connections."""
        if self._client:
            self._client.close()
            self._client = None
        if self._async_client:
            self._async_client.close()
            self._async_client = None
        self._db = None
        self._async_db = None
        self._connected = False
        logger.info("MongoDB connections closed")

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected

    # OCR Results operations
    def save_ocr_result(self, result: OCRResult) -> str:
        """Save OCR result to database."""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]
        result_dict = result.to_dict()
        result_dict["timestamp"] = datetime.utcnow()

        inserted = collection.insert_one(result_dict)
        logger.info(f"Saved OCR result with ID: {inserted.inserted_id}")
        return str(inserted.inserted_id)

    async def save_ocr_result_async(self, result: OCRResult) -> str:
        """Save OCR result to database (async)."""
        if not self._connected or self._async_db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._async_db[self.config.MONGODB_COLLECTION_RESULTS]
        result_dict = result.to_dict()
        result_dict["timestamp"] = datetime.utcnow()

        inserted = await collection.insert_one(result_dict)
        logger.info(f"Saved OCR result with ID: {inserted.inserted_id}")
        return str(inserted.inserted_id)

    def get_ocr_results_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all OCR results for a session."""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]
        results = list(collection.find(
            {"session_id": session_id}, {"_id": 0}).sort("timestamp", -1))

        logger.info(
            f"Retrieved {len(results)} OCR results for session: {session_id}")
        return results

    async def get_ocr_results_by_session_async(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all OCR results for a session (async)."""
        if not self._connected or self._async_db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._async_db[self.config.MONGODB_COLLECTION_RESULTS]
        cursor = collection.find(
            {"session_id": session_id}, {"_id": 0}).sort("timestamp", -1)
        results = await cursor.to_list(length=None)

        logger.info(
            f"Retrieved {len(results)} OCR results for session: {session_id} (async)")
        return results

    def get_all_ocr_results(self, limit: int = 100, skip: int = 0, sort_by: str = "timestamp",
                            sort_order: int = -1, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get all OCR results with pagination and filtering."""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]

        # Build query filter
        query_filter = filter_criteria if filter_criteria else {}

        # Create sort specification
        sort_spec = [(sort_by, sort_order)]

        # Query with pagination and filtering
        cursor = collection.find(query_filter, {"_id": 0}).sort(
            sort_spec).skip(skip).limit(limit)

        # Convert cursor to list
        results = list(cursor)
        logger.info(
            f"Retrieved {len(results)} OCR results with filter: {query_filter}")

        return results

    async def get_all_ocr_results_async(self, limit: int = 100, skip: int = 0,
                                        sort_by: str = "timestamp", sort_order: int = -1,
                                        filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get all OCR results with pagination and filtering (async)."""
        if not self._connected or self._async_db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._async_db[self.config.MONGODB_COLLECTION_RESULTS]

        # Build query filter
        query_filter = filter_criteria if filter_criteria else {}

        # Create sort specification
        sort_spec = [(sort_by, sort_order)]

        # Query with pagination and filtering
        cursor = collection.find(query_filter, {'_id': 0}).sort(
            sort_spec).skip(skip).limit(limit)

        # Convert cursor to list
        results = await cursor.to_list(length=limit)
        logger.info(
            f"Retrieved {len(results)} OCR results with filter: {query_filter} (async)")

        return results

    def get_ocr_results_count(self, filter_criteria: Dict[str, Any] = None) -> int:
        """Get count of OCR results."""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]
        query_filter = filter_criteria if filter_criteria else {}
        count = collection.count_documents(query_filter)

        return count

    async def get_ocr_results_count_async(self, filter_criteria: Dict[str, Any] = None) -> int:
        """Get count of OCR results (async)."""
        if not self._connected or self._async_db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._async_db[self.config.MONGODB_COLLECTION_RESULTS]
        query_filter = filter_criteria if filter_criteria else {}
        count = await collection.count_documents(query_filter)

        return count

    def search_by_id_number(self, id_number: str) -> List[Dict[str, Any]]:
        """Search OCR results by ID number."""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]

        # Search in extracted_info field for ID number
        query = {
            "$or": [
                {"extracted_info.id_number": {"$regex": id_number, "$options": "i"}},
                {"extracted_info.ID_number": {"$regex": id_number, "$options": "i"}},
                {"extracted_info.Id_number": {"$regex": id_number, "$options": "i"}}
            ]
        }

        results = list(collection.find(
            query, {"_id": 0}).sort("timestamp", -1))
        logger.info(
            f"Found {len(results)} OCR results for ID number: {id_number}")

        return results

    def delete_ocr_result(self, session_id: str, filename: str) -> bool:
        """Delete a specific OCR result."""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]
        result = collection.delete_one({
            "session_id": session_id,
            "image_filename": filename
        })

        success = result.deleted_count > 0
        if success:
            logger.info(
                f"Deleted OCR result for session {session_id}, file {filename}")
        else:
            logger.warning(
                f"No OCR result found for session {session_id}, file {filename}")

        return success

    def clear_all_data(self):
        """Clear all OCR results from database. Use with caution!"""
        if not self._connected or self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        collection = self._db[self.config.MONGODB_COLLECTION_RESULTS]
        result = collection.delete_many({})
        logger.warning(
            f"Cleared {result.deleted_count} OCR results from database")

        return result.deleted_count


# Global database client instance
db_client = MongoDBClient()
