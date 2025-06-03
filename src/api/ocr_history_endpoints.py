"""
Fixed OCR history endpoint implementation to return the complete OCR data.
"""


@self.app.get("/ocr-history")
async def get_ocr_history(
    limit: int = 100,
    skip: int = 0,
    sort_by: str = "timestamp",
    sort_order: int = -1,
    session_id: Optional[str] = None,
    success: Optional[bool] = None,
    filename: Optional[str] = None
):
    """
    Retrieve complete OCR processing data.

    Args:
        limit: Maximum number of results to return (default: 100)
        skip: Number of results to skip for pagination (default: 0)
        sort_by: Field to sort by (default: timestamp)
        sort_order: Sort order, 1 for ascending, -1 for descending (default: -1)
        session_id: Filter by session ID
        success: Filter by success status
        filename: Filter by image filename

    Returns:
        Complete OCR results including all extracted data with pagination info
    """
    try:
        # Check if MongoDB is connected
        if not hasattr(self, 'mongo_available') or not self.mongo_available:
            logger.error("MongoDB not connected, cannot retrieve OCR history")
            return {
                "status": "error",
                "message": "Database not available",
                "results": [],
                "count": 0,
                "pagination": {
                    "limit": limit,
                    "skip": skip,
                    "total": 0
                }
            }

        # Build filter criteria
        filter_criteria = {}
        if session_id:
            filter_criteria["session_id"] = session_id
        if success is not None:
            filter_criteria["success"] = success
        if filename:
            filter_criteria["image_filename"] = {
                "$regex": filename, "$options": "i"}

        # Get OCR results from MongoDB with filters
        results = self.db_client.get_all_ocr_results(
            limit=limit,
            skip=skip,
            sort_by=sort_by,
            sort_order=sort_order,
            filter_criteria=filter_criteria
        )

        # Get filtered count for pagination
        total_count = self.db_client.get_ocr_results_count(filter_criteria)

        logger.info(
            f"Retrieved {len(results)} complete OCR data records from MongoDB")

        return {
            "status": "success",
            "message": f"Retrieved {len(results)} complete OCR records",
            "results": results,
            "count": len(results),
            "pagination": {
                "limit": limit,
                "skip": skip,
                "total": total_count
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving OCR history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve OCR history: {str(e)}"
        )


@self.app.get("/ocr-history/session/{session_id}")
async def get_ocr_history_by_session(session_id: str):
    """
    Get complete OCR data for a specific session ID.

    Args:
        session_id: The session ID to retrieve OCR data for

    Returns:
        Complete OCR results including all extracted data for the session
    """
    try:
        # Check if MongoDB is connected
        if not hasattr(self, 'mongo_available') or not self.mongo_available:
            logger.error(
                "MongoDB not connected, cannot retrieve OCR data by session")
            return {
                "status": "error",
                "message": "Database not available",
                "results": []
            }

        # Get complete OCR results for the session
        results = self.db_client.get_ocr_results_by_session(session_id)

        logger.info(
            f"Retrieved {len(results)} complete OCR results for session: {session_id}")

        return {
            "status": "success",
            "message": f"Retrieved {len(results)} complete OCR records for session: {session_id}",
            "results": results
        }
    except Exception as e:
        logger.error(
            f"Error retrieving OCR history for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve OCR history: {str(e)}"
        )
