from pydantic_settings import BaseSettings


class MongoDBConfig(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "vnid_card_ocr"
    MONGODB_COLLECTION_RESULTS: str = "ocr_results"
    MONGODB_COLLECTION_SESSIONS: str = "sessions"
    MONGODB_COLLECTION_METRICS: str = "metrics"

    class Config:
        env_file = ".env"


mongodb_config = MongoDBConfig()
