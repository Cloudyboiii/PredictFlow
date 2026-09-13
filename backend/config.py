from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    MAX_FILE_SIZE_MB: int = 25
    MAX_ROWS: int = 50000
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
