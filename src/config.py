from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: str = "data"
    chroma_dir: str = "chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    alpha: float = 0.5

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()