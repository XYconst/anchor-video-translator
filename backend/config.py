from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    gemini_api_key: str = ""
    kie_api_key: str = ""
    elevenlabs_api_key: str
    auth_password: str = "changeme"
    upload_dir: str = os.path.join(os.path.dirname(__file__), "uploads")
    font_path: str = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "fonts", "Rubik-Bold.ttf")

    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(__file__), "..", ".env"))


settings = Settings()
