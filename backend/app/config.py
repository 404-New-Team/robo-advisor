from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Robo-Advisor Backend"
    app_version: str = "1.0.0"

    mysql_host: str = "mysql"
    mysql_port: int = 3306
    mysql_user: str = "robo"
    mysql_password: str = "robopassword"
    mysql_db: str = "roboadvisor"

    ai_service_url: str = "http://ai:8001"
    ai_timeout: float = 30.0
    request_timeout: float = 30.0

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24시간


settings = Settings()
