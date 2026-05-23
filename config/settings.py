"""
config/settings.py
CorePilora AI — Application Settings

Single source of truth for all configuration.
All environment variables loaded here.
"""

from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator(
        "APP_DOMAIN", "ELEVENLABS_VOICE_ID", "ELEVENLABS_API_KEY",
        "GROQ_API_KEY", "DEEPGRAM_API_KEY", "REDIS_URL", "DATABASE_URL",
        "TELNYX_API_KEY", "TELNYX_PHONE_NUMBER",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Security
    WEBHOOK_SECRET: str = "changeme"

    # AI
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-70b-8192"

    # Telnyx (replaces Twilio)
    TELNYX_API_KEY: str
    TELNYX_PHONE_NUMBER: str
    TELNYX_PUBLIC_KEY: str = ""

    # Deepgram
    DEEPGRAM_API_KEY: str

    # ElevenLabs
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str

    # HubSpot CRM
    HUBSPOT_ACCESS_TOKEN: str = ""
    HUBSPOT_CLIENT_SECRET: str = ""
    HUBSPOT_PORTAL_ID: str = ""

    # App
    APP_ENV: str = "development"
    ISA_NAME: str = "Jaiyana"
    PRIMARY_MARKET: str = "Dallas-Houston"

    # Domain — used for callback URLs
    APP_DOMAIN: str = "localhost"

    # CORS — comma-separated allowed origins. Use "*" only in development.
    ALLOWED_ORIGINS: str = "*"


settings = Settings()
