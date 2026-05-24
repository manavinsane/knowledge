import os

from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "sraGbRmjYQXmYdnrgPk!OFE35UP6n/QqeoED=iu/bUXBFSSPwnsuprP6T45Qsbwywu2khUka!6IIleY",
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRES_MINUTES = 180
REFRESH_TOKEN_EXPIRES_MINUTES = 15 * 24 * 60  # 15 days


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "0.0.0.0")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_DB = os.getenv("POSTGRES_DB", "test")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_FROM = os.getenv("MAIL_FROM", "")

OTP_EXPIRE_MINUTES = 10
