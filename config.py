from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel
load_dotenv()


class Config(BaseModel):
    DATABASE_PATH: str


config = Config(
    DATABASE_PATH=getenv("DATABASE_PATH")
)