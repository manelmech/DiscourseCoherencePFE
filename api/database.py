from click import echo
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# à mettre dans un fichier à part
DATABASE_URL = "postgresql://postgres:manel@localhost:5432/mydatabase"
engine = create_engine(DATABASE_URL,echo= True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()