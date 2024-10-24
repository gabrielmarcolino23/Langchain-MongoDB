import os
from dotenv import load_dotenv
from langchain_community.document_loaders.mongodb import MongodbLoader

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB', 'vector_db')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'documents')

loader = MongodbLoader(
    connection_string="mongodb+srv://mostockpinho:88583380lf@clusterluiz.c5aej.mongodb.net/",
    db_name="sample_mflix",
    collection_name="movies",
)

docs = loader.load()