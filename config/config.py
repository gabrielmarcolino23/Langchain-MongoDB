import os
from dotenv import load_dotenv
from langchain_community.document_loaders.mongodb import MongodbLoader

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')


def load_docs():
    loader = MongodbLoader(
        connection_string=MONGODB_URI,
        db_name=MONGODB_DB,
        collection_name=MONGODB_COLLECTION,
        field_names=["title",'imdb'],
        
    )

    docs = loader.load()
    return docs

docs=load_docs()

print(len(docs))
print(docs[20])
