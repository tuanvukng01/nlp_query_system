from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from jose import jwt, JWTError
import uvloop
import uvicorn
from ariadne import ObjectType, QueryType, graphql_sync, make_executable_schema
from ariadne.asgi import GraphQL
from inference.inference_pipeline import InferencePipeline
import torch

uvloop.install()
app = FastAPI(title="NLP Query System", version="2.0.0")
pipeline = InferencePipeline()

SECRET_KEY = "CHANGEME123"
ALGORITHM = "HS256"

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

class QueryInput(BaseModel):
    query: str

# Define your GraphQL schema
type_defs = """
    type Query {
        hello: String!
    }
"""

query = QueryType()

@query.field("hello")
def resolve_hello(_, info):
    return "Hello from GraphQL endpoint!"

schema = make_executable_schema(type_defs, query)

app.mount("/graphql", GraphQL(schema, debug=True))

@app.post("/predict")
def predict(item: QueryInput, request: Request):
    # Expecting token in Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth_header.split(" ")[1]
    verify_token(token)

    query = item.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    result = pipeline.run(query)
    return result