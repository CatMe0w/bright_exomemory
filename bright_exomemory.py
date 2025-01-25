import asyncio
import json
import os
import platform
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

DATABASE_CONNINFO = os.getenv("DATABASE_CONNINFO")
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")
AUTOENCODER_MODEL_PATH = os.getenv("AUTOENCODER_MODEL_PATH")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")


class Autoencoder(nn.Module):
    def __init__(self):
        # fmt: off
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1792)
        )
        # fmt: on

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class EmbeddingModel:
    """
    A model class that integrates SentenceTransformer encoding with custom Autoencoder dimensionality reduction
    """

    def __init__(self, st_model_name: str, autoencoder_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # Load SentenceTransformer model
        self.st_model = SentenceTransformer(st_model_name, device=device)

        # Load Autoencoder model
        self.autoencoder = Autoencoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device, weights_only=True))
        self.autoencoder.eval()

    def encode(self, text: str) -> np.ndarray:
        """
        Encode the input text by first using SentenceTransformer, then reducing dimensionality with Autoencoder and quantizing to float16
        """
        # Get the original embedding from SentenceTransformer
        embedding_original = self.st_model.encode(text, convert_to_tensor=True, device=self.device)

        # Dimensionality reduction with Autoencoder
        with torch.no_grad():
            embedding_reduced, _ = self.autoencoder(embedding_original.float())

        # float16 quantization
        embedding_fp16 = embedding_reduced.cpu().numpy().astype(np.float16)

        # If input is a single sentence, return shape (256,), otherwise (N, 256)
        if embedding_fp16.shape[0] == 1:
            return embedding_fp16[0]
        else:
            return embedding_fp16


class SearchResponse(BaseModel):
    date: str
    timestamp: str
    roomName: str
    username: str
    content: str
    files: Optional[List[Dict[str, Any]]] = None
    deleted: bool
    system: bool
    role: Optional[str] = None
    title: Optional[str] = None
    roomId: str
    senderId: str
    time: int


# Dependency injection my ass
pool: Optional[AsyncConnectionPool] = None
model: Optional[EmbeddingModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, model
    try:
        pool = AsyncConnectionPool(
            DATABASE_CONNINFO,
            min_size=1,
            max_size=8,
        )
        await pool.open()

        model = EmbeddingModel(
            st_model_name=TEXT_EMBEDDING_MODEL,
            autoencoder_path=AUTOENCODER_MODEL_PATH,
            device=MODEL_DEVICE,
        )

        yield
    finally:
        if pool:
            await pool.close()


app = FastAPI(lifespan=lifespan)


def parse_files(raw_files: Any) -> List[Dict[str, Any]]:
    """
    Convert JSON string stored in database to Python list; return [] if not a string or failed to convert
    """
    if isinstance(raw_files, str):
        try:
            return json.loads(raw_files)
        except:
            return []
    elif isinstance(raw_files, list):
        return raw_files
    return []


def parse_db_row(row: tuple) -> SearchResponse:
    """
    Parse a row of database results into a SearchResponse object
    """
    date = row[0]
    raw_timestamp = row[1]
    room_name = row[2]
    username = row[3]
    content = row[4]
    files = parse_files(row[5])
    deleted = row[6] if isinstance(row[6], bool) else False
    system = row[7] if isinstance(row[7], bool) else False
    role = row[8]
    title = row[9]
    room_id = str(row[10])
    sender_id = row[11]
    time = row[12]

    return SearchResponse(
        date=date,
        timestamp=raw_timestamp,
        roomName=room_name,
        username=username,
        content=content,
        files=files,
        deleted=deleted,
        system=system,
        role=role,
        title=title,
        roomId=room_id,
        senderId=sender_id,
        time=time,
    )


async def fetch_results(query: str, params: list) -> List[SearchResponse]:
    """
    Common helper function to execute a query and convert results to a list of SearchResponse
    """
    if pool is None:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
        return [parse_db_row(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=List[SearchResponse])
async def search(
    keyword: str = Query(..., description="Search keyword"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Search using reciprocal rank fusion of FTS and semantic similarity
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    embedding = model.encode(keyword)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    query = f"""
    WITH bm25_candidates AS (
        SELECT _id
        FROM "messages"
        WHERE "content" @@@ %s
        ORDER BY paradedb.score(_id) DESC
        LIMIT 100
    ),
    bm25_ranked AS (
        SELECT _id,
               RANK() OVER (ORDER BY paradedb.score(_id) DESC) AS rank
        FROM bm25_candidates
    ),
    semantic_search AS (
        SELECT _id,
               RANK() OVER (ORDER BY embedding <=> %s) AS rank
        FROM "messages"
        ORDER BY embedding <=> %s
        LIMIT 100
    )
    SELECT
        messages.date,
        messages."timestamp",
        rooms."roomName",
        messages.username,
        messages.content,
        messages.files,
        messages.deleted,
        messages.system,
        messages.role,
        messages.title,
        messages."roomId",
        messages."senderId",
        messages.time
    FROM semantic_search
    FULL OUTER JOIN bm25_ranked
        ON semantic_search._id = bm25_ranked._id
    JOIN messages
        ON messages._id = COALESCE(semantic_search._id, bm25_ranked._id)
    JOIN rooms
        ON messages."roomId" = rooms."roomId"::bigint
    ORDER BY COALESCE(1.0 / (60 + semantic_search.rank), 0.0)
           + COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0) DESC, content
    LIMIT {limit} OFFSET {offset};
    """

    return await fetch_results(query, [keyword, embedding_str, embedding_str])


@app.get("/search_fts_only", response_model=List[SearchResponse])
async def search_fts_only(
    keyword: str = Query(..., description="Search keyword"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Search using FTS only
    """
    query = f"""
    SELECT
        messages.date,
        messages."timestamp",
        rooms."roomName",
        messages.username,
        messages.content,
        messages.files,
        messages.deleted,
        messages.system,
        messages.role,
        messages.title,
        messages."roomId",
        messages."senderId",
        messages.time
    FROM "messages"
    JOIN rooms ON messages."roomId" = rooms."roomId"::bigint
    WHERE "content" @@@ %s
    ORDER BY messages.time DESC
    LIMIT {limit} OFFSET {offset};
    """

    return await fetch_results(query, [keyword])


@app.get("/search_semantic_only", response_model=List[SearchResponse])
async def search_semantic_only(
    keyword: str = Query(..., description="Search keyword"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Search using semantic similarity only
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    embedding = model.encode(keyword)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    query = f"""
    SELECT
        messages.date,
        messages."timestamp",
        rooms."roomName",
        messages.username,
        messages.content,
        messages.files,
        messages.deleted,
        messages.system,
        messages.role,
        messages.title,
        messages."roomId",
        messages."senderId",
        messages.time
    FROM "messages"
    JOIN rooms ON messages."roomId" = rooms."roomId"::bigint
    ORDER BY embedding <=> %s
    LIMIT {limit} OFFSET {offset};
    """

    return await fetch_results(query, [embedding_str])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
