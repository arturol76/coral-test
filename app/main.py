from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    eid: int
    mid: str = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

#https://fastapi.tiangolo.com/tutorial/body/
@app.post("/items/")
async def create_item(item: Item):
    return item