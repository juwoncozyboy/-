from typing import Union

from fastapi import FastAPI

from src.utils.util import cleaning, lemmatize, spacingWord

app = FastAPI()


@app.get("/")
def read_root(q: Union[str, None] = None):
    word = cleaning(q)
    result = spacingWord(word)
    return lemmatize(result)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
