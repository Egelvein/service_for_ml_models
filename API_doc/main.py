import sys
import os


import httpx

from fastapi import FastAPI, Request, Form, Depends, Header, status
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from jwt.jwt_c import create_access_token, decode_access_token

from passlib.context import CryptContext

from starlette.middleware.sessions import SessionMiddleware

from typing import Any, Annotated, List, Dict

from datetime import datetime, timedelta

import onnxruntime
import numpy as np
import pickle
from catboost import CatBoostClassifier
from base_model import Token, TokenData, User
from bd_schema import Database
from for_reg import UserLoginForm, UserRegisterForm
from jose import JWTError

# запуск по uvicorn main:app --reload

app = FastAPI()
db = Database()
app.add_middleware(SessionMiddleware, secret_key="mysecretkey", session_cookie="session")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все origins
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)

templates = Jinja2Templates(directory="templates")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
access_token = 0

catboost_model_path = "../models/catboost_model"
logistic_regression_model_path = "../models/logistic_regression_model.pkl"
random_forest_model_path = "../models/random_forest_model.pkl"


with open(logistic_regression_model_path, 'rb') as f:
    logistic_regression_model = pickle.load(f)
with open(random_forest_model_path, 'rb') as f:
    random_forest_model = pickle.load(f)
catboost_model = CatBoostClassifier()      # parameters not required.
catboost_model.load_model(catboost_model_path)


models_list = {
    #"Catboost": catboost_model,
    "Logistic Regression": logistic_regression_model,
    "Random Forest": random_forest_model,
}


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str):
    return pwd_context.hash(password)


def register_user(username: str, password: str):
    user = db.get_user(username)
    if user:
        return False

    hashed_password = get_password_hash(password)
    new_user = {
        "username": username,
        "hashed_password": hashed_password,
    }
    db.insert_new_user(**new_user)

    return True


def get_session(request: Request):
    return request.session


async def get_current_user(authorization: str = Header(...)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization:
        raise credentials_exception

    token = authorization.split(" ")[1]  # Извлекаем токен из заголовка
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = db.get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user


async def get_user_info(access_token: str):
    url = "http://localhost:8000/users/me/"
    headers = {"Authorization" : f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


@app.get("/users/me/", response_model=dict)
async def read_user_info(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/users/me/items/")
async def read_own_items(
    current_user: User = Depends(get_current_active_user)
):
    return [{"item_id": "Foo", "owner": current_user.username}]


@app.get("/login")
async def get_login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    user = db.get_user(username)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Username isn't exist",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # user.password = get_password_hash(password)
    if not pwd_context.verify(password, user.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/register")
async def show_register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register_for_access_token(username: str = Form(...), password: str = Form(...), request: Request = None):
    user = db.get_user(username)
    if user:
        raise HTTPException(
            status_code=422,
            detail="Username already exists",
            headers={"WWW-Authenticate": "Bearer"},
        )

    hashed_password = get_password_hash(password)
    new_user_id = db.insert_new_user(username, hashed_password)

    message = "Регистрация прошла успешно"
    return templates.TemplateResponse(
        "register_result.html", {"message": message, "success": True, "request": request}
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.get("/models")
def read_items():
    return models_list


@app.get("/models/{id}")
def read_item(id: str):
    model = models_list.get(id)
    if model:
        return model
    raise HTTPException(status_code=404, detail=f"Model {id} not found")


@app.get("/model_choice", response_class=HTMLResponse)
def choice_form(request: Request):
    return templates.TemplateResponse("model_choice.html", {"request": request})


@app.post("/make_choice", response_class=HTMLResponse)
def make_choice(request: Request, model: str = Form(...)):
    app.state.selected_model = model
    return templates.TemplateResponse("model_chosen.html", {"request": request, "selected_model": model})


@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    selected_model = app.state.selected_model or "default_model"

    # Ваша асинхронная логика предсказания
    prediction = await run_prediction_async()  # Замените это на реальное асинхронное предсказанное значение

    context = {"request": request, "selected_model": selected_model, "prediction": prediction}

    if selected_model == "logreg":
        result_html = "site_view_logreg.html"
    elif selected_model == "ranfor":
        result_html = "site_view.html"
    else:
        result_html = "default_site_view.html"

    return templates.TemplateResponse(result_html, context)

# Пример асинхронной функции для предсказания
async def run_prediction_async():
    # Ваша логика предсказания
    prediction = 42  # Замените это на реальное предсказанное значение
    return prediction


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            temperature: float = Form(1.0),
            humidity: float = Form(1.0),
            co2cos: float = Form(1.0),
            co2mg: float = Form(1.0),
            co: float = Form(1.0),
            mox: float = Form(1.0)):

    selected_model = app.state.selected_model or "default_model"

    if selected_model == "logreg":
        # Передаем только 2 переменные для модели "logreg"
        prediction = logistic_regression_model.predict([[temperature, humidity]])
    elif selected_model == "ranfor":
        # Передаем все 6 переменных для модели "ranfor"
        prediction = random_forest_model.predict([[temperature,
                                                          humidity,
                                                          co2cos,
                                                          co2mg,
                                                          co,
                                                          mox]])
    else:
        # Другие модели могут иметь свои требования к количеству параметров
        raise HTTPException(status_code=400, detail="Invalid model")

    # Вернем результат в HTML
    return templates.TemplateResponse("result_view.html",
                                      {"request": request,
                                       "selected_model": selected_model,
                                       "prediction_result": prediction})
