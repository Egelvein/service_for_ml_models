from pydantic import BaseModel

class UserLoginForm(BaseModel):
    username: str
    password: str

class UserRegisterForm(BaseModel):
    username: str
    password: str