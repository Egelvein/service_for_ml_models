import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists

from bd_creation import Base, Users, Models, Predictions, UsersHistory


class Database:
    def __init__(self):
        engine = create_engine('sqlite:///main_db.db', echo=True)
        self.session = sessionmaker(bind=engine)()

        if database_exists(engine.url):
            Base.metadata.drop_all(bind=engine)

        Base.metadata.create_all(bind=engine)

        self._add_models()

    def _add_models(self):
        lr_model = Models(model_name="Logistic_regression", price=1)
        tree_model = Models(model_name="Random_forest", price=2)
        forest_model = Models(model_name="Catboost", price=4)

        self.session.add(lr_model)
        self.session.add(tree_model)
        self.session.add(forest_model)

        self.session.commit()

    def get_model(self, model_name: str):
        return self.session.query(Models).filter(Models.model_name == model_name).first()

    def insert_new_user(self, username: str, hashed_password: str) -> int:
        user = Users(
            username=username,
            password=hashed_password,
            balance=100
        )
        self.session.add(user)
        self.session.commit()
        return user.user_id

    def get_user(self, username: str) -> Users:
        return self.session.query(Users).filter(Users.username == username).first()

    def update_user_balance(self, user: Users, user_balance: int):
        user.balance = user_balance
        self.session.commit()

    def insert_prediction(self, model_id: int,
                          predict: float):
        prediction = Predictions(
            model_id=model_id,
            predict=predict
        )
        self.session.add(prediction)
        self.session.commit()
        return prediction.prediction_id

    def insert_user_history(self, user_id: int, prediction_id: int) -> int:
        user_history = UsersHistory(
            user_id=user_id,
            prediction_id=prediction_id
        )
        self.session.add(user_history)
        self.session.commit()
        return user_history.user_history_id
