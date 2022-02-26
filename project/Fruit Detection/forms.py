from flask_wtf import FlaskForm
from wtforms import DecimalField, SubmitField
from wtforms.validators import DataRequired

class MainForm(FlaskForm):
    price = DecimalField("총 가격:", validators=[DataRequired()])
    submit = SubmitField("추가")