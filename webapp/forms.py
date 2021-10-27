from flask_wtf import FlaskForm
from wtforms.fields import StringField, IntegerField, SubmitField, SelectField
from wtforms.validators import DataRequired, InputRequired, Length, Regexp

class VisualizeStopForm(FlaskForm):
    stop_id = StringField('Stop_id: ', validators=[DataRequired(), Length(0,10)])
    stop_time_interval = IntegerField('Time Interval (Min): ', validators=[DataRequired()])
    stop_date_start = StringField('Start Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                  validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    stop_date_end = StringField('End Date: ', render_kw = {'placeholder' : 'YYYY/mm/DD HH:MM'},
                                validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    stop_timing = SelectField(label='Timing: ', choices=[(1, 'Alighting'), (2, 'Boarding')],
                              default=1, coerce=int)
    submit_stop = SubmitField('Visualize')

class VisualizeRouteForm(FlaskForm):
    route_id = StringField('Route_id: ', validators=[DataRequired()])
    route_time_interval = IntegerField('Time Interval (Min): ', validators=[DataRequired()])
    route_date_start = StringField('Start Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                   validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    route_date_end = StringField('End Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                 validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    route_timing = SelectField(label='Timing: ', choices=[(1, 'Alighting'), (2, 'Boarding')],
                               default=1, coerce=int)
    submit_route = SubmitField('Visualize')

class VisualizeStopRouteForm(FlaskForm):
    stop_route_id_stop = StringField('Stop_id: ', validators=[DataRequired()])
    stop_route_id_route = StringField('Route_id: ', validators=[DataRequired()])
    stop_route_time_interval = IntegerField('Time Interval (Min): ',
                                            validators=[DataRequired()])
    stop_route_date_start = StringField('Start Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                        validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    stop_route_date_end = StringField('End Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                      validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    stop_route_timing = SelectField(label='Timing: ', choices=[(1, 'Alighting'), (2, 'Boarding')],
                                    default=1, coerce=int)
    submit_stop_route = SubmitField('Visualize')

class PredictionStopForm(FlaskForm):
    predict_stop_id = StringField('Stop_id: ', validators=[DataRequired()])
    predict_stop_future_steps = IntegerField('Future Steps (15 min): ', validators=[DataRequired()])
    predict_stop_epoch = IntegerField('Epoch: ', validators=[DataRequired()])
    predict_stop_train = IntegerField('Training Size (%): ', validators=[DataRequired()])
    predict_stop_date_start = StringField('Start Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                          validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    predict_stop_date_end = StringField('End Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                        validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    predict_stop_timing = SelectField(label='Timing: ', choices=[(1, 'Alighting'), (2, 'Boarding')],
                                      default=1, coerce=int)
    predict_stop_model = SelectField(label='Model: ', choices=[(1, 'LSTM_MUL'), (2, 'LSTM'), (3, 'RNN'), (4, 'GRU')],
                                     default=1, coerce=int)
    submit_stop_predict = SubmitField('Predict')

class PredictionRouteForm(FlaskForm):
    predict_route_id = StringField('Route_id: ', validators=[DataRequired()])
    predict_route_future_steps = IntegerField('Future Steps (15 min): ', validators=[DataRequired()])
    predict_route_epoch = IntegerField('Epoch: ', validators=[DataRequired()])
    predict_route_train = IntegerField('Training Size (%): ', validators=[DataRequired()])
    predict_route_date_start = StringField('Start Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                           validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    predict_route_date_end = StringField('End Date: ', render_kw={'placeholder' : 'YYYY/mm/DD HH:MM'},
                                         validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    predict_route_timing = SelectField(label='Timing: ', choices=[(1, 'Alighting'), (2, 'Boarding')],
                                       default=1, coerce=int)
    predict_route_model = SelectField(label='Model: ', choices=[(1, 'LSTM_MUL'), (2, 'LSTM'), (3, 'RNN'), (4, 'GRU')],
                                      default=1, coerce=int)
    submit_route_predict = SubmitField('Predict')

class PredictionStopRouteForm(FlaskForm):
    predict_stop_route_id_stop = StringField('Stop_id: ', validators=[DataRequired()])
    predict_stop_route_id_route = StringField('Route_id: ', validators=[DataRequired()])
    predict_stop_route_future_steps = IntegerField('Future Steps (15 min): ', validators=[DataRequired()])
    predict_stop_route_epoch = IntegerField('Epoch: ', validators=[DataRequired()])
    predict_stop_route_train = IntegerField('Training Size (%): ', validators=[DataRequired()])
    predict_stop_route_date_start = StringField('Start Date: ', render_kw={'placeholder': 'YYYY/mm/DD HH:MM'},
                                                validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    predict_stop_route_date_end = StringField('End Date: ', render_kw={'placeholder': 'YYYY/mm/DD HH:MM'},
                                              validators=[DataRequired(), Regexp(r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}', message='Incorrect date format')])
    predict_stop_route_timing = SelectField(label='Timing: ', choices=[(1, 'Alighting'), (2, 'Boarding')],
                                            default=1, coerce=int)
    predict_stop_route_model = SelectField(label='Model: ', choices=[(1, 'LSTM_MUL'), (2, 'LSTM'), (3, 'RNN'), (4, 'GRU')],
                                           default=1, coerce=int)
    submit_stop_route_predict = SubmitField('Predict')

class RetryForm(FlaskForm):
    submit_retry = SubmitField('Retry')
