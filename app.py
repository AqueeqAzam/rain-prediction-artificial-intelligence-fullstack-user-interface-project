from flask import Flask, request, render_template
import pickle
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the model and scaler for prediction
with open('rain_prediction_model.pkl', 'rb') as file:
    saved_objects = pickle.load(file)
    model = saved_objects['model']
    scaler = saved_objects['scaler']

app = Flask(__name__)

def expert_system_rules(ml_prediction, user_input):
    temperature, humidity, wind_speed, precipitation, atmospheric_pressure = user_input[0]

    if temperature > 25 and humidity < 60:
        return "No Rain"
    if wind_speed > 20 and precipitation < 5:
        return "No Rain"
    if atmospheric_pressure < 1000:
        return "Rain"
    if precipitation > 10 and humidity > 80:
        return "Rain"
    if temperature < 15 and wind_speed < 10:
        return "No Rain"

    return "Rain" if ml_prediction > 0.5 else "No Rain"

def extract_number(text):
    doc = nlp(text)
    for token in doc:
        if token.like_num:
            return float(token.text)
    raise ValueError("No numerical value found in the input.")

def predict_rain(user_input):
    user_input_scaled = scaler.transform(user_input)
    ml_prediction = model.predict(user_input_scaled)[0]
    return expert_system_rules(ml_prediction, user_input)

def provide_recommendations(user_input, prediction):
    temperature, humidity, wind_speed, precipitation, atmospheric_pressure = user_input[0]
    recommendations = []

    if prediction == "Rain":
        recommendations.append("Carry an umbrella.")
        recommendations.append("Wear waterproof clothing.")
        recommendations.append("Avoid outdoor activities.")
    else:
        recommendations.append("It's a good day for outdoor activities.")
        recommendations.append("Wear light and comfortable clothing.")
        recommendations.append("Stay hydrated.")

    if temperature > 30:
        recommendations.append("It's very hot outside. Stay cool and hydrated.")
    elif temperature < 10:
        recommendations.append("It's quite cold. Dress warmly.")

    if humidity > 80:
        recommendations.append("High humidity levels. It might feel hotter than the actual temperature.")
    elif humidity < 30:
        recommendations.append("Low humidity levels. Stay moisturized and hydrated.")

    if wind_speed > 20:
        recommendations.append("High wind speeds. Be cautious of strong gusts and secure loose objects.")
    elif wind_speed < 5:
        recommendations.append("Calm winds. Ideal conditions for outdoor activities.")

    if atmospheric_pressure < 1000:
        recommendations.append("Low atmospheric pressure. It could indicate stormy weather.")
    elif atmospheric_pressure > 1020:
        recommendations.append("High atmospheric pressure. Expect clear skies and stable weather.")

    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        temperature = extract_number(request.form['temperature'])
        humidity = extract_number(request.form['humidity'])
        wind_speed = extract_number(request.form['wind_speed'])
        precipitation = extract_number(request.form['precipitation'])
        atmospheric_pressure = extract_number(request.form['atmospheric_pressure'])

        user_input = np.array([[temperature, humidity, wind_speed, precipitation, atmospheric_pressure]])
        prediction = predict_rain(user_input)
        recommendations = provide_recommendations(user_input, prediction)

        return render_template('index.html', prediction=prediction, recommendations=recommendations)
    
    return render_template('index.html', prediction=None, recommendations=[])

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)
    
