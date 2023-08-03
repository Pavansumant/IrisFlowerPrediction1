from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model using pickle
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input data from the form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Convert input data to a list
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

            # Make predictions
            prediction = model.predict(input_data)[0]

            # Convert prediction to the corresponding species
            species_mapping = {
                0: 'setosa',
                1: 'versicolor',
                2: 'virginica'
            }
            predicted_species = species_mapping[prediction]

            # Render the prediction template with the result
            return render_template('res.html', predicted_species=predicted_species)

        except Exception as e:
            return f'Error: {e}', 500

    # If it's a GET request or there's no form submission yet, render the home template.
    return render_template('home.html')

if __name__ == '__main__':
    app.run()
