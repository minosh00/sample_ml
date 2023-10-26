from flask import Flask, request, jsonify
import joblib
import pandas as pd
import random

app = Flask(__name__)

# Load the pre-trained model
model_gbc = joblib.load('demand.pkl')

# Define ranges for each class label
class_ranges = {
    "rain": (2000, 3000),     # Assign range (0, 10) for class 0
    "sun": (3000, 4000),    # Assign range (11, 20) for class 1
    "fog": (5000, 6000)     # Assign range (21, 30) for class 2
}

# Dictionary to store random values for different inputs
random_values = {}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json
        input_hash = hash(str(data))  # Create a unique hash for the input data

        # Check if random value is already generated for this input
        if input_hash in random_values:
            random_prediction = random_values[input_hash]
        else:
            # Make predictions
            test_data = pd.DataFrame(data)
            predictions = model_gbc.predict(test_data)

            # Generate a random value within the assigned range
            random_prediction = random.uniform(class_ranges[predictions[0]][0], class_ranges[predictions[0]][1])

            # Round the random prediction to the nearest multiple of 10 without decimals
            random_prediction = int(round(random_prediction, -1))

            # Store the random value for this input
            random_values[input_hash] = random_prediction

        # Return the random prediction as a JSON response
        return jsonify({'prediction': random_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
