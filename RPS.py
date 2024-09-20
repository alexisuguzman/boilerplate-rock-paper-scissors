import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Global variables for model training
opponent_history = []
player_history = []
model = None

def collect_data(opponent_history, player_history, N=5, H =30):
    opponent_history = opponent_history[-H:]
    player_history = player_history[-H:]

    data = []
    labels = []

    if len(opponent_history) > N and len(player_history) > N:
        for i in range(len(opponent_history) - N):
            # Ensure both histories have enough data for N
            if i + N <= len(player_history):
                # Select training pair
                input_sequence = opponent_history[i:i + N] + player_history[i:i + N]

                # Convert R-P-S to 0-1-2
                input_sequence = [convert_move(m) for m in input_sequence]

                # The label is the next move of the opponent
                next_move = convert_move(opponent_history[i + N])

                data.append(input_sequence)
                labels.append(next_move)

    if len(data) > 0:
        return np.array(data), np.array(labels)
    else:
        return np.empty((0, 2 * N)), np.empty((0,))  # Return empty arrays if no data

def convert_move(move):
    if move == "R":
        return 0
    elif move == "P":
        return 1
    elif move == "S":
        return 2

def reverse_convert_move(number):
    if number == 0:
        return "R"
    elif number == 1:
        return "P"
    elif number == 2:
        return "S"

def create_model(input_size):
    model = tf.keras.Sequential([
        layers.InputLayer(shape=(input_size,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(3, activation="softmax")  # 3 outputs for R, P, S
    ])

    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy", 
                  metrics=['accuracy'])

    return model

def player(prev_play, opponent_history=opponent_history, player_history=player_history):
    global model
    
    N = 10  # Sequence length
    H = 20  # Move history lenght

    if prev_play != "":
        opponent_history.append(prev_play)
    
    # If not enough data to make predictions, generate and return a random move
    if len(opponent_history) <= N:
        random_move = np.random.choice(["R", "P", "S"])
        player_history.append(random_move)  # Append the random move to player history
        print("Not enough data, returning random move")
        return random_move

    # Collect data
    data, labels = collect_data(opponent_history, player_history, N, H)
    
    # If model doesn't exist and there's enough data, create and train it
    if model is None and len(data) > 0:
        print("Creating and training the model")
        model = create_model(data.shape[1])

    # Only fit the model if there is enough data
    if model is not None and len(data) > 0:
        print("Training the model with new data")
        model.fit(data, labels, epochs=10, verbose=0)

    # If the model exists, make a prediction
    if model is not None:
        # Prepare input sequence for prediction
        input_sequence = opponent_history[-N:] + player_history[-N:]
        input_sequence = [convert_move(m) for m in input_sequence]
        input_sequence = np.array([input_sequence])

        # Predict the opponent's next move
        prediction = model.predict(input_sequence)
        predicted_move = np.argmax(prediction)

        # Choose the winning move against the predicted opponent move
        next_move = (predicted_move + 1) % 3  # 0 -> R, 1 -> P, 2 -> S
        player_move = reverse_convert_move(next_move)

        # Append the player's move to history
        player_history.append(player_move)

        return player_move

    # Default to a random move if no model is available or prediction is not possible
    random_move = np.random.choice(["R", "P", "S"])
    player_history.append(random_move)  # Append the random move to player history
    # print("Making random Prediction")
    return random_move