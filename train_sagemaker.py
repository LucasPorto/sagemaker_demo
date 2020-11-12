import tensorflow as tf

import argparse
import os
import pickle
import SimpleITK as sitk  # Arbitrary third-party library import


def create_nn(hidden_units):
    input_layer = tf.keras.layers.Input(shape=(2,))
    dense_1 = tf.keras.layers.Dense(units=hidden_units, activation='sigmoid')(input_layer)
    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_1)
    return tf.keras.Model(inputs=input_layer, outputs=output)


def train(X, y, model: tf.keras.Model, epochs, optimizer):
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X, y, epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker arguments
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--inputs_path', type=str, default=os.environ['SM_CHANNEL_INPUTS_PATH'])
    parser.add_argument('--outputs_path', type=str, default=os.environ['SM_CHANNEL_OUTPUTS_PATH'])

    # Optional arguments
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--units', type=int, default=2)
    args = parser.parse_args()

    print(args)
    with open(args.inputs_path, 'rb') as input_file:
        X = pickle.load(input_file)
    with open(args.outputs_path, 'rb') as output_file:
        y = pickle.load(output_file)

    # # Dataset (XOR)
    # X = np.array([
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1]
    # ])
    #
    # y = np.array([
    #     [0],
    #     [1],
    #     [1],
    #     [0]
    # ])

    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate)
    else:
        raise Exception('Argument --optimizer not valid, must be one of "sgd" or "adam".')

    # Model instantiation and training
    model = create_nn(hidden_units=args.units)
    train(X, y, model, epochs=args.epochs, optimizer=optimizer)
