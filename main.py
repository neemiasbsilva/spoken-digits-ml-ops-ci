from model import CNN_Architecture
from utils import load_dataset, split_train_test, save_json, save_file
import os
import tensorflow as tf
import argparse
import numpy as np

tf.random.set_seed(7)


def ensure_path(directory: str):
    if (not os.path.exists(directory)):
        os.makedirs(directory)
        print("Directory has bee created!")
    else:
        print("Directory has already created!")


def main(args):    
    checkpoint_path = args.checkpoint_path
    data_path = args.data_path
    json_path = args.json_path
    n_classes = args.n_classes
    filters = args.filters
    kernel_size = args.kernel_size
    batch_size = args.batch_size
    epochs = args.epochs

    print(n_classes)
    filters = [int(i) for i in filters]

    kernel_size_temp = [list(i.split(',')) for i in kernel_size]
    kernel_size = []

    for i in kernel_size_temp:
        kernel_size.append((int(i[0]), int(i[1])))
    
    data, target = load_dataset(data_path)
    data = np.asarray(data)
    target = np.asarray(target)

    train_dataset, test_dataset = split_train_test(data, target)
    X_train, y_train = train_dataset
    X_test, y_test = test_dataset

    input_shape = X_train.shape[1:]

    cnn_architecture = CNN_Architecture(input_shape, n_classes)

    model = cnn_architecture.get_model(filters, kernel_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    ensure_path(checkpoint_path)

    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, "best_model.h5"),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )


    callbacks = [mc]
    
    history = model.fit(X_train, y_train, batch_size=batch_size, 
                        epochs=epochs, verbose=True, callbacks=callbacks,
                        validation_split=.2, shuffle=True)    

    loss, acc = model.evaluate(X_test, y_test)
    
    result = {
        "loss": loss,
        "accuracy": acc
    }
    
#     save_json(json_path, result)
    save_file(json_path, result)
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get through run.sh file")

    parser.add_argument("--checkpoint_path", action="store", required=True,
                        help="The checkpoint for saved the model", dest="checkpoint_path")

    parser.add_argument("--data_path", action="store", required=True,
                        help="The data or url", dest="data_path")

    parser.add_argument("--json_path", action="store", required=True,
                        help="The json for save metrics", dest="json_path")

    parser.add_argument("--n_classes", action="store", type=int, dest="n_classes")
    
    parser.add_argument("--filters", action="extend", nargs="+", type=int, dest="filters")
    
    parser.add_argument("--kernel_size", action="extend", nargs="+", type=str, dest="kernel_size")

    parser.add_argument("--batch_size", action="store", type=int, dest="batch_size")
    
    parser.add_argument("--epochs", action="store", type=int, dest="epochs")


    args = parser.parse_args()

    main(args)
    
