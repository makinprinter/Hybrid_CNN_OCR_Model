import os
import json
import pickle
import winsound
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from CNN import Modded_CNN
import FullReportMaker as FR
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Sort_o_and_I import Is_O_or_I_Later_Sort, Is_O_or_I
from sklearn.model_selection import train_test_split
from Opener import getData


def build_attribute_dictionary():
    attr_dict = {"ALL": {'X': [], 'y': [], "Letters": []}}

    for i, image in tqdm(enumerate(train_x)):
        o_val, i_val, hl_val = Is_O_or_I_Later_Sort(image, [17, 4, 17, 4], 0.9)
        group_key = f"O_{o_val}_I_{i_val}_Hl_{hl_val}"
        
        if group_key not in attr_dict:
            attr_dict[group_key] = {'X': [], 'y': [], "Letters": []}

        label = np.argmax(train_y[i])
        if label not in attr_dict[group_key]['Letters']:
            attr_dict[group_key]['Letters'].append(label)

        attr_dict["ALL"]['X'].append(image)
        attr_dict["ALL"]['y'].append(train_y[i])

    for i, image in tqdm(enumerate(train_x)):
        label = np.argmax(train_y[i])
        for group_key in attr_dict:
            if group_key != "ALL" and label in attr_dict[group_key]['Letters']:
                attr_dict[group_key]['X'].append(image)
                attr_dict[group_key]['y'].append(train_y[i])

    keys_to_remove = [
        k for k in attr_dict if k != "ALL" and len(attr_dict[k]['X']) == len(attr_dict["ALL"]['X'])
    ]
    for k in keys_to_remove:
        attr_dict.pop(k)

    return attr_dict


def remap_label_indices(y_train, y_test):
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    y_train_mapped = np.array([label_map[lbl] for lbl in y_train])
    y_test_mapped = np.array([label_map[lbl] for lbl in y_test])

    return y_train_mapped, y_test_mapped, label_map


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def train_models_by_attribute(attribute_data, should_train):
    model_paths = {}
    os.makedirs("models", exist_ok=True)
    validation_data = {}

    for group_key, data in tqdm(attribute_data.items()):
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(data['X']), np.array(data['y']), test_size=0.3, random_state=42)

        y_train_int = np.argmax(y_train, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
        num_classes = np.unique(y_train_int).size

        y_train_int, y_test_int, label_map = remap_label_indices(y_train_int, y_test_int)

        if group_key not in validation_data:
            validation_data[group_key] = {'X': [], 'y': [], "map": label_map}
        validation_data[group_key]['X'].append(X_test)
        validation_data[group_key]['y'].append(y_test_int)

        if should_train:
            model_path = f"models/{group_key}.keras"
            Modded_CNN(X_train, X_test, y_train_int, y_test_int, num_classes, 2, "models", group_key)

        model_paths[group_key] = f"models/{group_key}.keras"

    save_pickle(model_paths, "models/model_index.pkl")
    return validation_data


def predict_with_full_model(image):
    model_paths = load_pickle("models/model_index.pkl")
    o_val, i_val, hl_val = Is_O_or_I(image)
    key = f"O_{o_val}_I_{i_val}_Hl_{hl_val}"

    if key in model_paths:
        model = load_model(model_paths[key])
        image = image.reshape(1, 32, 32, 1)
        prediction = model.predict(image)
        return np.argmax(prediction)
    else:
        return "No relevant model found."


def generate_individual_reports(validation_data):
    report = "System Performance Report\n\n"
    model_paths = load_pickle("models/model_index.pkl")

    for key, path in model_paths.items():
        model = load_model(path)
        X_test = validation_data[key]["X"]
        y_test = validation_data[key]["y"]
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        report += f"Model: {key}\nAccuracy: {accuracy * 100:.2f}%\n\n"

    with open("models/system_reportTry4.txt", "w") as f:
        f.write(report)
    print("Report saved as system_reportTry4.txt")


def generate_combined_report(sort_params, attr_dict, test_x, test_y, validation_data):
    total_correct = 0
    total_samples = 18799  # Update if dynamic size is preferred

    sorted_data = FR.SortData(test_x, test_y, sort_params, attr_dict)
    model_paths = load_pickle("models/model_index.pkl")

    for key, data in sorted_data.items():
        model = load_model(model_paths[key])
        label_map = validation_data[key]["map"]

        for idx, img in enumerate(data["data"]):
            reshaped_img = img.reshape(1, 28, 28, 1)
            pred = model.predict(reshaped_img)
            true_label_idx = np.argmax(test_y[data["indexs"][idx]])

            if key == "ALL":
                if np.argmax(pred) == label_map[true_label_idx]:
                    total_correct += 1
            else:
                if np.argmax(pred) == true_label_idx:
                    total_correct += 1
                    break  # prevent double counting

    accuracy = (total_correct / total_samples) * 100
    print(f"{total_correct} Total Correct Predictions")
    print(f"{accuracy:.2f}% Overall Accuracy")

    report = f"Total Correct: {total_correct}\nAccuracy: {accuracy:.2f}%\n"
    with open("models/Full_System_Report.txt", "w") as f:
        f.write(report)


# === Main Execution ===

train_x, train_y, test_x, test_y, full_x, full_y = getData()
attribute_dict_path = 'models/TestFasterAttributeDict7.pkl'

attribute_dict = load_pickle(attribute_dict_path)
print(attribute_dict.keys(), "Loaded Attribute Keys")

should_train_models = False
validation_results = train_models_by_attribute(attribute_dict, should_train_models)

grouping_parameters = [[17, 4, 17, 4], 0.9]
generate_combined_report(grouping_parameters, attribute_dict, test_x, test_y, validation_results)
