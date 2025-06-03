from pathlib import Path
import numpy as np
import random
import pandas as pd
import cv2
import streamlit as st
from hough import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy



top = st.container()


@st.cache_data
def create_dataset(dataset_path: str):
    #load all csv
    dataset = Path(dataset_path)
    # iterate all csv files
    csv_files = list(dataset.glob('*.csv'))
    
    images = []
    labels = []
    for csv_file in csv_files:
        subdir_name = csv_file.stem
        print(f"Loading {subdir_name}...")
        
        # check if directory exists
        assert dataset.joinpath(subdir_name).exists(), f"Directory {subdir_name} does not exist"

        csv = pd.read_csv(csv_file)

        for id, row in enumerate(csv.itertuples()):
            # if id % 20 != 0:
            #     continue

            image_name = str(row[1]).zfill(4)
            forward_signal = row[2]
            left_signal = row[3]

            # load image
            image_path = dataset.joinpath(subdir_name, image_name + ".jpg")
            assert image_path.exists(), f"Image {image_path} does not exist"
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

            
            images.append(img)
            labels.append(np.array([forward_signal, left_signal]))
            print(f"Images: {len(images)}, Labels: {len(labels)}", end='\r')

    assert len(images) == len(labels)
    dataset = zip(images, labels)

    return dataset

def get_random_images(dataset, n=16):
    return random.choices(list(dataset), k=16)

def get_lines_p_features2(lines):
    if lines is None:
        return None

    clusterer = KMeans(n_clusters=3, random_state=42)
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    angle = (np.arctan2(y2 - y1, x2 - x1) + np.pi) / (2 * np.pi)  # Normalize angle to [0, 1]
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2

    data = np.column_stack((angle, midpoint_x, midpoint_y))
    clusterer.fit(data)
    labels = clusterer.labels_

    # average_angle of each cluster
    average_angle = np.array([np.mean(data[labels == i, 0]) for i in range(clusterer.n_clusters)])
    average_midpoint_x = np.array([np.mean(data[labels == i, 1]) for i in range(clusterer.n_clusters)])
    average_midpoint_y = np.array([np.mean(data[labels == i, 2]) for i in range(clusterer.n_clusters)])

    ordered_indices = np.argsort(average_angle)
    ordered_average_angle = average_angle[ordered_indices]
    ordered_average_midpoint_x = average_midpoint_x[ordered_indices]
    ordered_average_midpoint_y = average_midpoint_y[ordered_indices]
    return np.concatenate((ordered_average_angle, ordered_average_midpoint_x, ordered_average_midpoint_y))





def get_lines_p_features(lines):
    if lines is None:
        return None

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    angle = np.arctan2(y2 - y1, x2 - x1)

    # length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
    average_angle = np.mean(angle)
    std_angle = np.std(angle)
    average_length = np.mean(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    std_length = np.std(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    
    return np.array([average_angle, std_angle, average_length, std_length])

@st.cache_resource
def get_regressor(X_train, y_train, X_test, y_test):
    regressors = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'SVR': MultiOutputRegressor(SVR()),
    'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror'),
    'LinearRegression': LinearRegression(),
    'KNN': KNeighborsRegressor()
}
    top = st.container()
# Define hyperparameter grids for each regressor
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        'SVR': {
            'estimator__kernel': ['rbf', 'linear'],
            'estimator__C': [0.1, 1, 10],
            'estimator__epsilon': [0.01, 0.1, 0.5]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        },
        'LinearRegression': {},  # No hyperparameters to tune
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Manhattan (p=1) or Euclidean (p=2) distance
        }
    } 
    results = {}
    for name, regressor in regressors.items():
        top.write(f"Training {name} Regressor...")
        grid_search = GridSearchCV(
            estimator=regressor,
            param_grid=param_grids[name],
            cv=5,  # 5-fold cross-validation
            scoring='neg_mean_squared_error',  # Optimize for MSE
            n_jobs=-1,  # Use all available CPU cores
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Store best model and score
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  # Convert to positive MSE
        results[name] = {
            'best_model': best_model,
            'best_score': best_score,
            'best_params': grid_search.best_params_
        }
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        results[name]['test_mse'] = test_mse
        
        top.write(f"{name} - Best CV MSE: {best_score:.4f}, Test MSE: {test_mse:.4f}, Best Params: {grid_search.best_params_}")

        # print 5 best models
        for i, (model_name, model_info) in enumerate(results.items()):
            top.write(f"{i+1}. {model_name}: Best MSE: {model_info['best_score']:.4f}, Test MSE: {model_info['test_mse']:.4f}, Best Params: {model_info['best_params']}")

        

        return best_model, top
class HoughTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
          # best 104, 147,255,1,1,1,36
        self.crop_top=crop_top
        self.threshold1 = 147
        self.threshold2 = 255
        self.blur_kernel_size = 1
        self.rho = 1
        self.theta = 1 * np.pi / 180
        self.threshold = 16
        self.min_line_length = 36
        self.max_line_gap = 10


    def fit(self, X, y=None):
        ...
    
    def transform(self, X):
        return hough_transform_p(X, self.crop_top, self.threshold1, self.threshold2, 3, self.blur_kernel_size, self.rho, self.theta, self.threshold, self.min_line_length, self.max_line_gap)

class FeatureExtraction(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting required for feature extraction
        return self
    
    def transform(self, X):

        return get_lines_p_features(X)
        


if __name__ == "__main__":
    dataset = list(create_dataset("dataset"))

    random.seed(0xc0ffee)

        # lines = hough_transformrow1 = st.rowumns(3)
    images = st.session_state.get('images', get_random_images(dataset))
    dummy = deepcopy(images[0])

    col1, col2 = st.columns([0.8, 0.2])

    hough_type = top.radio("Hough Transform Type", options=["Standard Hough Transform", "Probabilistic Hough Transform"], index=1)


    with col2:
        if st.button("Randomize Images"):
            st.session_state['images'] = get_random_images(dataset)

        canny_threshold1 = st.slider("Canny Threshold 1", min_value=0, max_value=255, value=147, step=1)
        canny_threshold2 = st.slider("Canny Threshold 2", min_value=0, max_value=300, value=255, step=1)

        blur_kernel_size = st.slider("Blur Kernel Size", min_value=1, max_value=21, value=1, step=2)
        crop_top = st.slider("Crop Top", min_value=0, max_value=166, value=104, step=1)

        if hough_type == "Probabilistic Hough Transform":
            # best 104, 147,255,1,1,1,50, 36, 10
            rho = st.slider("Rho", min_value=1, max_value=10, value=1, step=1) 
            theta = st.slider("Theta", min_value=1, max_value=180, value=1, step=1)
            threshold = st.slider("Threshold", min_value=1, max_value=100, value=16, step=1)
            min_line_length = st.slider("Min Line Length", min_value=1, max_value=100, value=36, step=1)
            max_line_gap = st.slider("Max Line Gap", min_value=1, max_value=100, value=10, step=1)


        else:
            rho = st.slider("Rho", min_value=1, max_value=10, value=1, step=1)
            theta = st.slider("Theta", min_value=1, max_value=180, value=1, step=1)
            threshold = st.slider("Threshold", min_value=1, max_value=100, value=50, step=1)


                                            

    get_lines = lambda img: hough_transform_p(img, crop_top, canny_threshold1, canny_threshold2, 3, blur_kernel_size, rho, theta * np.pi / 180, threshold, min_line_length, max_line_gap)

    feature_dataset = []
    none_counter = 0
    for img, label in dataset:
        lines = get_lines(img)
        if lines is None:
            none_counter += 1
            continue

        features = get_lines_p_features(lines)
        feature_dataset.append((features, label))

    training_data, testing_data = train_test_split(feature_dataset, test_size=0.2, random_state=42)
    X_train, y_train = zip(*training_data)
    X_test, y_test = zip(*testing_data)
    del training_data, testing_data

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    top.write(f"X_train shape: {X_train[0].shape}, y_train shape:")
    top.write(f"Found {len(feature_dataset)} images with lines, {none_counter} images without lines.")

    regressor, resbox = get_regressor(X_train, y_train, X_test, y_test)

    pipeline = Pipeline([
        ('hough_transform', HoughTransform()),
        ('feature_extraction', FeatureExtraction()),
        ('scaler', scaler),
        ('regressor', regressor)
        ])

    top.write(f"Pipeline output: {out}, real: ")
    top.write(X_train[0])
    top.write(f"scrumble output: {regressor.predict(X_train[0][np.newaxis,:])}, real: {y_train[1]}")

    # Save the pipeline to a file
    import joblib
    joblib.dump(pipeline, 'line_regressor_pipeline.pkl')
    top.write("Pipeline dumped to 'line_regressor_pipeline.pkl'.")

    del X_train, y_train, X_test, y_test
    del feature_dataset

    # feature_dataset = []

    # for img, label in dataset:
    #     lines = get_lines(img)
    #     if lines is None:
    #         continue

    #     features = get_lines_p_features2(lines)
    #     if features is not None:
    #         feature_dataset.append((features, label))

    # training_data, testing_data = train_test_split(feature_dataset, test_size=0.2, random_state=42)
    # X_train, y_train = zip(*training_data)
    # X_test, y_test = zip(*testing_data)
    # del training_data, testing_data
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # top.write("Clustere features extracted.")
    # top.write(f"Found {len(feature_dataset)} images with lines, {none_counter} images without lines.")
    # regressor2, resbox = get_regressor(X_train, y_train, X_test, y_test)
    # del X_train, y_train, X_test, y_test

















    # Hyperparameter tuning

    
    with col1:
        for (img, _) in images:
            row = st.columns(3)
            row[0].image(img,)


            canny = cv2.Canny(cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0), canny_threshold1, canny_threshold2, apertureSize=3)
            canny[:crop_top, :] = 0  # Crop the top part of the image
            row[1].image(canny)

            if hough_type == "Probabilistic Hough Transform":
                # best 104, 147,255,1,1,1,36
                hough_lines = hough_transform_p(img, crop_top, canny_threshold1, canny_threshold2, 3, blur_kernel_size, rho, theta * np.pi / 180, threshold, min_line_length, max_line_gap)
                hough_img = draw_hough_lines_p(img.copy(), hough_lines)
                row[2].image(hough_img)
                # best: 100, 147,255,1,1,2,34
            # best: 100, 147,255,1,1,2,34
            else:
                hough_lines = hough_transform(img, crop_top, canny_threshold1, canny_threshold2, 3, blur_kernel_size, rho, theta * np.pi / 180, threshold)
                hough_img = draw_hough_lines(img.copy(), hough_lines)
                row[2].image(hough_img)













