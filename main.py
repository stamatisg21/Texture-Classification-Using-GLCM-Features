import os, csv
from PIL import Image
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_and_process_images(input_folder, output_folder, target_size=(256, 256)):
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct full file path
                file_path = os.path.join(subdir, file)
                img = Image.open(file_path)
                img = img.resize(target_size)  # Resize image

                # Normalize image
                img_array = np.array(img) / 255.0

                # Prepare output path, mirroring input structure
                rel_path = os.path.relpath(subdir, input_folder)  # Relative path to maintain subfolder structure
                save_folder = os.path.join(output_folder, rel_path)
                create_dir(save_folder)  # Ensure the target directory exists

                # Save the processed image
                save_path = os.path.join(save_folder, file)
                # Convert array back to image
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                img.save(save_path)


# # # Define input and output folders
# input_folder = '/home/stamatis/project1/im/images'
# output_folder = '/home/stamatis/project1/out2'

# # # # Process and save images
# load_and_process_images(input_folder, output_folder)

def calculate_average_histogram(folder):
    histogram_sum = None
    num_images = 0

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
                hist = hist / hist.sum()  # Normalize histogram

                if histogram_sum is None:
                    histogram_sum = hist
                else:
                    histogram_sum += hist

                num_images += 1

    average_histogram = histogram_sum / num_images
    return average_histogram

def match_histogram(image, reference_histogram):
    old_shape = image.shape
    image = image.ravel()

    s_values, bin_indices, s_counts = np.unique(image, return_inverse=True, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    t_values = np.interp(s_quantiles, reference_histogram, np.arange(256))

    return t_values[bin_indices].reshape(old_shape).astype('uint8')

def apply_histogram_matching(input_folder, output_folder, reference_histogram):
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_img_path = os.path.join(subdir, file)
                img = Image.open(input_img_path).convert('L')

                img_array = np.array(img)
                matched_img = match_histogram(img_array, reference_histogram)
                matched_img = Image.fromarray(matched_img)

                output_subdir = subdir.replace(input_folder, output_folder)
                create_dir(output_subdir)  # Ensure the target directory exists

                output_img_path = os.path.join(output_subdir, file)
                matched_img.save(output_img_path)

# Specify your input and output folders
input_folder = '/home/stamatis/project1/out2'
output_folder = '/home/stamatis/project1/hist_matched_images_final'
create_dir(output_folder)

# Calculate the average histogram and apply histogram matching
average_hist = calculate_average_histogram(input_folder)
apply_histogram_matching(input_folder, output_folder, average_hist)


def calculate_entropy(glcm):
    # Sum over all intensity levels and angles to get a total probability distribution for each distance
    probabilities = glcm.sum(axis=(0, 1)) + 1e-12  # Adding a small constant to avoid division by zero
    probabilities /= probabilities.sum()  # Normalize the probabilities
    log_probabilities = np.log2(probabilities)
    entropy = -np.sum(probabilities * log_probabilities)
    return entropy

def extract_glcm_features(image, distances, angles, properties):
    # Ensure the image is in grayscale and convert to uint8
    if image.mode != 'L':
        image = image.convert('L')
    image_array = np.array(image, dtype=np.uint8)

    # Calculate the GLCM matrix
    glcm = graycomatrix(image_array, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Extract properties from the GLCM
    features = {}
    for prop in properties:
        features[prop] = [graycoprops(glcm, prop)[0, i] for i in range(len(angles))]

    # Calculate and store entropy
    entropy_values = [calculate_entropy(glcm[:, :, d, :]) for d in range(glcm.shape[2])]
    features['entropy'] = np.mean(entropy_values)  # Average entropy over all distances

    return features

def process_folder_and_save_features(input_folder, output_folder, distances, angles, properties):
    create_dir(output_folder)
    output_file_path = os.path.join(output_folder, 'glcm_features.csv')

    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['Filename']
        for prop in properties + ['entropy']:
            for angle in angles:
                for d in distances:
                    header.append(f"{prop}_angle_{np.degrees(angle):.0f}_distance_{d}")
        writer.writerow(header)
        
        for subdir, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, file)
                    img = Image.open(img_path)
                    features = extract_glcm_features(img, distances, angles, properties)
                    
                    row = [file]
                    for prop in properties + ['entropy']:
                        # Extend row with each property's features, ensuring they are iterable
                        feature_values = features[prop]
                        if isinstance(feature_values, np.float64):  # If the feature is a single scalar, make it a list
                            feature_values = [feature_values]
                        row.extend(feature_values)
                    writer.writerow(row)

# Setup parameters
distances = [1, 2, 3]  # Distances greater than 1
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 degrees
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']


# Specify the folder containing the preprocessed images
input_folder = '/home/stamatis/project1/out2'
output_folder = '/home/stamatis/project1/features_final'

#Process the folder and extract features
# process_folder_and_save_features(input_folder, output_folder, distances, angles, properties)

# csv_file_path = '/home/stamatis/project1/merged_output.csv'
# features_df = pd.read_csv(csv_file_path)

# if 'Filename' in features_df.columns:
#     features_df.drop('Filename', axis=1, inplace=True)


# # Check for NaN values in the DataFrame
# print(features_df.isna().sum())  # This will show the count of NaNs in each column

# # Alternatively, you can check if there are any NaN values at all
# if features_df.isna().any().any():
#     print("There are NaN values in the dataset.")
# else:
#     print("There are no NaN values in the dataset.")

# cleaned_features_df = features_df.dropna(axis=1, how='all')  # Drop columns with all NaN values

# imputer = SimpleImputer(strategy='mean')  # or median, most_frequent
# data_filled = pd.DataFrame(imputer.fit_transform(cleaned_features_df), columns=cleaned_features_df.columns)

# if data_filled.isna().any().any():
#     print("There are still NaN values in the dataset.")
# else:
#     print("No NaN values remain in the dataset.")

# pca = PCA(n_components=0.95)
# reduced_features = pca.fit_transform(cleaned_features_df)
# print("Reduced feature dimensions:", reduced_features.shape)
# scaler = StandardScaler()
# normalized_features = scaler.fit_transform(reduced_features if 'pca' in locals() else features_df)
# processed_df = pd.DataFrame(normalized_features)
# processed_df.to_csv('processed_features.csv', index=False)

# reloaded_df = pd.read_csv('processed_features.csv')
# print(reloaded_df.head())  # Check the first few rows


csv_file_path = '/home/stamatis/project1/merged_output.csv'
features_df = pd.read_csv(csv_file_path)

# Assuming 'labels' is the column with your target labels


# Assuming 'labels' column contains labels separated by space or another delimiter
# and is a string of labels for each sample
if 'labels' in features_df.columns:
    features_df['labels'] = features_df['labels'].apply(lambda x: x.split())  # Adjust split logic as needed
    labels = features_df['labels']
    labels = features_df['labels'].copy()  # Copy labels to a separate Series
    features_df.drop(['Filename', 'labels'], axis=1, inplace=True, errors='ignore')  # Drop filename and labels for feature processing

mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels)
encoded_labels_df = pd.DataFrame(labels_encoded, columns=mlb.classes_)

# Check and handle NaN values
print(features_df.isna().sum())
if features_df.isna().any().any():
    print("There are NaN values in the dataset.")
    # Drop columns where all values are NaN
    features_df.dropna(axis=1, how='all', inplace=True)

# Verify changes
print("Remaining columns after dropping full NaN columns:", features_df.shape[1])


# Impute missing values
# Apply imputation
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

# Verify no NaN values are left
if data_filled.isna().any().any():
    print("There are still NaN values in the dataset.")
else:
    print("No NaN values remain in the dataset.")


# PCA for dimensionality reduction
# PCA for dimensionality reduction, if applicable
if data_filled.shape[1] > 1:
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(data_filled)
    print("Reduced feature dimensions:", reduced_features.shape)

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(reduced_features)
    processed_df = pd.DataFrame(normalized_features)
    processed_df = pd.DataFrame(normalized_features, columns=[f'PC{i+1}' for i in range(normalized_features.shape[1])])
    processed_df['labels'] = labels.values  # Re-add labels to the processed DataFrame
else:
    print("Not enough features for PCA after dropping all NaN columns.")


# Combine processed features with encoded labels
processed_df.to_csv('processed_features.csv', index=False)
reloaded_df = pd.read_csv('processed_features.csv')
print(reloaded_df.head())  # Check the first few rows


# Verify the final output
reloaded_df = pd.read_csv('processed_features.csv')
print(reloaded_df.head())
