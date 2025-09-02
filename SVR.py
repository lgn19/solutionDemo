import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import joblib
import pandas as pd
import csv


# Calculate distances from points to a line segment
def points_to_segment_distances(points, pos1, pos2):
    # Segment vector
    segment_vector = pos2 - pos1  # (3,)
    # Vectors from the segment's start point to each point
    point_vectors = points - pos1  # (49, 3)
    # Calculate projection ratio t using row-wise dot product
    t = np.einsum('ij,j->i', point_vectors, segment_vector) / np.dot(segment_vector, segment_vector)
    # Restrict t to the range [0, 1]
    t = np.clip(t, 0, 1)  # (49,)
    # Calculate the position of the foot of the perpendicular
    projections = pos1 + np.outer(t, segment_vector)  # (49, 3)
    # Return the distance from the point to the foot of the perpendicular
    return np.linalg.norm(points - projections, axis=1)  # (49,)


# Calculate angles in degrees between vectors and a reference line
def get_degree(vector, line):
    dot_products = np.einsum('ij,j->i', vector, line)
    length_line = np.linalg.norm(line)
    length_vectors = np.linalg.norm(vector, axis=1)
    cos_theta = dot_products / (length_line * length_vectors)
    # Prevent floating-point errors from exceeding [-1, 1]
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angles_deg = np.degrees(angles)
    return angles_deg


# Find intersection point of two lines (y = m1x + b1 and y = m2x + b2)
def find_intersection(m1, b1, m2, b2):
    if m1 == m2:
        if b1 == b2:
            return "The two lines are coincident, no unique intersection point"
        else:
            return "The two lines are parallel, no intersection point"
    x = (np.float16(b2) - np.float16(b1)) / (np.float16(m1) - np.float16(m2))
    y = np.float16(m1) * np.float16(x) + np.float16(b1)
    return (x, y)


# Calculate circle parameters based on angle shifts
def calculate_circle_d(shift_A, shift_B):
    if shift_A == shift_B:
        return None, None
    else:
        m1 = np.tan(np.radians(shift_A))
        m2 = np.tan(np.radians(shift_B))
        b1 = -(m1 * 0.3)
        b2 = (m2 * 0.3)
        x, r = find_intersection(m1, b1, m2, b2)
        return -(x - 2.15), r


# Calculate intersection point of two circles
def circle_intersection(x, r1, r2):
    r1 = np.abs(r1)
    r2 = np.abs(r2)
    z1 = 0
    center1 = np.array([x, 0, z1])
    z2 = z1 + 1
    center2 = np.array([x, 0, z2])
    r1_squared = r1 ** 2
    r2_squared = r2 ** 2
    d = z2 - z1

    if d > r1 + r2 or d < abs(r1 - r2):
        return [0, 0, 0]

    a = (r1_squared - r2_squared + d ** 2) / (2 * d)
    h = np.sqrt(r1_squared - a ** 2)
    z_intersect = z1 + a
    y_intersect1 = h
    y_intersect2 = -h
    return np.array([x, y_intersect1 + 0.6, z_intersect + 1])


# Define reference positions
pos1 = np.array([1.9, 0.6, 1])
pos2 = np.array([2.5, 0.6, 1])
pos3 = np.array([1.9, 0.6, 2])
pos4 = np.array([2.5, 0.6, 2])

# Load data
x = np.load("sample/X.npy")
y = np.load("sample/Y.npy")

# Reshape data
y1 = y.reshape(-1, 3)
x1 = x.reshape(-1, 4)

# Remove invalid entries (-1)
y1 = y1[~np.any(x1 == -1, axis=1)]
x1 = x1[~np.any(x1 == -1, axis=1)]

# Calculate geometric features
# Group 1
A1, B1 = pos1, pos2
AB1 = A1 - B1
vectors1 = y1 - A1  # Vectors from each point to A1
vectors2 = y1 - B1  # Vectors from each point to B1

# Group 2
A2, B2 = pos3, pos4
AB2 = A2 - B2
vectors3 = y1 - A2  # Vectors from each point to A2
vectors4 = y1 - B2  # Vectors from each point to B2

# Calculate angles and distances
degree1 = get_degree(vectors1, AB1)
degree2 = get_degree(vectors2, AB1)
degree3 = get_degree(vectors3, AB2)
degree4 = get_degree(vectors4, AB2)

R1 = points_to_segment_distances(y1, pos1, pos2)
R2 = points_to_segment_distances(y1, pos3, pos4)

distances_to_mic = y1[:, 1] - 0.6

# Prepare features and targets
features = x1
targets = np.column_stack((
    R1, R2,
    y1[:, 0], y1[:, 1], y1[:, 2],  # x, y, z coordinates
    degree1, degree2, degree3, degree4  # Angles
))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, targets,
    test_size=0.2,
    random_state=25
)

# ------------------------------
# soluton 1: Angle Prediction
# ------------------------------
# Train SVR models for angle prediction
# Model for degree1
model_d1 = SVR(kernel='poly', C=12.0, epsilon=0.01, degree=2, gamma='scale', coef0=10.0)
model_d1.fit(X_train, y_train[:, 5])
y_pred_d1 = model_d1.predict(X_test)

# Model for degree2
model_d2 = SVR(kernel='poly', C=12.0, epsilon=0.01, degree=2, gamma='scale', coef0=10.0)
model_d2.fit(X_train, y_train[:, 6])
y_pred_d2 = model_d2.predict(X_test)

# Model for degree3
model_d3 = SVR(kernel='poly', C=12.0, epsilon=0.01, degree=2, gamma='scale', coef0=10.0)
model_d3.fit(X_train, y_train[:, 7])
y_pred_d3 = model_d3.predict(X_test)

# Model for degree4
model_d4 = SVR(kernel='poly', C=12.0, epsilon=0.01, degree=2, gamma='scale', coef0=10.0)
model_d4.fit(X_train, y_train[:, 8])
y_pred_d4 = model_d4.predict(X_test)

# Evaluate angle prediction performance
rmse_d1 = np.sqrt(mean_squared_error(y_test[:, 5], y_pred_d1))
r2_d1 = r2_score(y_test[:, 5], y_pred_d1)

rmse_d2 = np.sqrt(mean_squared_error(y_test[:, 6], y_pred_d2))
r2_d2 = r2_score(y_test[:, 6], y_pred_d2)

rmse_d3 = np.sqrt(mean_squared_error(y_test[:, 7], y_pred_d3))
r2_d3 = r2_score(y_test[:, 7], y_pred_d3)

rmse_d4 = np.sqrt(mean_squared_error(y_test[:, 8], y_pred_d4))
r2_d4 = r2_score(y_test[:, 8], y_pred_d4)

print("[soluton 1] Angle Prediction Metrics:")
print(f"  Degree1 - RMSE: {rmse_d1:.4f}, R²: {r2_d1:.4f}")
print(f"  Degree2 - RMSE: {rmse_d2:.4f}, R²: {r2_d2:.4f}")
print(f"  Degree3 - RMSE: {rmse_d3:.4f}, R²: {r2_d3:.4f}")
print(f"  Degree4 - RMSE: {rmse_d4:.4f}, R²: {r2_d4:.4f}")

# Calculate 3D points using predicted angles
calculate_points = []
calculate_r1 = []
calculate_r2 = []
for k in range(y_pred_d1.shape[0]):
    x1, r1 = calculate_circle_d(y_pred_d1[k], y_pred_d2[k])
    x2, r2 = calculate_circle_d(y_pred_d3[k], y_pred_d4[k])
    tem_point = circle_intersection((x1 + x2) / 2, r1, r2)
    calculate_points.append(tem_point)
    calculate_r1.append(r1)
    calculate_r2.append(r2)

calculate_points = np.array(calculate_points)
calculate_r1 = np.array(calculate_r1)
calculate_r2 = np.array(calculate_r2)

# Evaluate 3D point prediction from angles
rmse_r1 = np.sqrt(mean_squared_error(y_test[:, 0], calculate_r1))
r2_r1 = r2_score(y_test[:, 0], calculate_r1)

rmse_r2 = np.sqrt(mean_squared_error(y_test[:, 1], calculate_r2))
r2_r2 = r2_score(y_test[:, 1], calculate_r2)

rmse_x = np.sqrt(mean_squared_error(y_test[:, 2], calculate_points[:, 0]))
r2_x = r2_score(y_test[:, 2], calculate_points[:, 0])

rmse_y = np.sqrt(mean_squared_error(y_test[:, 3], calculate_points[:, 1]))
r2_y = r2_score(y_test[:, 3], calculate_points[:, 1])

rmse_z = np.sqrt(mean_squared_error(y_test[:, 4], calculate_points[:, 2]))
r2_z = r2_score(y_test[:, 4], calculate_points[:, 2])

# Calculate spatial distance error
array1 = y_test[:, 2:5]
array2 = calculate_points
distances = np.linalg.norm(array1 - array2, axis=1)
mean_distance_error = np.mean(distances)

print("\n[soluton 1] 3D Point Prediction from Angles:")
print(f"  R1 - RMSE: {rmse_r1:.4f}, R²: {r2_r1:.4f}")
print(f"  R2 - RMSE: {rmse_r2:.4f}, R²: {r2_r2:.4f}")
print(f"  X Coordinate - RMSE: {rmse_x:.4f}, R²: {r2_x:.4f}")
print(f"  Y Coordinate - RMSE: {rmse_y:.4f}, R²: {r2_y:.4f}")
print(f"  Z Coordinate - RMSE: {rmse_z:.4f}, R²: {r2_z:.4f}")
print(f"  Mean Spatial Distance Error: {mean_distance_error:.4f}")

print("\n------------------------------")

# ------------------------------
# soluton 2: Direct R and X Prediction
# ------------------------------
# Test data for prediction
XR = X_test  # Use original test features

# Train SVR models for direct prediction
# Model for R1
model_r1 = SVR(kernel='rbf', C=100, epsilon=0.000001)
model_r1.fit(X_train, y_train[:, 0])
y_pred_r1 = model_r1.predict(X_test)

# Model for R2
model_r2 = SVR(kernel='rbf', C=100, epsilon=0.01)
model_r2.fit(X_train, y_train[:, 1])
y_pred_r2 = model_r2.predict(X_test)

# Model for X coordinate
model_x = SVR(kernel='rbf', C=12, epsilon=0.01)
model_x.fit(X_train, y_train[:, 2])
y_pred_x = model_x.predict(XR)


# Evaluate direct prediction performance
rmse_r1_direct = np.sqrt(mean_squared_error(y_test[:, 0], y_pred_r1))
r2_r1_direct = r2_score(y_test[:, 0], y_pred_r1)

rmse_r2_direct = np.sqrt(mean_squared_error(y_test[:, 1], y_pred_r2))
r2_r2_direct = r2_score(y_test[:, 1], y_pred_r2)

rmse_x_direct = np.sqrt(mean_squared_error(y_test[:, 2], y_pred_x))
r2_x_direct = r2_score(y_test[:, 2], y_pred_x)

# Calculate 3D points using direct predictions
calculate_points_direct = []
for k in range(y_pred_r1.shape[0]):
    tem_point = circle_intersection(y_pred_x[k], y_pred_r1[k], y_pred_r2[k])
    calculate_points_direct.append(tem_point)
calculate_points_direct = np.array(calculate_points_direct)

# Evaluate 3D point prediction from direct features
rmse_x_final = np.sqrt(mean_squared_error(y_test[:, 2], calculate_points_direct[:, 0]))
r2_x_final = r2_score(y_test[:, 2], calculate_points_direct[:, 0])

rmse_y_final = np.sqrt(mean_squared_error(y_test[:, 3], calculate_points_direct[:, 1]))
r2_y_final = r2_score(y_test[:, 3], calculate_points_direct[:, 1])

rmse_z_final = np.sqrt(mean_squared_error(y_test[:, 4], calculate_points_direct[:, 2]))
r2_z_final = r2_score(y_test[:, 4], calculate_points_direct[:, 2])

# Calculate spatial distance error
array1 = y_test[:, 2:5]
array2 = calculate_points_direct
distances_direct = np.linalg.norm(array1 - array2, axis=1)
mean_distance_error_direct = np.mean(distances_direct)

print("\n[soluton 2] Direct Prediction Metrics:")
print(f"  R1 - RMSE: {rmse_r1_direct:.4f}, R²: {r2_r1_direct:.4f}")
print(f"  R2 - RMSE: {rmse_r2_direct:.4f}, R²: {r2_r2_direct:.4f}")
print(f"  X Coordinate (Direct) - RMSE: {rmse_x_direct:.4f}, R²: {r2_x_direct:.4f}")
print(f"  X Coordinate (Final) - RMSE: {rmse_x_final:.4f}, R²: {r2_x_final:.4f}")
print(f"  Y Coordinate - RMSE: {rmse_y_final:.4f}, R²: {r2_y_final:.4f}")
print(f"  Z Coordinate - RMSE: {rmse_z_final:.4f}, R²: {r2_z_final:.4f}")
print(f"  Mean Spatial Distance Error: {mean_distance_error_direct:.4f}")

# Save error metrics to CSV
errors_krr_r1 = np.abs(y_pred_r1 - y_test[:, 0])
errors_krr_r2 = np.abs(y_pred_r2 - y_test[:, 1])
errors_krr_x = np.abs(calculate_points_direct[:, 0] - y_test[:, 2])
errors_krr_y = np.abs(calculate_points_direct[:, 1] - y_test[:, 3])
errors_krr_z = np.abs(calculate_points_direct[:, 2] - y_test[:, 4])

errors_combined = np.array([
    errors_krr_r1,
    errors_krr_r2,
    errors_krr_x,
    errors_krr_y,
    errors_krr_z,
    distances_direct
])


print("\n------------------------------")

# ------------------------------
# soluton 3: Y and Z Coordinate Prediction
# ------------------------------
# Train SVR models for Y and Z coordinates
# Model for Y

model_y = SVR(kernel='rbf', C=200, epsilon=0.01)
model_y.fit(X_train, y_train[:, 3])
y_pred_y = model_y.predict(XR)

# Model for Z
model_z = SVR(kernel='rbf', C=200, epsilon=0.01)
model_z.fit(X_train, y_train[:, 4])
y_pred_z = model_z.predict(XR)

# Get number of support vectors for Z model
num_support_vectors_z = model_z.support_vectors_.shape[0]
# Evaluate Y and Z prediction performance
rmse_y_direct = np.sqrt(mean_squared_error(y_test[:, 3], y_pred_y))
r2_y_direct = r2_score(y_test[:, 3], y_pred_y)

rmse_z_direct = np.sqrt(mean_squared_error(y_test[:, 4], y_pred_z))
r2_z_direct = r2_score(y_test[:, 4], y_pred_z)

# Calculate full 3D coordinates from direct predictions (X from step 2, Y and Z from step 3)
full_predicted_points = np.column_stack((
    calculate_points_direct[:, 0],  # X from step 2
    y_pred_y,                       # Y from step 3
    y_pred_z                        # Z from step 3
))

# Evaluate full 3D position prediction
rmse_x_full = np.sqrt(mean_squared_error(y_test[:, 2], full_predicted_points[:, 0]))
rmse_y_full = np.sqrt(mean_squared_error(y_test[:, 3], full_predicted_points[:, 1]))
rmse_z_full = np.sqrt(mean_squared_error(y_test[:, 4], full_predicted_points[:, 2]))

r2_x_full = r2_score(y_test[:, 2], full_predicted_points[:, 0])
r2_y_full = r2_score(y_test[:, 3], full_predicted_points[:, 1])
r2_z_full = r2_score(y_test[:, 4], full_predicted_points[:, 2])

# Calculate overall spatial distance error
full_distances = np.linalg.norm(y_test[:, 2:5] - full_predicted_points, axis=1)
mean_full_distance_error = np.mean(full_distances)

print("\n[soluton 3] Full 3D Position Prediction:")
print(f"  X Coordinate - RMSE: {rmse_x_full:.4f}, R²: {r2_x_full:.4f}")
print(f"  Y Coordinate - RMSE: {rmse_y_full:.4f}, R²: {r2_y_full:.4f}")
print(f"  Z Coordinate - RMSE: {rmse_z_full:.4f}, R²: {r2_z_full:.4f}")
print(f"  Mean Spatial Distance Error: {mean_full_distance_error:.4f}")

# Save full prediction results (optional)
prediction_results = pd.DataFrame({
    'True_X': y_test[:, 2],
    'Pred_X': full_predicted_points[:, 0],
    'True_Y': y_test[:, 3],
    'Pred_Y': full_predicted_points[:, 1],
    'True_Z': y_test[:, 4],
    'Pred_Z': full_predicted_points[:, 2],
    'Distance_Error': full_distances
})

