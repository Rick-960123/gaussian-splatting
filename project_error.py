import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

poses = Variable(torch.randn(num_poses, 6), requires_grad=True)
landmarks = Variable(torch.randn(num_landmarks, 3), requires_grad=True)

def reprojection_error(poses, landmarks, measurements):
    num_poses = poses.shape[0]
    num_landmarks = landmarks.shape[0]
    total_error = 0.0

    for pose_idx, landmark_idx, observed_point in measurements:

        current_pose = poses[pose_idx]
        current_landmark = landmarks[landmark_idx]

        transformed_landmark = torch.matmul(current_pose[:3, :3], current_landmark) + current_pose[:3, 3]

        projected_point = transformed_landmark[:2] / transformed_landmark[2]

        observed_pixel = observed_point[:2]
        error = torch.norm(projected_point - observed_pixel)

        total_error += error.item()

    return total_error / len(measurements)

optimizer = optim.Adam([poses, landmarks], lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    loss = reprojection_error(poses, landmarks, measurements)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

# 最终优化后的结果
optimized_poses = poses.data.numpy()
optimized_landmarks = landmarks.data.numpy()

def DataReader():
    all_points = np.array([])
    with open("/home/rick/Datasets/Custom/groundtruth.txt") as f:
        index = 0
        while True:
            index += 1
            line = f.readline()
            if line.startswith("#"):
                continue
            p = [float(i) for i in line]
            pose = Pose(index,*p)
            if len(all_points) == 0:
                all_points = transform_points_world_to_body(points, pose)
            else:
                np.vstack((all_points, transform_points_world_to_body(points, pose)))
    save_point_cloud_to_ply("./test.ply",all_points)
