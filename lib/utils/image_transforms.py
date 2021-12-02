import numpy as np
import tensorflow as tf
import json

@tf.function
def tf_get_cam_params(camera):
    R = tf.constant(camera['R'], dtype=tf.float32)
    T = tf.constant(camera['T'], dtype=tf.float32)
    fx = tf.constant(camera['fx'], dtype=tf.float32)
    fy = tf.constant(camera['fy'], dtype=tf.float32)
    f = tf.reshape(tf.stack([fx, fy], axis=-1), (2,1))
    cx = tf.constant(camera['cx'], dtype=tf.float32)
    cy = tf.constant(camera['cy'], dtype=tf.float32)
    c = tf.reshape(tf.stack([cx, cy], axis=-1), (2,1))
    k = tf.constant(camera['k'], dtype=tf.float32)
    p = tf.constant(camera['p'], dtype=tf.float32)
    return R, T, f, c, k, p

@tf.function
def tf_point_3d_to_2d(points, R, T, f, c, k, p):
    B, Np, Nj, Nd = points.shape # [batch_size, Number of persons, Number of joints, Dimension]
    
    points = points.reshape(B, -1, 3).transpose(1, 2) # [B, Np*Nj, Nd] --> [B, Nd, Np*Nj]
    
    points_cam = tf.matmul(R, points - T) # Translation + Rotation
    out = points_cam[:, :2, :]/(points_cam[:, 2:, :] + 1e-5) # devision by depth (and avoiding division by 0)
    
    # Camera distortion
    r = tf.reduce_sum(out ** 2, axis=1) # [B, Np*Nj]
    d = 1 + k[:, 0] * r + k[:, 1] * (r ** 2) + k[:, 2] * (r ** 3)
    u = out[:, 0, :] * d + 2 * p[:, 0] * out[:, 0, :] * out[:, 1, :] + p[:, 1] * (r + 2 * out[:, 0, :] * out[:, 0, :])  # [B, Np*Nj]
    v = out[:, 1, :] * d + 2 * p[:, 1] * out[:, 0, :] * out[:, 1, :] + p[:, 0] * (r + 2 * out[:, 1, :] * out[:, 1, :])  # [B, Np*Nj]
    out = tf.stack([u, v], axis=1)
    
    out_pixel = f * out + c # [B, 2, Np*Nj]
    out_pixel = out_pixel.transpose(1, 2) # [B, Np*Nj, 2]
    
    # get back the original shape
    out_pixel = out_pixel.reshape(B, Np, Nj, 2) # [B, Np, Nj, 2]
    return out_pixel

@tf.function
def tf_pose_3d_to_2d(points, camera):
    R, T, f, c, k, p = tf_get_cam_params(points, camera)
    return tf_point_3d_to_2d(points, R, T, f, c, k, p)

@tf.function
def tf_point_2d_to_3d(points, depth, R, T, f, c, k, p):
    B, Np, Nj, Nd = points.shape # [batch_size, Number of persons, Number of joints, Dimension]
    
    points = points.reshape(B, -1, 2).transpose(1, 2) # [B, 2, Np*Nj]
    
    out = (points - c) / f
    
    # remove distortion
    r = tf.reduce_sum(out ** 2, dim=1) # [B, Np*Nj]
    d = 1 - k[:, 0] * r - k[:, 1] * (r ** 2) - k[:, 2] * (r ** 3) # [B, Np*Nj]
    u = out[:, 0, :] * d - 2 * p[:, 0] * out[:, 0, :] * out[:, 1, :] - p[:, 1] * (r + 2 * out[:, 0, :] * out[:, 0, :])  # [B, Np*Nj]
    v = out[:, 1, :] * d - 2 * p[:, 1] * out[:, 0, :] * out[:, 1, :] - p[:, 0] * (r + 2 * out[:, 1, :] * out[:, 1, :])  # [B, Np*Nj]
    out = tf.stack([u, v], dim=1) # [B, 2, Np*Nj]
    
    # deproject
    out = tf.concat([out, tf.ones(B, 1, out.shape(-1))], dim=1) # [B, 2, Np*Nj]
    out = out.reshape(B, 3, Np, Nj) # [B, 2, Np, Nj]
    d = depth.reshape(B, 1, -1, 1) # for broadcasting
    out = out * d
    out = out.reshape(B, 3, -1) # [B, 3, Np*Nj]
    
    out_points = tf.matmul(tf.linalg.inverse(R), out) + T # [B, 3, Np*Nj]
    out_points = out_points.transpose(1, 2) # [B, Np*Nj, 3]
    out_points = out_points.reshape(B, Np, Nj, 3) # [B, Np, Nj, 3]
    
    return out_points

@tf.function
def tf_pose_2d_to_3d(points, depth, camera):
    R, T, f, c, k, p = tf_get_cam_params(points, depth, camera)
    return tf_point_2d_to_3d(points, R, T, f, c, k, p)

if __name__ == "__main__":
    f = open('../../data/Campus/calibration_campus.json')
    data = json.load(f)['0'] # Camera 1
    print(data)
    R, T, f, c, k, p = tf_get_cam_params(data)
    print(f)

    