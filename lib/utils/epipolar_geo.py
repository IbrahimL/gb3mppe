
def fundamental_matrix_from_projections(P1, P2):
    """
    Get the Fundamental matrix from Projection matrices P1 & P2
    Adapted from 
    [
    https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/epipolar
    /_metrics.html#sampson_epipolar_distance
    ]
    
    """
    def vstack(x, y):
            return tf.concat([x,y], axis=0, name='concat')
    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)
    F_vec = tf.concat(
        [
            tf.reshape(tf.linalg.det(X1Y1),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X2Y1),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X3Y1),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X1Y2),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X2Y2),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X3Y2),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X1Y3),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X2Y3),shape=(-1,1)),
            tf.reshape(tf.linalg.det(X3Y3),shape=(-1,1)),
        ],
        axis=-1
    )

    return tf.reshape(F_vec,shape=(*P1.shape[:-2],3,3))
    
    
    
 

def tf_symmetrical_epipolar_distance(pts1,pts2,Fm, squared, eps) : 
    """
    Return symmetrical epipolar distance for correspondences given the fundamental matrix.

    """
    
    if not isinstance(Fm, tf.Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

        ###########Acompleter
 #   if pts1.size(-1) == 2:
 #       pts1 = convert_points_to_homogeneous(pts1)

 #   if pts2.size(-1) == 2:
  #      pts2 = convert_points_to_homogeneous(pts2)


    # From Hartley and Zisserman, symmetric epipolar distance (11.10)
    # sed = (x'^T F x) ** 2 /  (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))
    # Instead we can just transpose F once and switch the order of multiplication
    F_t = tf.transpose(Fm, perm=(0,2,1), conjugate=False, name='permute')
    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator  = tf.pow(tf.math.reduce_sum((pts2 * line1_in_2),2),2)


    # denominator_inv =  1/ (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))    
    denominator_inv = 1.0 / (tf.pow(tf.norm(line1_in_2[..., :2],axis=2),2)) + 1.0 / (
        tf.pow(tf.norm(line2_in_1[..., :2],axis=2),2)
    )
    
    out = numerator * denominator_inv
    if squared:
        return out
    return tf.math.sqrt(out + eps)

if __name__ == "__main__":

	P2=tf.constant([[162.36,-438.34,-17.508,3347.4],[73.3,-10.043,-443.34,1373.5],[0.99035,-0.047887,-0.13009,6.6849]])
	P1=tf.constant([[439.0,180.81,-26.946,185.95],[-5.3416,88.523,-450.95,1324],[0.0060594,0.99348,-0.11385,5.227]])
	F=fundamental_matrix_from_projections(P1,P2)
	print(F)

