
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
    
    
    
    
if __name__ == "__main__":

	P2=tf.constant([[162.36,-438.34,-17.508,3347.4],[73.3,-10.043,-443.34,1373.5],[0.99035,-0.047887,-0.13009,6.6849]])
	P1=tf.constant([[439.0,180.81,-26.946,185.95],[-5.3416,88.523,-450.95,1324],[0.0060594,0.99348,-0.11385,5.227]])
	F=fundamental_matrix_from_projections(P1,P2)
	print(F)

