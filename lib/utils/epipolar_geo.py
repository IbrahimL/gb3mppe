import tensorflow as tf
import os


class epipolar_geometry():
    def __init__(self,dataset='campus'):
        '''
        On définit les matrices P de chacune des cameras ( campus )
        On calcule les matrices fondamentales, permettant la projection entre les vues
        
        '''
        
        # Matrice 1
        P1t=tf.constant([[439.0,180.81,-26.946,185.95],[-5.3416,88.523,-450.95,1324],[0.0060594,0.99348,-0.11385,5.227]])
        # Matrice 2
        P2t=tf.constant([[162.36,-438.34,-17.508,3347.4],[73.3,-10.043,-443.34,1373.5],[0.99035,-0.047887,-0.13009,6.6849]])
        # Matrice 3
        P3t=tf.constant([[237.58,679.93,-26.772,-1558.3],[-43.114,21.982,-713.6,1962.8],[-0.83557,0.53325,-0.13216,11.202]])

        # Construction des matrices de projections
        self.p_matrix=[P1t,P2t,P3t]
        self.ft12=tf.constant([self.fundamental_matrix_from_projections(P1t,P2t).numpy()])
        self.ft13=tf.constant([self.fundamental_matrix_from_projections(P1t,P3t).numpy()])
        self.ft21=tf.constant([self.fundamental_matrix_from_projections(P2t,P1t).numpy()])
        self.ft23=tf.constant([self.fundamental_matrix_from_projections(P2t,P3t).numpy()])
        self.ft31=tf.constant([self.fundamental_matrix_from_projections(P3t,P1t).numpy()])
        self.ft32=tf.constant([self.fundamental_matrix_from_projections(P3t,P2t).numpy()])

    def tf_convert_points_to_homogeneous(self,points):
        """Function that converts points from Euclidean to homogeneous space.

        See :class:`~torchgeometry.ConvertPointsToHomogeneous` for details.

        Examples::

            >>> input = torch.rand(2, 4, 3)  # BxNx3
            >>> output = tgm.convert_points_to_homogeneous(input)  # BxNx4
        """
        if not tf.is_tensor(points):
            raise TypeError("Input type is not a tf.Tensor. Got {}".format(
                type(points)))
        if len(points.shape) < 2:
            raise ValueError("Input must be at least a 2D tensor. Got {}".format(
                points.shape))
        paddings = tf.constant([[0,0], [0,1]])

        return tf.pad(points, paddings, mode='CONSTANT', constant_values=1)


    def tf_symmetrical_epipolar_distance(self,pts1,pts2,Fm, squared= True, eps = 1e-8) : 
        """
        Return symmetrical epipolar distance for correspondences given the fundamental matrix.

        """

        if not isinstance(Fm, tf.Tensor):
            raise TypeError(f"Fm type is not a tf.Tensor. Got {type(Fm)}")

        if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
            raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

            ###########Acompleter
        if pts1.shape[1] == 2:
            pts1 = self.tf_convert_points_to_homogeneous(pts1)

        if pts2.shape[1] == 2:
            pts2 = self.tf_convert_points_to_homogeneous(pts2)

        # From Hartley and Zisserman, symmetric epipolar distance (11.10)
        # sed = (x'^T F x) ** 2 /  (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))
        # Instead we can just transpose F once and switch the order of multiplication

        F_t = tf.transpose(Fm, perm=(0,2,1), conjugate=False, name='permute')
        line1_in_2 = pts1 @ F_t
        line2_in_1 = pts2 @ Fm

        # numerator = (x'^T F x) ** 2
        #numerator  = (pts2 * line1_in_2).sum(2).pow(2)
        numerator  = tf.pow(tf.math.reduce_sum((pts2 * line1_in_2),2),2)


        # denominator_inv =  1/ (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))

        denominator_inv = 1.0 / (tf.pow(tf.norm(line1_in_2[..., :2],axis=2),2)) + 1.0 / (
            tf.pow(tf.norm(line2_in_1[..., :2],axis=2),2)
        )

        out = numerator * denominator_inv
        if squared:
            return out
        return tf.math.sqrt(out + eps)

    def fundamental_matrix_from_projections(self,P1, P2):
        """
        Get the Fundamental matrix from Projection matrices.
        Adapted from 
        [
        https://kornia.readthedocs.io/en/latest/_modules/kornia/
        geometry/epipolar/fundamental.html
        ]
        """
        if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
            raise AssertionError(P1.shape)
        if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
            raise AssertionError(P2.shape)
        if P1.shape[:-2] != P2.shape[:-2]:
            raise AssertionError

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
    
    def edge_score(self,point1,point2,num_cam1,num_cam2,m=10):
        '''
        Compute edge score between  two 2D projections, given the camera number of each projection
        m is a constant, its value is choosen empirically
        !!! num_cam1 et num_cam2  prennent leurs valeurs dans 1,2,3
        
        '''
        P1t = self.p_matrix[num_cam1-1]
        P2t = self.p_matrix[num_cam2-1]
        Fundamental = tf.constant([self.fundamental_matrix_from_projections(P1t,P2t).numpy()])
        d = self.tf_symmetrical_epipolar_distance(point1,point2,Fundamental,squared=True,eps = 1e-8)
        return tf.exp(-m*d)

    

if __name__ == "__main__":

    '''
    En prenant le même point dans deux vues différentes, le score devrait être égal à 1 :
    '''
    import sys
    sys.path.append(os.path.dirname(os.path.abspath('../dataset/gt_campus.py')))
    from gt_campus import *
    
    campus_gt=gt()
    frame=10
    actor1=2
    actor2=2
    camera1=1
    camera2=3
    pt1=tf.constant(gt().get_human_center(frame,actor1,camera1),shape=(1,2),dtype=tf.float32)
    pt2=tf.constant(gt().get_human_center(frame,actor2,camera2),shape=(1,2),dtype=tf.float32)
    # compute Edge attribute 
    s=epipolar_geometry().edge_score(pt1,pt2,camera1,camera2,10).numpy()
    print(s)