# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import tensorflow as tf
import os
import numpy as np



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
        if pts1.shape == tf.TensorShape([2]) :
            pts1=tf.reshape(pts1,shape=(1,2))
            
        if pts2.shape == tf.TensorShape([2]) :
            pts2=tf.reshape(pts2,shape=(1,2))
            
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
            !!! num_cam1 et num_cam2  prennent leurs valeurs dans 0,1,2
            
            '''
            P1t = self.p_matrix[num_cam1]
            P2t = self.p_matrix[num_cam2]
            Fundamental = tf.constant([self.fundamental_matrix_from_projections(P1t,P2t).numpy()])
            d = self.tf_symmetrical_epipolar_distance(point1,point2,Fundamental,squared=True,eps = 1e-8)
            return tf.exp(-m*d)

    def tf_triangulate_points(
        self, P1: tf.Tensor, P2: tf.Tensor, points1: tf.Tensor, points2: tf.Tensor
        ) -> tf.Tensor:
        r"""Reconstructs a bunch of points by triangulation.

        Triangulates the 3d position of 2d correspondences between several images.
        Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

        The input points are assumed to be in homogeneous coordinate system and being inliers
        correspondences. The method does not perform any robust estimation.

        Args:
            P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
            P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
            points1: The set of points seen from the first camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            points2: The set of points seen from the second camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.

        Returns:
            The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.

        """
        if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
            raise AssertionError(P1.shape)
        if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
            raise AssertionError(P2.shape)
        if len(P1.shape[:-2]) != len(P2.shape[:-2]):
            raise AssertionError(P1.shape, P2.shape)
        if not (len(points1.shape) >= 2 and points1.shape[-1] == 2):
            raise AssertionError(points1.shape)
        if not (len(points2.shape) >= 2 and points2.shape[-1] == 2):
            raise AssertionError(points2.shape)
        if len(points1.shape[:-2]) != len(points2.shape[:-2]):
            raise AssertionError(points1.shape, points2.shape)
        if len(P1.shape[:-2]) != len(points1.shape[:-2]):
            raise AssertionError(P1.shape, points1.shape)

        # allocate and construct the equations matrix with shape (*, 4, 4)

        a = [points1.shape[0],points1.shape[1]]
        b = [points2.shape[0],points2.shape[1]]
        points_shape = max(a,b)

        #points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
        X = tf.zeros(points_shape[:-1] + [4, 4])  #.type_as(points1)
        temp = np.zeros(points_shape[:-1] + [4,4])


        for i in range(4):
            temp[:, 0, i] = points1[:, 0] * P1[ 2:3, i] - P1[ 0:1, i]
            temp[:, 1, i] = points1[:, 1] * P1[ 2:3, i] - P1[ 1:2, i]
            temp[:, 2, i] = points2[:, 0] * P2[ 2:3, i] - P2[ 0:1, i]
            temp[:, 3, i] = points2[:, 1] * P2[ 2:3, i] - P2[ 1:2, i]

        X = tf.convert_to_tensor(temp)

        # 1. Solve the system Ax=0 with smallest eigenvalue
        # 2. Return homogeneous coordinates

        _, _, V = tf.linalg.svd(X)

        points3d_h = V[..., -1]
        points3d: tf.Tensor = self.tf_convert_points_from_homogeneous(points3d_h)
        return points3d



    def tf_convert_points_from_homogeneous(self, points, eps: float = 1e-8) :
        r"""Function that converts points from homogeneous to Euclidean space.

        Args:
            points: the points to be transformed of shape :math:`(B, N, D)`.
            eps: to avoid division by zero.

        Returns:
            the points in Euclidean space :math:`(B, N, D-1)`.

        Examples:
            >>> input = torch.tensor([[0., 0., 1.]])
            >>> convert_points_from_homogeneous(input)
            tensor([[0., 0.]])
        """

        if not tf.is_tensor(points):

            raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")

        if len(points.shape) < 2:
            raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

        # we check for points at max_val
        z_vec  = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask = tf.abs(z_vec) > eps
        scale = tf.where(mask, 1.0 / (z_vec + eps), tf.ones_like(z_vec))

        return scale * points[..., :-1]



    

if __name__ == "__main__":

    '''
    En prenant le même point dans deux vues différentes, le score devrait être égal à 1 :
    '''
    import pickle
    import os
    file = os.path.join(os.path.dirname(os.path.abspath('epipolar_geo.py')), 
                        '../../data/Campus/voxel_2d_human_centers.pkl')
 #   file = os.path.abspath('../../data/Campus/voxel_2d_human_centers.pkl')
    a_file = open(file, "rb")
    human_centers = pickle.load(a_file)
    a_file.close()
    
    '''
    Les images vont de 704 à 1998
    
    '''
    cameras= ['camera_0','camera_1','camera_2']
    hc1 = human_centers['image_1004']['camera_0'][:]
    hc2 = human_centers['image_1004']['camera_1'][:]
    hc3 = human_centers['image_1004']['camera_2'][:]
    
    pt1=tf.constant(hc1,shape=hc1.shape,dtype=tf.float32)
    pt2=tf.constant(hc2,shape=hc2.shape,dtype=tf.float32)
    pt3=tf.constant(hc3,shape=hc3.shape,dtype=tf.float32)
    
    # compute Edge attribute 
    camera1=0
    camera2=1
    camera3=2
    
    edge1=epipolar_geometry().edge_score(pt1,pt2,camera1,camera2,10).numpy()
    edge2=epipolar_geometry().edge_score(pt1,pt3,camera1,camera3,10).numpy()
    edge3=epipolar_geometry().edge_score(pt2,pt3,camera2,camera3,10).numpy()
    
    print(edge1)
    print(edge2)
    print(edge3)

    # Test de la fonction triangulate
    tf.constant
    P1t=tf.constant([[439.0,180.81,-26.946,185.95],[-5.3416,88.523,-450.95,1324],[0.0060594,0.99348,-0.11385,5.227]], shape=(1,3,4))
    P2t=tf.constant([[162.36,-438.34,-17.508,3347.4],[73.3,-10.043,-443.34,1373.5],[0.99035,-0.047887,-0.13009,6.6849]], shape=(1,3,4))
    pt3Dt = tf.constant([[2.9872, 4.0063, 0.1581]])
    # passage en coord homogens (rajout d'un 1)
    pt3D_ht=tf_convert_points_to_homogeneous(pt3Dt)
    # projections
    # camera 2
    x2t=tf.matmul(P2t,tf.transpose(pt3D_ht, perm=(1,0)))
    x2t=x2t/x2t[2]
    # camera 1
    x1t=tf.matmul(P1t,tf.transpose(pt3D_ht, perm=(1,0)))
    x1t=x1t/x1t[2]
    x1t = tf.transpose(x1t[:2],(1,0)).numpy()

    test = epipolar_geometry().tf_triangulate_points(P1t,P2t,x1t,x1t)
    print("Hello")
    print(test)


