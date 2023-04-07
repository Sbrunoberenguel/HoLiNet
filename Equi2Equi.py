import cv2
import numpy as np

#--------------------------#
'''
Program to deal with the geometric part of the dataset
composed by the video sequence of the Fix Wing drone.
'''
#--------------------------#
np.set_printoptions(precision=2,suppress=True)

Theta_init = -np.pi
Theta_end  = np.pi
varPhi_init= np.pi/2.0
varPhi_end = -np.pi/2.0

def SVD(A):
	A = np.array(A,dtype=np.float64)
	u,s,vh = np.linalg.svd(A)
	return u,s,vh

# Transform pixel coordinates to spherical coordinates (Equirectangular projection)
def px2vec(px,H=512,W=1024):
    y,x = px
    theta =  x*(Theta_end-Theta_init)/W + Theta_init
    phi = y*(varPhi_end-varPhi_init)/H + varPhi_init
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*st),
                    (-sp),
                    (cp*ct)]).reshape(3,-1)
    return vec

def rotate_img(img,Rot):
    H,W = img.shape[:2]
    rimg = np.zeros_like(img)
    vec = get_vec(W,H)
    rvec = np.dot(Rot,vec)
    varPhi,theta = xyz2angles(rvec)
    u,v = angles2xy(varPhi,theta,H,W)
    p_x = np.asarray(u,dtype=np.int32)
    p_y = np.asarray(v,dtype=np.int32)
    rimg = img[p_y,p_x].reshape(H,W,-1)
    return rimg

#Generate 3D vectors in the sphere for an image of HxW resolution
def get_vec(W,H):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta =  x*(Theta_end-Theta_init)/W + Theta_init
    phi = y*(varPhi_end-varPhi_init)/H + varPhi_init
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*st),
                    (-sp),
                    (cp*ct)]).reshape(3,-1)
    return vec

#Generate rotation matrix arround a generic axis
def rotation_matrix( axis, angle):
    if angle == 0:
        return np.eye(3)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    ROT = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], dtype=float).reshape(3,3)
    return ROT

#Transform spherical angles into 2D pixels (equirectangular projection)
def angles2xy(varPhi,theta,H,W):
    x = W*(theta-Theta_init)/(Theta_end-Theta_init) - 0.5
    y = H*(varPhi-varPhi_init)/(varPhi_end-varPhi_init) - 0.5
    return x,y

# Get spherical coordinates from 3D points (equirectangular projection)
def xyz2angles(point):
    x,y,z = point
    theta = np.arctan2(x,z)
    varPhi = np.arctan2(-y,np.sqrt(x**2+z**2))
    return varPhi,theta

#Ransac approach to get a main direction from the vanishing points
def RANSAC_VP(VP):
    # Tunning Parameters 
    early_break = 0.8       # % of data that is an inlier of a solution
    list_of_data = VP>0.99   # 'Confidence' of the horizon line
    eps = np.radians(20)     # Acceptable error for inlier count

    H,W = VP.shape
    P,eps_in = 0.999,0.7
    v,u = np.where(list_of_data)
    pts = np.asarray([v,u])
    vec = px2vec(pts,H,W)
    num_data = v.shape[0]
    m= int(min(8,num_data)) # Number of sumples at a time
    Ransac_max_iter = min(1e2,np.log(1-P)/np.log(1-eps_in**m))
    best_vote = 0
    best_g = np.array([0,1,0])
    for _ in range(int(Ransac_max_iter)):
        if best_vote >= num_data*early_break: 
            break
        idx = np.random.choice(np.arange(num_data),int(m),replace=False)
        A = vec[:,idx].T
        A = A * np.sign(A[:,1]).reshape(m,1)
        g = np.mean(A,axis=0)
        g = g / np.linalg.norm(g)
        distance = abs(np.dot(g,vec))
        vote = np.sum(distance>np.cos(eps))
        if vote > best_vote:
            best_vote = vote
            best_g = g
    return best_g

#Ransac approach to get a 3D plane from the horizon line and the direction of the Vanishing points
def RANSAC_HL(HL,ref):
    # Tunning Parameters 
    list_of_data = HL>0.99  # 'Confidence' of the horizon line
    eps = 5e-3              # Acceptable error for inlier count
    eps2 = 0.7              ## not np.cos(np.randias(30))

    H,W = HL.shape
    P,eps_in = 0.999,0.7
    v,u = np.where(list_of_data)
    pts = np.asarray([v,u])
    vec = px2vec(pts,H,W)
    num_data = v.shape[0]
    m= int(min(8,num_data)) # Number of sumples at a time
    Ransac_max_iter = min(np.log(1-P)/np.log(1-eps_in**m),1e5)
    votes = []
    planes = []
    for _ in range(int(2*Ransac_max_iter)+1):
        multiplyer = 1
        idx = np.random.choice(np.arange(num_data),int(m),replace=False)
        # idx.sort()
        A = vec[:,idx].T
        _,_,Vt = SVD(A)
        n = np.transpose(Vt)[:,-1]
        n = n / np.linalg.norm(n)
        n = n*np.sign(n[1])
        distance2 = abs(np.dot(n,ref))
        if distance2 < eps2:
            multiplyer = 0.5
        distance = abs(np.dot(n,vec))
        vote = np.sum(distance<eps) * multiplyer
        votes.append(vote)
        planes.append(n)
    votes = np.asarray(votes)
    planes = np.asarray(planes)
    idx = np.where(votes==votes.max())[0]
    return planes[idx[0]]

#Main program to get the orientation from the horizon line and vanishing points
def HL2Plane(HL,VP):
        HL = HL / HL.max()
        VP = VP / VP.max()
        gravity = RANSAC_VP(VP)
        Plane = RANSAC_HL(HL,gravity)
        return Plane

if __name__ == '__main__':
    ref = cv2.imread('hl.png')
    img = cv2.imread('e_000000.png')
    angles = np.load('orientation_v2.npy')
    vec = angles[0,:]
    axis = vec / np.linalg.norm(vec)
    ang = np.linalg.norm(vec)
    Rot = rotation_matrix(axis,ang).T
    rref = rotate_img(ref,Rot)
    out = cv2.addWeighted(rref,1,img,0.75,1)
    cv2.imwrite('rot_0.png',out)