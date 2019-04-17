import os,cv2
import numpy as np

def transform(x, y, ang, s0, s1):
    x0 = x - s0[1] / 2
    y0 = y - s0[0] / 2
    xx = x0 * np.cos(ang) - y0 * np.sin(ang) + s1[1]/2
    yy = x0 * np.sin(ang) + y0 * np.cos(ang) + s1[0]/2
    return xx, yy

def guard(x, n):
    x[x < 1] = 1
    if n[1] > 0:
        if x[0] > n[1]:
            x[0] = n[1]
        if x[1] > n[1]:
            x[1] = n[1]
    if n[0] > 0:
        if x[2] > n[0]:
            x[2] = n[0]
        if x[3] > n[0]:
            x[3] = n[0]
    x[x < 1] = 1
    return x


def align(img, keypoints, crop_size=128, ec_mc_y=48, ec_y=40):
    keypoints = keypoints.astype(np.float64) 
    if keypoints[0,0] == keypoints[1,0]:
        ang = 0
    else:
        ang_tan = (keypoints[0,1] - keypoints[1,1]) / \
            (keypoints[0,0] - keypoints[1,0])
        ang = np.arctan(ang_tan) / np.pi * 180
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2),ang, 1.0)
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    h, w = img.shape[:2]
    x = (keypoints[0,0] + keypoints[1,0]) / 2
    y = (keypoints[0,1] + keypoints[1,1]) / 2
    
    ang = -ang / 180 * np.pi
    xx, yy = transform(x, y, ang, img.shape[:2], img_rotate.shape[:2])
    eyec = np.array([np.round(xx), np.round(yy)])

    x = (keypoints[3,0] + keypoints[4,0]) / 2
    y = (keypoints[3,1] + keypoints[4,1]) / 2
    xx, yy = transform(x, y, ang, img.shape[:2], img_rotate.shape[:2])
    mouthc = np.array([np.round(xx), np.round(yy)])
    
    resize_scale = ec_mc_y / (mouthc[1] - eyec[1])
    if resize_scale < 0:
        resize_scale = 1
    img_resize = cv2.resize(img_rotate, (int(resize_scale*img_rotate.shape[0]), int(resize_scale*img_rotate.shape[1])))
    eyec2 = (eyec - np.array([img_rotate.shape[1]/2, img_rotate.shape[0]/2])) * resize_scale + np.array([img_resize.shape[1]/2, img_resize.shape[0]/2])
    eyec2 = np.round(eyec2)
    trans_points = np.zeros((5,2))
    trans_points[:,0], trans_points[:,1] = transform(keypoints[:,0],keypoints[:,1], ang, img.shape[:2], img_rotate.shape[:2])
    trans_points = np.round(trans_points)
    trans_points[:,0] -= img_rotate.shape[1]/2
    trans_points[:,1] -= img_rotate.shape[0]/2
    trans_points *= resize_scale
    trans_points[:,0] += img_resize.shape[1] / 2
    trans_points[:,1] += img_resize.shape[0] / 2
    trans_points = np.round(trans_points)

    img_crop = np.zeros((crop_size, crop_size, 3))
    crop_y = int(eyec2[1] - ec_y)
    crop_y_end = int(crop_y + crop_size - 1)
    crop_x = int(eyec2[0] - np.floor(crop_size/2))
    crop_x_end = int(crop_x + crop_size - 1)
    box = guard(np.array([crop_x, crop_x_end, crop_y, crop_y_end]), img_resize.shape)
    box = box.astype(np.int32)
    img_crop[box[2]-crop_y+1:box[3]-crop_y+1, box[0]-crop_x+1:box[1]-crop_x+1,:] = img_resize[box[2]:box[3],box[0]:box[1],:]
    trans_points[:,0] -= crop_x - 1
    trans_points[:,1] -= crop_y - 1
    trans_points = trans_points.astype(np.int32)
    cropped = img_crop / 255
    return img_crop, trans_points

if __name__ == '__main__':
    path = './other_data/'
    files = os.listdir(path)
    for f in files:
        if 'png' not in f:
            continue
        img = cv2.imread(path+f)
        with open(path+f.replace('png','5pt')) as ff:
            lines = ff.readlines()
        keypoints = np.zeros((5,2))
        for index, line in enumerate(lines):
            line = line.strip()
            keypoints[index,:] = line.split(',')
        img_cropped, trans_pt = align(img, keypoints)
        cv2.imwrite('./solved_data/'+f, img_cropped)
        keypoints = keypoints.astype(np.int32)
        with open('./solved_data/'+f.replace('png','5pt'),'a') as ff:
            for point in trans_pt:
                ff.write(str(point[0])+','+str(point[1])+'\n')
                #cv2.circle(img_cropped, (int(point[0]),int(point[1])),2,(255,255,0),3)
        #cv2.imwrite(f,img_cropped)

