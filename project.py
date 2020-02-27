import cv2
from cv2 import  xfeatures2d
import numpy as np

def cv_show(name,img):
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,800,400)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#第一步 构造sift 求解出sift特征点与sift特征向量
def getKpsAndDes(img):
    sift = xfeatures2d.SIFT_create()
    kps, des = sift.detectAndCompute(img, None)  # get feature points and descriptors
    return kps, des

def drawKPimg(img,kps,imgname):                               #画出标记了关键特征点的图像
    ret = cv2.drawKeypoints(img, kps, img, color=(0, 0, 255))
    cv_show('ret',ret)

def getGoodMatches(desA,desB):
    bf = cv2.BFMatcher()                            #第二步 构造BFmatcher 用于match
    matches = bf.knnMatch(desA,desB,k=2)            #第三步 获得匹配结果并按照距离排序

    #flann = cv2.FlannBasedMatcher()
    #matches = flann.knnMatch(desA,desB,k=2)
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good = sorted(good,key = lambda x :x.distance)
    return good

#第四步 画出匹配结果
def drawmatch(imgA,kpsA,imgB,kpsB,good_points_list,matchesMask):
    drawPrams = dict(matchColor=(0, 255, 0),
                         singlePointColor=(255, 0, 0),
                         matchesMask=matchesMask,
                         flags=2)                               #flags = 0画出所有特征点
                                                                # 只画出匹配的特征点
    ret = cv2.drawMatches(imgA,kpsA,imgB,kpsB,good_points_list,None,**drawPrams)

    cv_show('ret',ret)



MIN_MATCH_COUNT = 10           #设置最低特征点匹配数量为10

def main():
    imgA = cv2.imread('s.jpg', 0)

    imgB = cv2.imread('b.jpg', 0)

    kpsA, desA = getKpsAndDes(imgA)  # get feature points and descriptors
    kpsB, desB = getKpsAndDes(imgB)

    good = getGoodMatches(desA,desB)

    if len(good)>MIN_MATCH_COUNT:                       #若匹配特征点数量大于最低数量

        print('length of good',len(good))
        #drawmatch(imgA, kpsA, imgB, kpsB, good)

        # 小图的特征点坐标
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # 大图的特征点坐标
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h,w = (imgA.shape[0],imgA.shape[1])

        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)

        matchMask = mask.ravel().tolist()
        print('单应性矩阵：',H.shape)
        pts = np.float32([[0, 0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        cv2.polylines(imgB,[np.int32(dst)],True,0,2,cv2.LINE_AA)
    else:
        print("Not enough matches are found -%d%d"%(len(good),MIN_MATCH_COUNT))
        matchMask = None
    drawmatch(imgA,kpsA,imgB,kpsB,good,matchMask)

main()







