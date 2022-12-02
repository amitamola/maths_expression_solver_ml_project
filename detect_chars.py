import cv2
import numpy as np

def detect_text(img):
    #Create MSER object
    mser = cv2.MSER_create()

    #Converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()

    #detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    #this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)
    
    return vis, text_only


def __get_coordinates_rows(img, nonzero_cols):
    new_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final = []
    
    for x1, x2 in list(zip(nonzero_cols[::2], nonzero_cols[1::2])):
        crop_to_work = new_out[:, x1:x2]
        new_arr = np.where(np.sum(crop_to_work, axis=1)!=0, 1, 0)
        
        imp2 = []
        for i in range(len(new_arr)-1):
            if (new_arr[i]==0 and new_arr[i+1]==1):
                imp2.append(i)
            elif (new_arr[i]==1 and new_arr[i+1]==0):
                imp2.append(i+1)
            else:
                continue
                
        final.append((x1,x2, imp2[0], imp2[-1]))
        
    return final
        

def get_coordinates_cols(out_text):
    new_out = cv2.cvtColor(out_text, cv2.COLOR_BGR2GRAY)
    arr = np.where(np.sum(new_out, axis=0)!=0, 1, 0)

    imp = []
    for i in range(len(arr)-1):
        if (arr[i]==0 and arr[i+1]==1):
            imp.append(i)
        elif (arr[i]==1 and arr[i+1]==0):
            imp.append(i+1)
        else:
            continue

    if len(imp)%2!=0:
        imp.append(new_out.shape[1])
    
    rows=__get_coordinates_rows(out_text, imp)
    
    return rows