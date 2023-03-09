import os, os.path
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # Return the result
    return result_img


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    # Make sure that the matches are good
    verify_ratio = 0.8  # Source: stackoverflow
    verified_matches = []
    for m1, m2 in matches:
        # Add to array only if it's a good match
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    # Mimnum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        messagebox.showinfo("Eror", "Error: Not enough matches")
        exit()


# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

# import and read folder from input of UI "Select File from Directory"
def getImg():
    global imgs,img
    path = filedialog.askdirectory()
    # Get input set of images

    imgs = []
    valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(cv2.imread(os.path.join(path, f)))
    # Equalize histogram
    img = []
    for i in imgs:
        img.append(equalize_histogram_color(i))

def panorama():
    # Use SIFT to find keypoints and return homography matrix
    result_image = img[0]
    for i in range(1, len(img)):
        M = get_sift_homography(result_image, img[i])

        # Stitch the images together using homography matrix
        result_image = get_stitched_image(img[i], result_image, M)

    # Write the result to the same directory
    messagebox.showinfo("Save ImagePanorama", "Select Folder and type Name to save result image")
    Name = filedialog.asksaveasfilename(defaultextension='.png')
    cv2.imwrite(Name, result_image)
    if result_image.shape[1]<1500:
        cv2.imshow('result',result_image)
    else:
        messagebox.showinfo("Panorama Done", "Please Check The Saved Image Because Result Image Too Large \n"
                                             "Tick OK To Continue")

def destroy():
    cv2.destroyAllWindows()

# Main function definition
def main():
    root = tk.Tk()
    canvas1 = tk.Canvas(root, width=300, height=250, bg='lightsteelblue2', relief='raised')
    canvas1.pack()
    label1 = tk.Label(root, text='File Panorama Tool', bg='lightsteelblue2')
    label1.config(font=('helvetica', 20))
    canvas1.create_window(150, 60, window=label1)
    Select_file = tk.Button(text="      Select File from Directory     ", command=getImg, bg='royalblue', fg='white',
                            font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 130, window=Select_file)
    Panorama = tk.Button(text='Panorama', command=panorama, bg='royalblue', fg='white',
                         font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 180, window=Panorama)
    tk.Button(root, text="Quit", command=root.destroy).pack()
    root.mainloop()


# Call main function
if __name__ == '__main__':
    main()