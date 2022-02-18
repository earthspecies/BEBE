#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import h5py
import tqdm
import scipy
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Returns cropped image using rect tuple
def crop_and_flip(rect, src, points, ref_index):
    #Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    #Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)

    #shift DLC points
    x_diff = center[0] - size[0]//2
    y_diff = center[1] - size[1]//2

    dlc_points_shifted = []

    for i in points:
        point=cv.transform(np.array([[[i[0], i[1]]]]),M)[0][0]

        point[0] -= x_diff
        point[1] -= y_diff

        dlc_points_shifted.append(point)

    # Perform rotation on src image
    dst = cv.warpAffine(src.astype('float32'), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)

    #check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = ((size[0]//2,size[0]//2),size,180)
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        #Get rotation matrix
        M = cv.getRotationMatrix2D(center, theta, 1)


        #shift DLC points
        x_diff = center[0] - size[0]//2
        y_diff = center[1] - size[1]//2

        points = dlc_points_shifted
        dlc_points_shifted = []

        for i in points:
            point=cv.transform(np.array([[[i[0], i[1]]]]),M)[0][0]

            point[0] -= x_diff
            point[1] -= y_diff

            dlc_points_shifted.append(point)

        # Perform rotation on src image
        dst = cv.warpAffine(out.astype('float32'), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)

    return out, dlc_points_shifted


def background(path_to_file,filename,file_format='.mp4',num_frames=1000):
    """
    Compute background image from fixed camera
    """

    capture = cv.VideoCapture(os.path.join(path_to_file,"videos",filename+file_format))

    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,"videos",filename+file_format)))

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()

    height, width, _ = frame.shape
    frames = np.zeros((height,width,num_frames))

    for i in tqdm.tqdm(range(num_frames), disable=not True, desc='Compute background image for video %s' %filename):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1,rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[...,i] = gray

    print('Finishing up!')
    medFrame = np.median(frames,2)
    background = scipy.ndimage.median_filter(medFrame, (5,5))

    np.save(os.path.join(path_to_file,"videos",filename+'-background.npy'),background)

    capture.release()
    return background


def get_rotation_matrix(adjacent, opposite, crop_size=(300, 300)):

    tan_alpha = np.abs(opposite) / np.abs(adjacent)
    alpha = np.arctan(tan_alpha)
    alpha = np.rad2deg(alpha)

    if adjacent < 0 and opposite > 0:
        alpha = 180-alpha

    if adjacent  < 0 and opposite < 0:
        alpha = -(180-alpha)

    if adjacent > 0 and opposite < 0:
        alpha = -alpha

    rot_mat = cv.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2),alpha, 1)

    return rot_mat


#Helper function to return indexes of nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


#Interpolates all nan values of given array
def interpol(arr):

    y = np.transpose(arr)

    nans, x = nan_helper(y[0])
    y[0][nans]= np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans]= np.interp(x(nans), x(~nans), y[1][~nans])

    arr = np.transpose(y)

    return arr


def get_animal_frames(cfg, filename, pose_ref_index, start, length, subtract_background, file_format='.mp4', crop_size=(300, 300)):
    path_to_file = cfg['project_path']
    time_window = cfg['time_window']
    lag = int(time_window / 2)
    #read out data
    data = pd.read_csv(os.path.join(path_to_file,"videos","pose_estimation",filename+'.csv'), skiprows = 2)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:,1:]

    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1]/3)):
        pose_list.append(data_mat[:,i*3:(i+1)*3])

    #list of reference coordinate indices for alignment
    #0: snout, 1: forehand_left, 2: forehand_right,
    #3: hindleft, 4: hindright, 5: tail

    pose_ref_index = pose_ref_index

    #list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index

    # compute background
    if subtract_background == True:
        try:
            print("Loading background image ...")
            bg = np.load(os.path.join(path_to_file,"videos",filename+'-background.npy'))
        except:
            print("Can't find background image... Calculate background image...")
            bg = background(path_to_file,filename, file_format)

    images = []
    points = []
     
    for i in pose_list:
        for j in i:
            if j[2] <= 0.8:
                j[0],j[1] = np.nan, np.nan


    for i in pose_list:
        i = interpol(i)

    capture = cv.VideoCapture(os.path.join(path_to_file,"videos",filename+file_format))
    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,"videos",filename++file_format)))

    for idx in tqdm.tqdm(range(length), disable=not True, desc='Align frames'):
        try:
            capture.set(1,idx+start+lag)
            ret, frame = capture.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if subtract_background == True:
                frame = frame - bg
                frame[frame <= 0] = 0
        except:
            print("Couldn't find a frame in capture.read(). #Frame: %d" %idx+start+lag)
            continue

       #Read coordinates and add border
        pose_list_bordered = []

        for i in pose_list:
            pose_list_bordered.append((int(i[idx+start+lag][0]+crop_size[0]),int(i[idx+start+lag][1]+crop_size[1])))

        img = cv.copyMakeBorder(frame, crop_size[1], crop_size[1], crop_size[0], crop_size[0], cv.BORDER_CONSTANT, 0)

        punkte = []
        for i in pose_ref_index:
            coord = []
            coord.append(pose_list_bordered[i][0])
            coord.append(pose_list_bordered[i][1])
            punkte.append(coord)
        punkte = [punkte]
        punkte = np.asarray(punkte)

        #calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)

        #change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)

        center, size, theta = rect

        #crop image
        out, shifted_points = crop_and_flip(rect, img,pose_list_bordered,pose_flip_ref)

        images.append(out)
        points.append(shifted_points)

    capture.release()
    return images
