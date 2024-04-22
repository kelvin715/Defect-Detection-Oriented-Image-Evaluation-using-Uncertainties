'''
this file is used to calculate the quality prediction of the dataset using three metrics, visibility, exposure, and distribution visibility.
'''
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def cal_visibility_based_on_chan_vese(I, label):
    """calculate visibility within each box, namely the ratio of foreground pixels in the box
    
    Args:
        I (_type_): original gray image with shape of (H, W)
        x (_type_): x centre
        y (_type_): y centre
        width (_type_): width
        height (_type_): height
    """
    image_width, image_height = I.shape
    
    if len(label) == 0:
        return None
    
    x, y, width, height = label[0], label[1], label[2], label[3]
    target = I[max(0, int((y - height / 2) * image_width)) : min(image_height, int((y + height / 2) * image_width)), max(0, int((x - width / 2) * image_height)) : min(image_width, int((x + width / 2) * image_height))]
    if np.size(target) == 0:
        return None
    cv = chan_vese(target, mu=0.05, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200,
                dt=0.5, init_level_set="checkerboard", extended_output=True)

    seg_mask = cv[0]
    
    # change cordinate of seg_mask to the original image
    y1, y2, x1, x2 = max(0, int((y - height / 2) * image_width)), min(image_height, int((y + height / 2) * image_width)), max(0, int((x - width / 2) * image_height)), min(image_width, int((x + width / 2) * image_height))
    seg_mask_original = np.zeros_like(I)
    seg_mask_original[y1:y2, x1:x2] = seg_mask
    
    
    x_bar = np.sum(target[seg_mask == True]) / np.prod(target.shape)
    
    seg_mask_original_inverse = np.ones_like(seg_mask_original) - seg_mask_original
    background_extended = (I * seg_mask_original_inverse)[max(0, int((y - height) * image_width)) : min(image_height, int((y + height) * image_width)), max(0, int((x - width) * image_height)) : min(image_width, int((x + width) * image_height))]
    
    
    
    x_bar_ = np.sum(background_extended) / np.prod(background_extended.shape)
    
    v_a = np.abs(x_bar - x_bar_) / np.max(I)

    return v_a

def cal_visibility(I, label):
    """calculate visibility within each box, namely the variance of the pixel value in the box
    
    Args:
        I (_type_): original gray image with shape of (H, W)
        x (_type_): x centre
        y (_type_): y centre
        width (_type_): width
        height (_type_): height
    """
    image_width, image_height = I.shape
    
    x, y, width, height = label[0], label[1], label[2], label[3]
    target = I[max(0, int((y - height / 2) * image_width)) : min(image_height, int((y + height / 2) * image_width)), max(0, int((x - width / 2) * image_height)) : min(image_width, int((x + width / 2) * image_height))]
    if np.max(target) == 0:
        return np.array([[0]], dtype=np.float32)
    else:
        return np.var(target / np.max(target), axis=(0, 1), ddof=0, keepdims=True, dtype=np.float32)    
    

def cal_visibility(I, label):
    """calculate visibility within each box, namely the variance of the pixel value in the box
    
    Args:
        I (_type_): original gray image with shape of (H, W)
        x (_type_): x centre
        y (_type_): y centre
        width (_type_): width
        height (_type_): height
    """
    image_width, image_height = I.shape
    
    x, y, width, height = label[0], label[1], label[2], label[3]
    target = I[max(0, int((y - height / 2) * image_width)) : min(image_height, int((y + height / 2) * image_width)), max(0, int((x - width / 2) * image_height)) : min(image_width, int((x + width / 2) * image_height))]
    
    # find the threshold based on binary segmentation
    
    
    if np.max(target) == 0:
        return np.array([[0]], dtype=np.float32)
    else:
        return np.var(target / np.max(target), axis=(0, 1), ddof=0, keepdims=True, dtype=np.float32)  
    

def cal_distribution_visibility(vis):
    """

    Args:
        vis (np.array): one-dimensioned vector of visibility of all defects over dataset
    Returns:
        _type_: _description_
    """
    return np.var(vis, keepdims=True, dtype=np.float32)


def cal_exposure(I, threshold:int=245):
    """calculate exposure
    Args:
        I (_type_): original gray image with shape of (H, W)
        threshold (_type_): threshold
    """
    return np.sum(I >= threshold) / np.prod(I.shape)



def cal_overexposure_and_overdarkness(I, high_threshold:int=240, low_threshold:int=20):
    """calculate overexposure and overdarkness
    Args:
        I (_type_): original gray image with shape of (H, W)
        high_threshold (_type_): high threshold
        low_threshold (_type_): low threshold
    """
    return (np.sum(I >= high_threshold) + np.sum(I <= low_threshold)) / np.prod(I.shape)


def cal_comprehensive_score(expo, distri, vis, alpha=1/3, beta=1/3, gama=1/3):
    """calculate comprehensive score
    Args:
        expo (_type_): exposure
        distri (_type_): distribution visibility
        vis (_type_): visibility
    """
    return alpha * vis + beta * (1 - distri) + gama * (1 - expo)



def quality_prediction_of_dataset(path):
    """quality prediction of the whole dataset
    Args:
        path (_type_): path of dataset
    """
    visibility_record = []
    exposure_record = []
    for file in os.listdir(os.path.join(path, "images")):
        prelix = file[:-4]
        image_path = os.path.join(path, "images", file)
        label_path = os.path.join(path, "labels", prelix+".txt")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        labels = np.loadtxt(label_path)
        
        if len(labels.shape) == 2:
            for label in labels:        
                visibility = cal_visibility_based_on_chan_vese(img, label[1:])
                if visibility is not None:
                    visibility_record.append(visibility)
        else:
            visibility = cal_visibility_based_on_chan_vese(img, labels[1:])
            if visibility is not None:
                visibility_record.append(visibility)
                    
        # exposure = cal_exposure(img) 
        exposure = cal_overexposure_and_overdarkness(img)
        exposure_record.append(exposure)
    
    visibility = np.mean(np.array(visibility_record), axis=0)
    exposure = np.mean(np.array(exposure_record), axis=0)
    distribution_visibility = np.var(np.array(visibility_record / np.max(visibility_record)), axis=0)
    
    score = cal_comprehensive_score(exposure, distribution_visibility, visibility)
    
    return score, visibility, exposure, distribution_visibility


def quality_prediction_of_dataset_save_csv(path, df):
    """1. calculate quality prediction based on visibility, exposure, and distribution of visibility 
       2. save result to dataframe
    Args:
        path (_type_): path of dataset
    """
    visibility_record = []
    exposure_record = []
    bar = tqdm(os.listdir(os.path.join(path, "images")), desc="quality prediction of dataset", total=len(os.listdir(os.path.join(path, "images"))))
    for file in bar:
        prelix = file[:-4]
        image_path = os.path.join(path, "images", file)
        label_path = os.path.join(path, "labels", prelix+".txt")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        labels = np.loadtxt(label_path)
        
        if len(labels.shape) == 2:
            for label in labels:        
                visibility = cal_visibility_based_on_chan_vese(img, label[1:])
                if visibility is not None:
                    visibility_record.append(visibility)
        else:
            visibility = cal_visibility_based_on_chan_vese(img, labels[1:])
            if visibility is not None:
                visibility_record.append(visibility)
                    
        # exposure = cal_exposure(img)
        exposure = cal_overexposure_and_overdarkness(img)
        exposure_record.append(exposure)
        
        # add to df
        df.loc[len(df)] = [path, file, visibility, exposure]
    
    visibility = np.mean(np.array(visibility_record), axis=0)
    exposure = np.mean(np.array(exposure_record), axis=0)
    distribution_visibility = np.var(np.array(visibility_record / np.max(visibility_record)), axis=0)
    
    score = cal_comprehensive_score(exposure, distribution_visibility, visibility)
    

    return score, visibility, exposure, distribution_visibility, df


if __name__ == '__main__':
    
    # columns = ['dataset', 'score', 'visibility', 'exposure', 'distribution_visibility']
    # dir = ['test']
    # for j in dir:
    #     # for i in [60,120,180,240,300]:
    #     #     # save to csv
    #     #     score, vis, expo, distr = quality_prediction_of_dataset(f'/Data4/student_zhihan_data/data/GC10-DET_BilateralBlur_{i}/{j}')
    #     #     print(['GC10-DET_BilateralBlur_'+str(i), score, vis, expo, distr])
    #     #     df = pandas.DataFrame([['GC10-DET_BilateralBlur_'+str(i), score, vis, expo, distr]], columns=columns)
    #     #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
    
    #     # for i in ['0.05:0.1', '0.1:0.15000000000000002', '0.15000000000000002:0.2', '0.2:0.25', '0.25:0.3']: 
    #     for i in ["0.5", "1.5", "2.0", "2.5", "3"]:  
    #         # save to csv
    #         score, vis, expo, distr = quality_prediction_of_dataset(f'/Data4/student_zhihan_data/data/GC10-DET_Sharpening_{i}/{j}')
    #         print(['GC10-DET_Sharpening_'+str(i), score, vis, expo, distr])
    #         df = pandas.DataFrame([['GGC10-DET_Sharpening_'+str(i), score, vis, expo, distr]], columns=columns)
    #         df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
            
    columns = ['dataset', 'score', 'visibility', 'exposure', 'distribution_visibility']
    dir = ['test']
    for j in dir:
        # for i in [60,120,180,240,300]:
        #     # save to csv
        #     score, vis, expo, distr = quality_prediction_of_dataset(f'/Data4/student_zhihan_data/data/GC10-DET_BilateralBlur_{i}/{j}')
        #     print(['GC10-DET_BilateralBlur_'+str(i), score, vis, expo, distr])
        #     df = pandas.DataFrame([['GC10-DET_BilateralBlur_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
    
        # for i in ['0.05:0.1', '0.1:0.15000000000000002', '0.15000000000000002:0.2', '0.2:0.25', '0.25:0.3']: 
        # for i in ["0.5", "1.5", "2.0", "2.5", "3"]:  
        #     # save to csv
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_Sharpening_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_Sharpening_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_Sharpening_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_Sharpening_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
        
        # for i in [15, 29, 43, 57, 71]:  
        #     # save to csv
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_MedianBlur_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_MedianBlur_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_MedianBlur_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_MedianBlur_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)   
        

        # for i in [60, 120, 180, 240, 300]:
        #     # save to csv
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_BilateralBlur_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_BilateralBlur_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_BilateralBlur_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_BilateralBlur_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
        
        # for i in [-150, -100, -50, -30, -20, -15, -10, 10, 20, 30, 50, 60, 70, 90, 110]:
        #     # save to csv
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_brightness_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_brightness_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_brightness_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_brightness_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
            
        # for i in [5, 10, 15, 20, 25]:
        #     # save to csv
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_Sharpen_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_Sharpen_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_Sharpen_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_Sharpen_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
        
        # for i in ['0.0:0.05', '0.1:0.15', '0.2:0.25', '0.05:0.1', '0.25:0.3', '0.15:0.2']:
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_Transform_Scale_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_Transform_Scale_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_Transform_Scale_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_Transform_Scale_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)
            
        
        # df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        # score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET/train', df_img)
        # df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_train'+'.csv', mode='a', header=False, index=False)
        # print(['GC10-DET'+str(i), score, vis, expo, distr])
        # df = pd.DataFrame([['GGC10-DET'+str(i), score, vis, expo, distr]], columns=columns)
        # df.to_csv(j+'demo.csv', mode='a', header=False, index=False) 
        
        # for i in [3, 5, 7, 9, 11, 13, 15, 17]:
        #     # save to csv
        #     df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        #     score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET_GaussianBlur_{i}/{j}', df_img)
        #     df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_GaussianBlur_{i}'+'.csv', mode='a', header=False, index=False)
        #     print(['GC10-DET_GaussianBlur_'+str(i), score, vis, expo, distr])
        #     df = pd.DataFrame([['GGC10-DET_GaussianBlur_'+str(i), score, vis, expo, distr]], columns=columns)
        #     df.to_csv(j+'demo.csv', mode='a', header=False, index=False)  
        

        df_img = pd.DataFrame(columns=['dataset', 'img_name', 'visibility', 'exposure'])
        score, vis, expo, distr, df_img = quality_prediction_of_dataset_save_csv(f'/Data4/student_zhihan_data/data/GC10-DET/train', df_img)
        df_img.to_csv(f'/Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/Proposed_Score_Record/GC10-DET_train'+'.csv', mode='a', header=False, index=False)
        print(['GC10-DET_train', score, vis, expo, distr])
        df = pd.DataFrame([['GGC10-DET_train', score, vis, expo, distr]], columns=columns)
        df.to_csv(j+'demo.csv', mode='a', header=False, index=False)  