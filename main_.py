from pathlib import Path
from configs.config import MainConfig
from confz import BaseConfig, FileSource
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.utils import load_detector, load_classificator, open_mapping, extract_crops
import pandas as pd
from itertools import repeat
from datetime import datetime, timedelta
from exif import Image

main_config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
device = main_config.device

detector_config = main_config.detector
classificator_config = main_config.classificator

# Load models
detector = load_detector(detector_config)#.to(device)
classificator = load_classificator(classificator_config)#.to(device)
if torch.cuda.is_available():
    detector.to(device)
    classificator.to(device)


def get_path_to_time(list_path):
    images = []
    time_set = []
    for path in list_path:
        with open(path, "rb") as palm_file:
            palm_image = Image(palm_file)
        images.append(palm_image)
    for index, image in enumerate(images):
        time = str(image.datetime_original) + " " + str(image.get('offset_time', ''))
        time = time.replace(":", "-", 2)
        time = time[:-1]
        time_set.append(time)
    return time_set
def infer(list_path,ii, path_csv):
    # Load main config

    # Load imgs from source dir
    pathes_to_imgs = list_path

    # Load mapping for classification task
    mapping = open_mapping(path_mapping=main_config.mapping)

    # Separate main config

    # Inference
    if len(pathes_to_imgs):

        list_predictions = []

        num_packages_det = np.ceil(len(pathes_to_imgs) / detector_config.batch_size).astype(np.int32)
        with torch.no_grad():
            for i in tqdm(range(num_packages_det), colour="green"):
                # Inference detector
                batch_images_det = pathes_to_imgs[detector_config.batch_size * i:
                                                        detector_config.batch_size * (1 + i)]
                try:                                  
                    results_det = detector(batch_images_det,
                                        iou=detector_config.iou,
                                        conf=detector_config.conf,
                                        imgsz=detector_config.imgsz,
                                        verbose=False,
                                        device=device)
                    if len(results_det) > 0:
                        # Extract crop by bboxes
                        dict_crops = extract_crops(results_det, config=classificator_config)

                        # Inference classificator
                        for img_name, batch_images_cls in dict_crops.items():
                            # if len(batch_images_cls) > classificator_config.batch_size:
                            num_packages_cls = np.ceil(len(batch_images_cls) / classificator_config.batch_size).astype(
                                np.int32)
                            for j in range(num_packages_cls):
                                batch_images_cls = batch_images_cls[classificator_config.batch_size * j:
                                                                    classificator_config.batch_size * (1 + j)]
                                logits = classificator(batch_images_cls.to(device))
                                
                                
                                probabilities = torch.nn.functional.softmax(logits, dim=1)
                                top_p, top_class_idx = probabilities.topk(1, dim=1)

                                # Locate torch Tensors to cpu and convert to numpy
                                top_p = top_p.cpu().numpy().ravel()
                                top_class_idx = top_class_idx.cpu().numpy().ravel()
                                
                                class_names = [mapping[top_class_idx[idx]] for idx, _ in enumerate(batch_images_cls)]

                                list_predictions.extend([[name, cls, prob] for name, cls, prob in
                                                        zip(repeat(img_name, len(class_names)), class_names, top_p)])
                except:
                    pass
        # Create Dataframe with predictions
        table = pd.DataFrame(list_predictions, columns=["image_name", "class_name", "confidence"])
        table = table[table.class_name != 'Empty']
        #table.to_csv("table.csv", index=False) # Раскомментируйте, если хотите увидеть результаты предсказания
        # нейронной сети по каждому найденному объекту

        agg_functions = {
            'class_name': ['count'],
            "confidence": ["mean"]
        }
        groupped = table.groupby(['image_name', "class_name"]).agg(agg_functions)
        img_names = groupped.index.get_level_values("image_name").unique()
        #groupped.to_csv('group.csv', index=False)
        final_res = []

        for img_name in img_names:
            groupped_per_img = groupped.query(f"image_name == '{img_name}'")
            max_num_objects = groupped_per_img["class_name", "count"].max()
            # max_confidence = groupped_per_img["class_name", "confidence"].max()
            statistic_by_max_objects = groupped_per_img[groupped_per_img["class_name", "count"] == max_num_objects]

            if len(statistic_by_max_objects) > 1:
                # statistic_by_max_mean_conf = statistic_by_max_objects.reset_index().max().values
                statistic_by_max_mean_conf = statistic_by_max_objects.loc[[statistic_by_max_objects["confidence", "mean"].idxmax()]]
                final_res.extend(statistic_by_max_mean_conf.reset_index().values)
            else:
                final_res.extend(statistic_by_max_objects.reset_index().values)
        # groupped.to_csv("table_agg.csv", index=True) # Раскомментируйте, если хотите увидеть результаты аггрегации

        final_table = pd.DataFrame(final_res, columns=["image_name", "class_name", "count", "confidence"])
        final_table.to_csv(os.path.join(path_csv, f"table_final_{ii}.csv"), index=False)



def get_final_csv(sample_submission):
    
    sample_submission = sample_submission[["name_folder", "class_name", "time", "count", "confidence",  "image_name"]]
    sample_submission.loc[sample_submission['confidence'] >= 0.5, 'True_class'] = sample_submission['class_name']
    sample_submission['time'] = pd.to_datetime(sample_submission['time'])

    for k in tqdm(range(20)):
        for i in range(1, len(sample_submission)):
            if sample_submission.loc[i, 'confidence'] < 0.5 and pd.isna(sample_submission.loc[i, 'True_class']):
                time_diff = sample_submission.loc[i, 'time'] - sample_submission.loc[i-1, 'time']
                if time_diff < timedelta(minutes=30) and sample_submission.loc[i, 'name_folder'] == sample_submission.loc[i-1, 'name_folder'] and not pd.isna(sample_submission.loc[i-1, 'True_class']):
                    sample_submission.loc[i, 'True_class'] = sample_submission.loc[i-1, 'True_class']

        for i in range(len(sample_submission) - 1):
            if sample_submission.loc[i, 'confidence'] < 0.5 and pd.isna(sample_submission.loc[i, 'True_class']):
                time_diff = sample_submission.loc[i, 'time'] - sample_submission.loc[i+1, 'time']
                if time_diff < timedelta(minutes=30) and sample_submission.loc[i, 'name_folder'] == sample_submission.loc[i+1, 'name_folder'] and not pd.isna(sample_submission.loc[i+1, 'True_class']):
                    sample_submission.loc[i, 'True_class'] = sample_submission.loc[i+1, 'True_class']

    sample_submission = sample_submission.dropna(subset=['True_class'])
    sample_submission = sample_submission.reset_index(drop=True)

    # Сортируем DataFrame по времени, чтобы гарантировать корректную последовательность
    sample_submission.sort_values(by='time', inplace=True)

    # Создаем новый DataFrame для итоговых данных
    final_df = pd.DataFrame(columns=['name_folder', 'class', 'data_regestration_start', 'data_regestration_end', 'count'])

    # Инициализация переменных
    start_index = 0
    max_count = 0
    start_time = sample_submission.loc[0, 'time']
    start_class = sample_submission.loc[0, 'True_class']

    for i in tqdm(range(1, len(sample_submission))):
        current_time = sample_submission.loc[i, 'time']
        current_class = sample_submission.loc[i, 'True_class']
        current_count = sample_submission.loc[i, 'count']
        time_diff = current_time - start_time

        # Проверяем условие завершения регистрации
        if current_class != start_class or time_diff > timedelta(minutes=30):
            # Добавляем запись в итоговый DataFrame
            final_df = pd.concat([final_df, pd.DataFrame({
                'name_folder': [sample_submission.loc[start_index, 'name_folder']],
                'class': [start_class],
                'data_regestration_start': [start_time],
                'data_regestration_end': [sample_submission.loc[i-1, 'time']],
                'count': [max_count]
            })], ignore_index=True)
            # Обновляем стартовые значения
            start_index = i
            start_time = current_time
            start_class = current_class
            max_count = current_count
        else:
            # Обновляем максимальный count
            if current_count > max_count:
                max_count = current_count

    # Добавляем последнюю запись
    final_df = pd.concat([final_df, pd.DataFrame({
        'name_folder': [sample_submission.loc[start_index, 'name_folder']],
        'class': [start_class],
        'data_regestration_start': [start_time],
        'data_regestration_end': [sample_submission.loc[len(sample_submission)-1, 'time']],
        'count': [max_count]
    })], ignore_index=True)

    final_df['data_regestration_start'] = pd.to_datetime(final_df['data_regestration_start'])
    final_df['data_regestration_end'] = pd.to_datetime(final_df['data_regestration_end'])

    for i in range(1, len(final_df)-1):
        time_differ = final_df.loc[i, 'data_regestration_end'] - final_df.loc[i, 'data_regestration_start']
        time_differ_previos = final_df.loc[i, 'data_regestration_start'] - final_df.loc[i-1, 'data_regestration_end']
        time_differ_next = final_df.loc[i+1, 'data_regestration_start'] - final_df.loc[i, 'data_regestration_end']
        if time_differ < timedelta(seconds=1):
            if time_differ_previos > time_differ_next:
                final_df.loc[i, "class"] = final_df.loc[i+1, "class"]
            else:
                final_df.loc[i, "class"] = final_df.loc[i-1, "class"]

    final_df = final_df.sort_values(by=['name_folder', "data_regestration_start", "data_regestration_end"], ascending=True)

    return final_df


def main(path):
    try:
        os.mkdir(os.path.join(path, 'temporary'))
    except:
        pass
    for id in os.listdir(path):
        try:
            l = [os.path.join(path, id, i) for i in os.listdir(os.path.join(path, id))]
            infer(l,id,os.path.join(path, 'temporary'))
        except:
            pass
    pred = pd.DataFrame()

    for i in os.listdir(os.path.join(path, 'temporary')):
        
        id = i.split('.')[0]
        id = id.split('_')[2]
        prom = pd.read_csv(os.path.join(path, 'temporary', i))
        prom['image_name'] = [f'{id}/{i}' for i in prom['image_name'].to_list()]
        pred = pd.concat([pred,prom],ignore_index=True)
    
    pred['time'] = get_path_to_time([os.path.join(path, i) for i in pred['image_name'].to_list()])
    pred['name_folder'] = [i.split('/')[0] for i in pred['image_name']]
    pred.to_csv('pred.csv')
    final = get_final_csv(pred)
    final.to_csv('finall.csv', index=False)


