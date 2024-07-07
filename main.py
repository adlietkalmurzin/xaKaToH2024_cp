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
import io
import os
import time
import xlsxwriter as xs
from time import sleep
import torch
import pandas as pd
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import sys

exs = ['png', 'jpg', 'jpeg']
ui = '''<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>MainWindow</class>
 <widget class="QMainWindow" name="MainWindow">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>793</width>
    <height>599</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>MainWindow</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <widget class="QPushButton" name="openButton">
    <property name="geometry">
     <rect>
      <x>240</x>
      <y>220</y>
      <width>271</width>
      <height>51</height>
     </rect>
    </property>
    <property name="text">
     <string>Открыть папку</string>
    </property>
   </widget>
   <widget class="QLabel" name="label">
    <property name="geometry">
     <rect>
      <x>130</x>
      <y>370</y>
      <width>661</width>
      <height>51</height>
     </rect>
    </property>
    <property name="text">
     <string>Папка:</string>
    </property>
   </widget>
   <widget class="QPushButton" name="predictButton">
    <property name="enabled">
     <bool>false</bool>
    </property>
    <property name="geometry">
     <rect>
      <x>330</x>
      <y>460</y>
      <width>89</width>
      <height>25</height>
     </rect>
    </property>
    <property name="text">
     <string>Предикт</string>
    </property>
   </widget>
   <widget class="QProgressBar" name="progressBar">
    <property name="enabled">
     <bool>false</bool>
    </property>
    <property name="geometry">
     <rect>
      <x>200</x>
      <y>510</y>
      <width>361</width>
      <height>23</height>
     </rect>
    </property>
    <property name="value">
     <number>0</number>
    </property>
   </widget>
  </widget>
  <widget class="QMenuBar" name="menubar">
   <property name="geometry">
    <rect>
     <x>0</x>
     <y>0</y>
     <width>793</width>
     <height>22</height>
    </rect>
   </property>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections/>
</ui>'''

main_config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
device = main_config.device

detector_config = main_config.detector
classificator_config = main_config.classificator

# Load models
detector = load_detector(detector_config)  # .to(device)
classificator = load_classificator(classificator_config)  # .to(device)
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


def infer(list_path, ii, path_csv, thread):
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
                    thread.finish_signal.emit({'index': '+'})
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
        # table.to_csv("table.csv", index=False) # Раскомментируйте, если хотите увидеть результаты предсказания
        # нейронной сети по каждому найденному объекту

        agg_functions = {
            'class_name': ['count'],
            "confidence": ["mean"]
        }
        groupped = table.groupby(['image_name', "class_name"]).agg(agg_functions)
        img_names = groupped.index.get_level_values("image_name").unique()
        # groupped.to_csv('group.csv', index=False)
        final_res = []

        for img_name in img_names:
            groupped_per_img = groupped.query(f"image_name == '{img_name}'")
            max_num_objects = groupped_per_img["class_name", "count"].max()
            # max_confidence = groupped_per_img["class_name", "confidence"].max()
            statistic_by_max_objects = groupped_per_img[groupped_per_img["class_name", "count"] == max_num_objects]

            if len(statistic_by_max_objects) > 1:
                # statistic_by_max_mean_conf = statistic_by_max_objects.reset_index().max().values
                statistic_by_max_mean_conf = statistic_by_max_objects.loc[
                    [statistic_by_max_objects["confidence", "mean"].idxmax()]]
                final_res.extend(statistic_by_max_mean_conf.reset_index().values)
            else:
                final_res.extend(statistic_by_max_objects.reset_index().values)
        # groupped.to_csv("table_agg.csv", index=True) # Раскомментируйте, если хотите увидеть результаты аггрегации

        final_table = pd.DataFrame(final_res, columns=["image_name", "class_name", "count", "confidence"])
        final_table.to_csv(os.path.join(path_csv, f"table_final_{ii}.csv"), index=False)


def get_final_csv(sample_submission):
    sample_submission = sample_submission[["name_folder", "class_name", "time", "count", "confidence", "image_name"]]
    sample_submission.loc[sample_submission['confidence'] >= 0.5, 'True_class'] = sample_submission['class_name']
    sample_submission['time'] = pd.to_datetime(sample_submission['time'])

    for k in tqdm(range(20)):
        for i in range(1, len(sample_submission)):
            if sample_submission.loc[i, 'confidence'] < 0.5 and pd.isna(sample_submission.loc[i, 'True_class']):
                time_diff = sample_submission.loc[i, 'time'] - sample_submission.loc[i - 1, 'time']
                if time_diff < timedelta(minutes=30) and sample_submission.loc[i, 'name_folder'] == \
                        sample_submission.loc[i - 1, 'name_folder'] and not pd.isna(
                    sample_submission.loc[i - 1, 'True_class']):
                    sample_submission.loc[i, 'True_class'] = sample_submission.loc[i - 1, 'True_class']

        for i in range(len(sample_submission) - 1):
            if sample_submission.loc[i, 'confidence'] < 0.5 and pd.isna(sample_submission.loc[i, 'True_class']):
                time_diff = sample_submission.loc[i, 'time'] - sample_submission.loc[i + 1, 'time']
                if time_diff < timedelta(minutes=30) and sample_submission.loc[i, 'name_folder'] == \
                        sample_submission.loc[i + 1, 'name_folder'] and not pd.isna(
                    sample_submission.loc[i + 1, 'True_class']):
                    sample_submission.loc[i, 'True_class'] = sample_submission.loc[i + 1, 'True_class']

    sample_submission = sample_submission.dropna(subset=['True_class'])
    sample_submission = sample_submission.reset_index(drop=True)

    # Сортируем DataFrame по времени, чтобы гарантировать корректную последовательность
    sample_submission.sort_values(by='time', inplace=True)

    # Создаем новый DataFrame для итоговых данных
    final_df = pd.DataFrame(
        columns=['name_folder', 'class', 'data_regestration_start', 'data_regestration_end', 'count'])

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
                'data_regestration_end': [sample_submission.loc[i - 1, 'time']],
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
        'data_regestration_end': [sample_submission.loc[len(sample_submission) - 1, 'time']],
        'count': [max_count]
    })], ignore_index=True)

    final_df['data_regestration_start'] = pd.to_datetime(final_df['data_regestration_start'])
    final_df['data_regestration_end'] = pd.to_datetime(final_df['data_regestration_end'])

    for i in range(1, len(final_df) - 1):
        time_differ = final_df.loc[i, 'data_regestration_end'] - final_df.loc[i, 'data_regestration_start']
        time_differ_previos = final_df.loc[i, 'data_regestration_start'] - final_df.loc[i - 1, 'data_regestration_end']
        time_differ_next = final_df.loc[i + 1, 'data_regestration_start'] - final_df.loc[i, 'data_regestration_end']
        if time_differ < timedelta(seconds=1):
            if time_differ_previos > time_differ_next:
                final_df.loc[i, "class"] = final_df.loc[i + 1, "class"]
            else:
                final_df.loc[i, "class"] = final_df.loc[i - 1, "class"]

    final_df = final_df.sort_values(by=['name_folder', "data_regestration_start", "data_regestration_end"],
                                    ascending=True)

    return final_df


def main(path):
    try:
        os.mkdir(os.path.join(path, 'temporary'))
    except:
        pass
    for id in os.listdir(path):
        try:
            l = [os.path.join(path, id, i) for i in os.listdir(os.path.join(path, id))]
            infer(l, id, os.path.join(path, 'temporary'))
        except:
            pass
    pred = pd.DataFrame()

    for i in os.listdir(os.path.join(path, 'temporary')):
        id = i.split('.')[0]
        id = id.split('_')[2]
        prom = pd.read_csv(os.path.join(path, 'temporary', i))
        prom['image_name'] = [f'{id}/{i}' for i in prom['image_name'].to_list()]
        pred = pd.concat([pred, prom], ignore_index=True)

    pred['time'] = get_path_to_time([os.path.join(path, i) for i in pred['image_name'].to_list()])
    pred['name_folder'] = [i.split('/')[0] for i in pred['image_name']]
    pred.to_csv('pred.csv')
    final = get_final_csv(pred)
    final.to_csv('finall.csv', index=False)


def predict(filename):
    return 'Медведь'


def get_images_len(folder):
    res = 0
    for i in os.listdir(folder):
        if i.split('.')[-1].lower() in exs:
            res += 1
    return res


class Worker(QObject):
    finish_signal = pyqtSignal(dict)

    def __init__(self, selectedFolder):
        super().__init__()
        self.selectedFolder = selectedFolder

    def start(self):
        # try:
        #     header = ('Название файла', 'Класс')
        #     xlsx = open(os.path.join(self.selectedFolder, 'results.xlsx'), 'wb')
        #     workbook = xs.Workbook(xlsx)
        #     worksheet = workbook.add_worksheet()
        #     for i, row in enumerate(header):
        #         worksheet.set_column(i, i, 35)
        #         worksheet.write(0, i, row)
        #     images = get_images(self.selectedFolder)
        #     for i, photo in enumerate(images):
        #         filename = os.path.split(photo)[-1]
        #         class_photo = predict(photo)
        #         row = (filename, class_photo)
        #         for j, data in enumerate(row):
        #             worksheet.write(i + 1, j, data)
        #         self.finish_signal.emit({'index': i})
        #     workbook.close()
        #     xlsx.close()
        #     self.finish_signal.emit({'index': i + 1})
        #     self.finish_signal.emit({'finish': True})
        # except Exception as e:
        #     print(f"Error: {e}")
        #     self.finish_signal.emit({'finish': True, 'error': e})
        path = self.selectedFolder
        try:
            os.mkdir(os.path.join(path, 'temporary'))
        except:
            pass
        for id in os.listdir(path):
            try:
                l = [os.path.join(path, id, i) for i in os.listdir(os.path.join(path, id))]
                infer(l, id, os.path.join(path, 'temporary'), self)
                self.finish_signal.emit({'finish': True})
            except:
                pass
        pred = pd.DataFrame()

        for i in os.listdir(os.path.join(path, 'temporary')):
            id = i.split('.')[0]
            id = id.split('_')[2]
            prom = pd.read_csv(os.path.join(path, 'temporary', i))
            prom['image_name'] = [f'{id}/{i}' for i in prom['image_name'].to_list()]
            pred = pd.concat([pred, prom], ignore_index=True)

        pred['time'] = get_path_to_time([os.path.join(path, i) for i in pred['image_name'].to_list()])
        pred['name_folder'] = [i.split('/')[0] for i in pred['image_name']]
        pred.to_csv('pred.csv')
        final = get_final_csv(pred)
        final.to_csv('finall.csv', index=False)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        self.selectedFolder = ""
        super(Ui, self).__init__()  # Call the inherited classes __init__ method
        inn = io.BytesIO()
        inn.write(ui.encode('utf-8'))
        inn.seek(0)
        uic.loadUi(inn, self)  # Load the .ui file
        self.show()  # Show the GUI
        self.openButton.clicked.connect(self.open_folder)
        self.predictButton.clicked.connect(self.predict_images)

    def open_folder(self):
        fname = QFileDialog.getExistingDirectory(self, 'Открыть папку с фотографиями...', "")
        self.selectedFolder = fname
        self.label.setText("Папка: %s" % self.selectedFolder)
        self.predictButton.setEnabled(True)

    def predict_images(self):
        self.label.setText("В процессе...")
        self.progressBar.setEnabled(True)
        self.worker = Worker(self.selectedFolder)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.finish_signal.connect(self.thread.quit)
        self.worker.finish_signal.connect(self.finish)
        self.thread.start()

    def finish(self, result):
        length = 0
        for id in os.listdir(self.selectedFolder):
            if os.path.isdir(os.path.join(self.selectedFolder, id)):
                length += len(os.listdir(os.path.join(self.selectedFolder, id)))
        if result.get('finish') and not result.get('error'):
            self.selectedFolder = ""
            self.label.setText("Результаты сохранены в 'results.xlsx'")
            self.predictButton.setEnabled(False)
            self.progressBar.setValue(0)
            self.progressBar.setEnabled(False)
        elif result.get('error'):
            self.selectedFolder = ""
            self.label.setText("Что-то пошло не так...")
            self.predictButton.setEnabled(False)
            self.progressBar.setValue(0)
            self.progressBar.setEnabled(False)
        elif result.get('index'):
            self.progressBar.setValue((self.progressBar.value() / length) * 100)


app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
window = Ui()  # Create an instance of our class
app.exec_()  # Start the application
