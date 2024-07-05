import os
import time

import xlsxwriter as xs
from time import sleep
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import sys

exs = ['png', 'jpg', 'jpeg']


def predict(filename):
    time.sleep(0.01)
    return 'Медведь'


def get_images(folder):
    res = []
    for i in os.listdir(folder):
        if i.split('.')[-1].lower() in exs:
            res.append(os.path.join(folder, i))
    return res


class Worker(QObject):
    finish_signal = pyqtSignal(dict)

    def __init__(self, selectedFolder):
        super().__init__()
        self.selectedFolder = selectedFolder

    def start(self):
        try:
            header = ('Название файла', 'Класс')
            xlsx = open(os.path.join(self.selectedFolder, 'results.xlsx'), 'wb')
            workbook = xs.Workbook(xlsx)
            worksheet = workbook.add_worksheet()
            for i, row in enumerate(header):
                worksheet.set_column(i, i, 35)
                worksheet.write(0, i, row)
            images = get_images(self.selectedFolder)
            for i, photo in enumerate(images):
                filename = os.path.split(photo)[-1]
                class_photo = predict(photo)
                row = (filename, class_photo)
                for j, data in enumerate(row):
                    worksheet.write(i + 1, j, data)
                self.finish_signal.emit({'index': i})
            workbook.close()
            xlsx.close()
            self.finish_signal.emit({'index': i + 1})
            self.finish_signal.emit({'finish': True})
        except Exception as e:
            print(f"Error: {e}")
            self.finish_signal.emit({'finish': True, 'error': e})


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        self.selectedFolder = ""
        super(Ui, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi('app.ui', self)  # Load the .ui file
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
        self.progressBar.setValue(len(os.listdir(self.selectedFolder)))
        self.progressBar.setEnabled(True)
        self.worker = Worker(self.selectedFolder)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.finish_signal.connect(self.thread.quit)
        self.worker.finish_signal.connect(self.finish)
        self.thread.start()

    def finish(self, result):
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
            self.progressBar.setValue(result.get('index'))


app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
window = Ui()  # Create an instance of our class
app.exec_()  # Start the application
