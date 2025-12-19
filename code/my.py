import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGridLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MainWindow')
        self.setGeometry(100, 100, 1000, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        self.original_image = None
        self.label_size = 350
        
        self.initUI()

    def initUI(self):
        button_layout = QVBoxLayout()
        
        self.btn_load = QPushButton('Load Image', self)
        self.btn_load.clicked.connect(self.loadImage)
        button_layout.addWidget(self.btn_load)
        
        self.btn_smooth = QPushButton('Smooth Filter', self)
        self.btn_smooth.clicked.connect(self.applySmoothFilter)
        button_layout.addWidget(self.btn_smooth)
        
        self.btn_sharp = QPushButton('Sharp', self)
        self.btn_sharp.clicked.connect(self.applySharpFilter)
        button_layout.addWidget(self.btn_sharp)

        self.btn_gaussian = QPushButton('Gaussian', self)
        self.btn_gaussian.clicked.connect(self.applyGaussianFilter)
        button_layout.addWidget(self.btn_gaussian)

        self.btn_lowerpass = QPushButton('Lower-pass', self)
        self.btn_lowerpass.clicked.connect(self.applyLowerPassFilter)
        button_layout.addWidget(self.btn_lowerpass)
        
        button_layout.addStretch()

        image_layout = QGridLayout()
        
        self.label1 = self.createImageLabel('Original Image')
        self.label2 = self.createImageLabel('Result 1')
        self.label3 = self.createImageLabel('Result 2')
        self.label4 = self.createImageLabel('Result 3')

        image_layout.addWidget(self.label1, 0, 0)
        image_layout.addWidget(self.label2, 0, 1)
        image_layout.addWidget(self.label3, 1, 0)
        image_layout.addWidget(self.label4, 1, 1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addLayout(image_layout)
        self.main_layout.setStretch(1, 4)

    def createImageLabel(self, text):
        label = QLabel(self)
        label.setFixedSize(self.label_size, self.label_size)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        label.setText(text)
        return label

    def clearLabels(self):
        self.label1.clear()
        self.label2.clear()
        self.label3.clear()
        self.label4.clear()
        
        self.label1.setText('Original Image')
        self.label2.setText('Result 1')
        self.label3.setText('Result 2')
        self.label4.setText('Result 3')

    def displayImage(self, label, cv_img):
        if cv_img is None:
            label.clear()
            return

        display_img = cv_img.copy()

        if display_img.dtype != np.uint8:
            display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX)
            display_img = display_img.astype(np.uint8)
        
        h, w = display_img.shape
        bytes_per_line = w
        q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(self.label_size, self.label_size, Qt.KeepAspectRatio))

    def loadImage(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.original_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            if self.original_image is not None:
                self.clearLabels()
                self.displayImage(self.label1, self.original_image)
                self.label1.setText('')
            else:
                self.clearLabels()
                self.label1.setText('Failed to load image')

    def applySmoothFilter(self):
        if self.original_image is None:
            return
            
        self.clearLabels()
        self.label1.setText('')
        self.label2.setText('1(a) Average filter')
        self.label3.setText('1(a) Median filter')
        self.label4.setText('1(b) Fourier transform')

        img_avg = cv2.blur(self.original_image, (5, 5))
        img_median = cv2.medianBlur(self.original_image, 5)

        dft = cv2.dft(np.float32(self.original_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = self.original_image.shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 60
        cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        self.displayImage(self.label1, self.original_image)
        self.displayImage(self.label2, img_avg)
        self.displayImage(self.label3, img_median)
        self.displayImage(self.label4, img_back)

    def applySharpFilter(self):
        if self.original_image is None:
            return

        self.clearLabels()
        self.label1.setText('')
        self.label2.setText('No use')
        self.label3.setText('2(a) Sobel mask')
        self.label4.setText('2(b) Fourier transform')

        sobelx = cv2.Sobel(self.original_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.original_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobelx, sobely)
        sobel_img = cv2.convertScaleAbs(sobel_mag)

        dft = cv2.dft(np.float32(self.original_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = self.original_image.shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 30
        cv2.circle(mask, (ccol, crow), r, (0, 0), -1)
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        self.displayImage(self.label1, self.original_image)
        self.displayImage(self.label3, sobel_img)
        self.displayImage(self.label4, img_back)

    def applyGaussianFilter(self):
        if self.original_image is None:
            return
            
        self.clearLabels()
        self.label1.setText('')
        self.label2.setText('Result')
        self.label3.setText('No use')
        self.label4.setText('No use')

        gaussian_img = cv2.GaussianBlur(self.original_image, (5, 5), 0)

        self.displayImage(self.label1, self.original_image)
        self.displayImage(self.label2, gaussian_img)

    def applyLowerPassFilter(self):
        if self.original_image is None:
            return

        self.clearLabels()
        self.label1.setText('')
        self.label2.setText('Result')
        self.label3.setText('No use')
        self.label4.setText('No use')

        dft = cv2.dft(np.float32(self.original_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = self.original_image.shape
        crow, ccol = rows // 2, cols // 2

        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        xx, yy = np.meshgrid(x, y)
        D = np.sqrt(xx**2 + yy**2)
        
        D0 = 30
        mask_2d = np.exp(-(D**2) / (2 * D0**2))
        
        mask = np.zeros((rows, cols, 2), np.float32)
        mask[:, :, 0] = mask_2d
        mask[:, :, 1] = mask_2d
        
        fshift = dft_shift * mask
        
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        self.displayImage(self.label1, self.original_image)
        self.displayImage(self.label2, img_back)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())