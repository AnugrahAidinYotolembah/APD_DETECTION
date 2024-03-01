SYNAPSIS AI ENGINEER FULLTIME CHALLENGE (PT Synapsis Sinergi Digital)

NAMA : ANUGRAH AIDIN YOTOLEMBAH

Sistem Pengenalan Alat Pelindung Diri
Dalam industri konstruksi, manufaktur, dan laboratorium, penggunaan Alat Pelindung Diri (APD) sangat penting untuk menjaga keselamatan kerja. Sebuah sistem yang dapat mengenali penggunaan APD secara otomatis dapat membantu memastikan bahwa semua pekerja mematuhi standar keselamatan. 
Pada challenge kali ini anda diminta untuk menegembangkan sistem AI yang dapat mengenali berbagai jenis APD pada pekerja dalam lingkungan kerj. Sistem harus mampu mengidentifikasi kehadiran atau ketiadaan item APD spesifik seperti helm, masker, sarung tangan, kacamata pelindung, dan sepatu keselamatan.

A. Langkah - Langkah menjalankan kode program Sistem Penganalan Alat Pelingdung 
   Diri (APD DETECTION) : 

1. download atau clone repository :
   https://github.com/AnugrahAidinYotolembah/APD_DETECTION.git

2. Setelah download atau clone repository, masuk ke dalam folder code, dan buka kode program yang telah di buat yang bernama :

   Anugrah_Aidin_Yotolembah_APD DETECTION.ipynb

3. Pastikan telah melakukan download atau clone repository di atas dan pastikan
   telah membuka file berformat .ipynb di atas dan  instal semua library dependensi yang diperlukan seperti dibawah ini : 

   ```python
   !pip install super-gradients==3.1.0
   !pip install imutils
   !pip install roboflow
   !pip install pytube --upgrade
   !pip install super-gradients
   ```

5. Import semua modul yang diperlukan :
   ```python
   from super_gradients.training import Trainer
   from super_gradients.training import dataloaders
   from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
   from IPython.display import clear_output
   from super_gradients.training.losses import PPYoloELoss
   from super_gradients.training.metrics import DetectionMetrics_050
   from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
   from super_gradients.training import models
   from roboflow import Roboflow
   import os
   ```

6. Membuat check point dalam directiory untuk data train :
   ```python
   CHECKPOINT_DIR = 'checkpoints2'
   trainer = Trainer(experiment_name='ppe_yolonas_run2', ckpt_root_dir=CHECKPOINT_DIR)
   '''

7. Unduh dataset dari website Roboflow dengan menjalankan kode program berikut :
   ```python
   !pip install roboflow
   from roboflow import Roboflow
   rf = Roboflow(api_key="6jnEorjCrjLrIuI3IGb4")
   project = rf.workspace("objet-detect-yolov5").project("eep_detection-u9bbd")
   dataset = project.version(1).download("yolov5")
   ```

8. Siapkan data training, validation, testing :
   ```python
   dataset_params = {
   'data_dir':'/content/EEP_Detection-1',
   'train_images_dir':'train/images',
   'train_labels_dir':'train/labels',
   'val_images_dir':'valid/images',
   'val_labels_dir':'valid/labels',
   'test_images_dir':'test/images',
   'test_labels_dir':'test/labels',
   'classes': ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']
   }
9. 
10. 


