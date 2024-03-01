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

3. Pastikan telah melakukan download atau clone repository di atas, telah membuka file berformat .ipynb di atas, mengubah runtime ke gpu,  dan  instal semua library dependensi yang diperlukan seperti dibawah ini : 

   ```python
   !pip install super-gradients==3.1.0
   !pip install imutils
   !pip install roboflow
   !pip install pytube --upgrade
   !pip install super-gradients
   ```

4. Import semua modul yang diperlukan :
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

5. Membuat check point dalam directiory untuk data train :
   ```python
   CHECKPOINT_DIR = 'checkpoints2'
   trainer = Trainer(experiment_name='ppe_yolonas_run2', ckpt_root_dir=CHECKPOINT_DIR)
   '''

6. Unduh dataset dari website Roboflow dengan menjalankan kode program berikut :
   ```python
   !pip install roboflow
   from roboflow import Roboflow
   rf = Roboflow(api_key="6jnEorjCrjLrIuI3IGb4")
   project = rf.workspace("objet-detect-yolov5").project("eep_detection-u9bbd")
   dataset = project.version(1).download("yolov5")
   ```

7. Siapkan data training, validation, testing :
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
   ```
8. Menginisialisasikan menginisialisasikan class, labels directory, images directory dalam train data, validation data, testing data dalam dataset

   ```python
   rain_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
   )

   val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
   )

   test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
   )

   clear_output()
   ```
   
9. membuat transfors data pada dataset
    ```python
    train_data.dataset.transforms
    train_data.dataset.dataset_params['transforms'][1]
    train_data.dataset.dataset_params['transforms'][1]['DetectionRandomAffine']['degrees'] = 10.42
    ```
    
10. menampilkan visualisasi setiap plot yang berhasil melakukan detection pada object APD
    ```python
    train_data.dataset.plot()
    ```
11. membuat inisialisasi model
    pada tahap ini saya memakai model dari yolo dan super gradiens dan untuk 
    weight pada dataset ini menggunakan coco bawaan dari yolo :

    ```python
    model = models.get('yolo_nas_s',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )
    ```
    
12. Membuat Inisiliasi parameter training

    ```python
    train_params = {
    # ENABLING SILENT MODE
    'silent_mode': True,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 10,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
    }
    ```
13. mendownload demo videos

    ```python

    !gdown "https://drive.google.com/drive/folders/1kCItlg3nLnSWPyPKk4In4SBXxQgPIbid"
    ```

14. training models
    ```python
    trainer.train(model=model,
              training_params=train_params,
              train_loader=train_data,
              valid_loader=val_data)
    ```

15. mendapatkan best models
    ```python
    best_model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="/content/checkpoints2/ppe_yolonas_run2/RUN_20240229_070004_684804/ckpt_best.pth")
    ```
16. membuat evalutions models

    ```python
    trainer.test(model=best_model,
            test_loader=test_data,
            test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                   top_k_predictions=300,
                                                   num_cls=len(dataset_params['classes']),
                                                   normalize_targets=True,
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                                          nms_top_k=1000,
                                                                                                          max_predictions=300,
                                                                                                          nms_threshold=0.7)
                                                  ))
    ```
17. membuat object predict dengan menggunakan best models
    ```python
    img_url = '/content/EEP_Detection- 
    1/valid/images/Sikctin0132_jpg.rf.9e9d6b610c5dbebd34e34c729065ec8d.jpg'
    best_model.predict(img_url).show()
    ```
18. testing models dengan video
    ```python
    input_video_path = f"/content/JapanPPE.mp4"
    output_video_path = "detections.mp4"
    import torch
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    best_model.to(device).predict(input_video_path).save(output_video_path)
    ```

19. menampilkan kode program output dalam bentuk video
    ```python
    from IPython.display import HTML
    from base64 import b64encode
    import os

    # Input video path
    save_path = '/content/detections.mp4'

    # Compressed video path
    compressed_path = "/content/result_compressed2.mp4"

    os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

    # Show video
    mp4 = open(compressed_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls>
      <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
    ```

20. semua kode program di atas bisa di dapatkan di folder code dalam repository ini atau bisa lewat link google colab di bawah ini : 

[Anugrah Aidin Yotolembah-APD DETECTION](https://colab.research.google.com/drive/1BqS0frAQ793V9Vshe3TA-1CZpooPhu5s?usp=sharing)


B. OUTPUT APD DETECTION
1. output with image
   <img width="631" alt="Screenshot 2024-03-02 at 01 41 52" src="https://github.com/AnugrahAidinYotolembah/APD_DETECTION/assets/108518030/71a3ee49-72b5-40ca-a5fd-91abac3e15d6">
   <img width="408" alt="Screenshot 2024-03-02 at 01 42 15" src="https://github.com/AnugrahAidinYotolembah/APD_DETECTION/assets/108518030/e2e64a40-d9af-475d-8aea-498eb9e8009d">

   
2. output with videos
   
https://github.com/AnugrahAidinYotolembah/APD_DETECTION/assets/108518030/5e6e5e91-9af0-49b8-813d-b7b463a38afc



https://github.com/AnugrahAidinYotolembah/APD_DETECTION/assets/108518030/02d6e120-7ffb-4bb0-89c2-8e348df6d2be


C. CONCLUSION

1. saya menggunakan library python package yaitu :
   - super gradients
   - roboflow
   - pytorch
   - imutils

2. Runtime yang saya gunakan yaitu GPU bawaan dari google colab

3. di sini saya menggunakan image dataset dari api key roboflow dengan kata kunci 
   PPE (Personal Protective Equipmnet) di karnakan dataset ini cocok di terapkan 
   untuk mendeteksi APD (Alat Perlindungan Diri). dataset ini juga menampilkan 
   object dengan baik,  dari gambar yang dekat, pencahayaan yang jelas, bentuk 
   object yang jelas, dll.

4. saya membuat 7 class yang akan melakukan deteksi dan sesuai yang di minta yaitu:
   - Protective Helmet
   - Shield
   - Protective Jacket
   - Dust Mask
   - Eye Wear
   - Glove
   - Protective Boots

5. saya menggunakan Algoritma AI  yolov5/ yoloNas dengan bobot/ weight yaitu coco bawaan dari yolo, alasan memakai ini di karnakan yolov5 merupakan algoritma yang pendeteksiannya lebih akurat dan lebih baik dari model atau algoritma yang lainnya, dan algoritma AI yolov5 ini sedang tren untuk teknologi Object detections

6. pada saat training image saya menggunakan 10 epoch dan threshold sebanyak 0.7 di karnakan value dari epoch dan threshold untuk detection objeck sudah menghasilkan gambar detection yang baik.

7. output saat di tampilkan pada mode gambar, kode program berhasil medeteksi object PPE / APD dengan baik dan sesuai class yang telah di tentukan, contohnya ketika muncul gambar sarung tangan/ glove, kode program atau models yolo bisa mendeteksi bahwa itu sarung tangan, begitu pun dengan class yang lain.

8. output saat di tampilkan pada mode video, awalnya saya mencoba contoh video yang di berikan, tetapi hasil untuk mendeteksi object APD kurang akurat di karnakan menurut saya faktor video yang di berikan jauh dari pandangan untuk melakukan detection object.

untuk itu saya menggunakan video saya sendiri dan ketika kode program / models yang telah di latih menggunakan yolo, object yang didalam video berhasil terdeteksi sesuai class detection yang telah di tentukan, hampir semuanya mendekati akurat dan sesuai untuk melakukan deteksi object APD, tetapi ada sedikit keliru ketika mendeteksi papan iklan dalam video yang berwarna orange, papan iklan tersebut di deteksi sebagai jacket pekerja bangunan.








   



