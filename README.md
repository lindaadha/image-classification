# Level 1_ Image Classification
Membuat _Image Classification_ beserta API nya. Terdapat kelas object yang diklasifikasi, yaitu :
1. 'bike'
2. 'car'
3. 'helicopter'
4. 'ship'
5. 'truck'

### Requirements :
- Python
- Pytorch
- Flask
- PIL
- Numpy
- Torch-W (MobilenetV2)

### Preparing Dataset
Folder yang perlu disiapkan yakni Train(80%) dan Validasi(20%) dari total gambar yang dimiliki, folder yang perlu disiapkan adalah 
Split_dataset
|-- Train
Bike    

### Training 
Proses training dapat menggunakan [Training.py] dengan mengubah directory train_set dan test_set 
```sh
train_set = 'split_dataset/train'
test_set = 'split_dataset/test'
```

### Inference
Hasil inference yang ingin didapatkan dari seluruh Test yang tidak digunakan dalam proses Training, yang ditampilkan adalah hasil result dan confidence yang didapatkan
```sh
(base) F:\AI_Trial\Mentoring_AI\Level 1_Vehicle Detection>python inference.py
0.9877373
ship
```

### Result API 
Membuat API dengan return JSON, berikut hasil returnya 
```sh
{
    "data": {
        "prob": 0.6374825239181519,
        "result": "helicopter"
    }
}
```