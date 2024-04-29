  **encode=gbk**  
文件结构：  
│  
├─data //原图片  
│  ├─test  
│  │  └─image  
│  └─train  
│      ├─image  
│      └─label  
├─npy_data  
│     ├─imgs_mask_train.npy  
│     ├─imgs_test.npy  
│     └─imgs_train.npy  
├─results  
│  ├─results_jpg//此文件夹存放训练结果的黑白图像  
│  │    
│  └─imgs_mask_test.npy //训练结果的矩阵
│  data.py //数据预处理  
│  main.py     
│  my_model_weights.h5 //模型参数  
│  unet.hdf5 //训练记录文件  
│  unet.py //模型主体  

**重新生成模型参数时，若报错oom（out of memory）可设置使用cpu运行**  
直接运行main.py即可

tensorflow-gpu            2.4.1
numpy                     1.19.2
python                    3.8.19