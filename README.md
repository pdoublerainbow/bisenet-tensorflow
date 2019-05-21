# Bisenet-tensorflow
This project aims at providing a re-implement of bisenet by using tensorflow.  
![image](https://github.com/pdoublerainbow/bisenet-tensorflow/blob/master/example/0001TP_007170_L.png)
# Prerequisites
* tensorflow-gpu >=1.5.0
* cv2
* sacred

* pip install tensorflow-gpu==1.5.0
* pip install sacred
* pip install opencv-python  
  
#Optional: Install nvidia-ml-py for automatically selecting GPU
* #pip install nvidia-ml-py
  
# Train
  You can simply run train.py by default hyper-parameter, or modify the hyper-parameter in configuration.py by yourself first.
### Tensorboard Result
![image](https://github.com/pdoublerainbow/bisenet-tensorflow/blob/master/tensorboard_result.png)
# Test and Predict
*  If you have train this model by yourself, you can simply run test.py, and then view the result by using tensorboard. And you can predict one picture that have spcified in the example dir by simply run predict.py, then the predict will be saved in the example dir too.
*  If you just want to test and predict this model trained by me. You should do things as follow:
1. cd bisenet-tensorflow
2. mkdir -p Logs/bisenet/checkpoints/bisenet-v2
3. download the model in this link （will update soon） and realse them in the dir that have created
4. then you can run test.py or predict.py

# Citation
Please consider citing the BiSeNet in your publications if it helps your research.

@inproceedings{yu2018bisenet,
  title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={European Conference on Computer Vision},
  pages={334--349},
  year={2018},
  organization={Springer}
}

# Reference
* https://github.com/ycszen/TorchSeg
* https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
* https://arxiv.org/abs/1808.00897
