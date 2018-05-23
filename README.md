### baidu salers logo prediction
---

#### Basic info
* This project is based on gluon or a more famous parent: mxnet.

#### Project code structure
* dataloader: data preprocess
    * dataPreProcess.py
        * 80% of dataset used for train, 20% of dataset used for valid.
    * augmentation.py
        * add padding, rotate
        * ColorJitterAug: 0.3
        * random gray, prob: 0.5
        * calc dataset mean and std（utils/calc_mean_std.py），used for dataset normalize
* logs
    * training logs
* model
    * define model
* utils
    * calc_mean_std.py
        * calc dataset mean and std，used for dataset normalize
* weights
    * save model weights
* train.py
    * train model code
    * change pretrianed_model_name for different pretrained models, [pretrained models](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html?highlight=model#module-mxnet.gluon.model_zoo.vision)
* predict.py
    * predict code for test and valid
* predict_wrong_ana.py
    * predict valid dataset label, find what's wrong with model
    * used for find best concat params
* concateResult.py
    * embedding model predict result

#### Future Work
- [ ] valid data can join train, use cv for valid
- [ ] [mixup：BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
- [ ] pseudo label
- [ ] more augmentation
- [ ] more init methods