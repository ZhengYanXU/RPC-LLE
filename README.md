# RPC-LLE
## Experiment

PyTorch implementation of RPC-LLE

### Requirements

- Python 3.7 
- PyTorch 1.4.0
- opencv
- torchvision 
- numpy 
- pillow 
- scikit-learn 
- tqdm 
- matplotlib 
- visdom 

SCL-LLE does not need special configurations. Just basic environment.

### Folder structure

The following shows the basic folder structure.
```python
├── datasets
│   ├── data
│   │   ├── cityscapes
|   ├── test_data
│   ├── cityscapes.py
|   └── util.py
├── network
├── lowlight_test.py # low-light image enhancement testing code
├── train.py # training code
├── lowlight_model.py
├── Myloss.py
├── checkpoints
│   ├── LLE_model.pth #  A pre-trained SCL-LLE model
```

### Test

- cd RPC-LLE


```
python lowlight_test.py
```

The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "datasets". You can find the enhanced images in the "result" folder.

### Train

1. cd RPC-LLE
2. download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset
3. download the cityscapes training data <a href="https://drive.google.com/file/d/1FzYwO-VRw42vTPFNMvR28SnVWpIVhtmU/view?usp=sharing">google drive</a> and contrast training data <a href="https://drive.google.com/file/d/1A2VWyQ9xRXClnggz1vI-7WVD8QEdKJQX/view?usp=sharing">google drive</a> 
4. unzip and put the downloaded "train" folder to "datasets/data/cityscapes/leftImg8bit" folder



```
python train.py
```
