# Neural Style
An easy-to-use, lightweight style transfer implementation with [SqueezeNet](https://arxiv.org/abs/1602.07360). Less than 1 minute runtime per image on my 2014 macbook pro. Adapted from Assignment 3 of [Stanford cs231n](http://cs231n.stanford.edu/)

## Step by Step Usage 
1. Download this folder on this github page.
2. Enter this folder from command line and create a virtual environment equipped with python 3.5+. To do so, try something like the following. If you use anaconda, click [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) for help. General help on virtual environments can be found [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
  ```
  cd style_transfer-master
  sudo pip install virtualenv      # This may already be installed
  virtualenv -p python3 ve       # Create a virtual environment (python3) named ve
  source ve/bin/activate         # Activate the virtual environment
  source deactive                 # deactivate when done.
  ``` 
3. Install dependencies via ```pip install -r requirements.txt```
4. You are good to go. Transfer style by running ```python3 styletransfer.py input_image style_image
```. For example, run ```python3 styletransfer.py input/stats.jpg style/starry_night.jpg```. Note that you may need to replace ```python3``` by whatever name your python interpreter has in the virtual environment. 


## Additional Info
* This is an implementation of [\"Image Style Transfer Using Convolutional Neural Networks\" (Gatys et al., CVPR 2015)](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* Main function with adjustable parameters is in styletransfer.py. Work horse functions are in model_utils.py