# EarthMapper #

Project repository for EarthMapper.  This is a toolbox for the semantic segmentation of non-RGB (i.e., multispectral/hyperspectral) imagery.  

<p align="center">
<img  src="http://www.cis.rit.edu/~rmk6217/img/earthmapper.png">
</p>

## Description ##

TODO

## Dependencies ##
* Python 3.5 (We recommend the [Anaconda Python Distribution](https://www.anaconda.com/download/))
* numpy, scipy, and matplotlib
* [scikit-learn](http://scikit-learn.org/stable/)
* [spectral python](http://www.spectralpython.net/)
* [gdal](http://www.gdal.org/)
* [tensorflow](https://www.tensorflow.org/)
* [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
* [gco_python](https://github.com/amueller/gco_python)

## Instructions ##

### Installation ###
```console
$ python setup.py
```
### Run example ###
```console
$ python examples/example_pipeline.py
```
## Citations ##

If you use our product, please cite:
* Kemker, R., Luu, R., and Kanan C. [Low-Shot Learning for the Semantic Segmentation of Remote Sensing Imagery](https://arxiv.org/abs/1803.09824). In review at the IEEE Transactions on Geoscience and Remote Sensing (TGRS).
* Kemker, R., Kanan C. (2017) [Self-Taught Feature Learning for Hyperspectral Image Classification](http://ieeexplore.ieee.org/document/7875467/). IEEE Transactions on Geoscience and Remote Sensing (TGRS), 55(5): 2693-2705. 10.1109/TGRS.2017.2651639
* U. B. Gewali and S. T. Monteiro, [A tutorial on modeling and inference in undirected graphical models for hyperspectral image analysis](https://arxiv.org/abs/1801.08268), In review at the International Journal of Remote Sensing (IJRS).
* U. B. Gewali and S. T. Monteiro, [Spectral angle based unary energy functions for spatial-spectral hyperspectral classification using Markov random fields](http://ieeexplore.ieee.org/abstract/document/8071716/), in Proc. IEEE Workshop on Hyperspectral Image and Signal Processing : Evolution in Remote Sensing (WHISPERS), Los Angeles, CA, USA, Aug. 2016.

## Points of Contact ##
* Ronald Kemker -  http://www.cis.rit.edu/~rmk6217/
* Utsav Gewali - http://www.cis.rit.edu/~ubg9540/
* Chris Kanan - http://www.chriskanan.com/

## Also Check Out ##
* RIT-18 Dataset - https://github.com/rmkemker/RIT-18
