## How to download datasets

The file structure of each dataset folder is:

```
dataset/
|-- images/
|-- concepts/
|-- splits/
```

- Place all images for each dataset in the `images/` directory. 
- The `concepts/` directory contains a JSON file,`class2concepts_gemini.json`, which stores a dictionary where the keys are class names and values are the lists of candidate concepts generated using LLM and then filtered. 
- The `splits/` directory contains the train/val/test splits in pickle format.

Here is the downloading links for each dataset:

* **CIFAR-10**: [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
> **Note:** The downloaded files are raw images. Extract them by running `python cifar_dataset_gen.py`.
* **CUB**: [https://data.caltech.edu/records/20098](https://data.caltech.edu/records/20098)
* **flower**: [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* **food**: [http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)