# Discrete Denoising Diffusion Approach to Integer Factorization

This repository contains the official TensorFlow implementation of the following paper:
> **Discrete Denoising Diffusion Approach to Integer Factorization**
>
>Integer factorization is a famous computational problem unknown whether being solvable in the polynomial time. With the rise of deep neural networks, it is interesting whether they can facilitate faster factorization. We present an approach to factorization utilizing deep neural networks and discrete denoising diffusion that works by iteratively correcting errors in a partially-correct solution. To this end, we develop a new seq2seq neural network architecture, employ relaxed categorical distribution and adapt the reverse diffusion process to cope better with inaccuracies in the denoising step. The approach is able to find factors for integers of up to 56 bits long. Our analysis indicates that investment in training leads to an exponential decrease of sampling steps required at inference to achieve a given success rate, thus counteracting an exponential run-time increase depending on the bit-length. 
>
> 

## Requirements

* Python 3.8
* Nvidia T4 (16Gb) or better
* 16GB of RAM

To install python dependencies run:

```sh
pip install -r requirements.txt
```

## Usage

To train the model, run this command:
```sh
python3 trainer.py
```

To run sampling use:
```sh
python3 diffussion_sampler.py
```
It will use our pretrained model by default. The pretrained model used to produce the results in 
this paper is provided in the folder `pretrained_model`. The global configuration is given in `config.py`, settings for 
training are specified at the top of `trainer.py`, the settings for sampling at the top of `diffusion_sampler.py`.   
