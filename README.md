<!-- SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited. -->

# Deep Learning-Based Synchronization for Uplink NB-IoT

Implementation of the NPRACH detection algorithm from
[[A]](https://arxiv.org/abs/2205.10805) using the
[Sionna link-level simulator](https://nvlabs.github.io/sionna/).

## Abstract

We propose a neural network (NN)-based algorithm for device detection and time
of arrival (ToA) and carrier frequency offset (CFO) estimation for the
narrowband physical random-access channel (NPRACH) of narrowband internet of
things (NB-IoT). The introduced NN architecture leverages residual convolutional
networks as well as knowledge of the preamble structure of the 5G New Radio
(5G NR) specifications. Benchmarking on a 3rd Generation Partnership Project
(3GPP) urban microcell (UMi) channel model with random drops of users against a
state-of-the-art baseline shows that the proposed method enables up to 8 dB
gains in false negative rate (FNR) as well as significant gains in false
positive rate (FPR) and ToA and CFO estimation accuracy. Moreover, our
simulations indicate that the proposed algorithm enables gains over a wide range
of channel conditions, CFOs, and transmission probabilities. The introduced
synchronization method operates at the base station (BS) and, therefore,
introduces no additional complexity on the user devices. It could lead to an
extension of battery lifetime by reducing the preamble length or the transmit
power.

## Setup

Running this code requires [Sionna](https://nvlabs.github.io/sionna/).
To run the notebooks on your machine, you also need [Jupyter](https://jupyter.org).
We recommend Ubuntu 20.04, Python 3.8, and TensorFlow 2.8.

## Structure of this repository

Two notebooks may serve as starting point:

* [Train.ipynb](Train.ipynb) : Implements the training loop of the deep learning-based synchronization algorithm.
* [Evaluate.ipynb](Evaluate.ipynb) : Evaluates the trained deep learning-based synchronization algorithm and a baseline. This notebook reproduces the plots from the paper related to this repository [[A]](https://arxiv.org/abs/2205.10805).

These notebooks rely on the following modules:

* [nprach/](nprach/) : Implements the NPRACH waveform.
* [synch/](synch/) : Implements two NPRACH synchronization algorithms, the deep learning-based one that we propose [[A]](https://arxiv.org/abs/2205.10805) and a strong baseline [[B]](https://ieeexplore.ieee.org/abstract/document/9263250/).
* [e2e/](e2e/) : Implements a model for simulating the end-to-end system, which includes NPRACH waveform generation, 3GPP UMi channel model, and synchronization using the two available algorithms.

In addition, the [parameters.py](parameters.py) file defines the key simulation parameters, and the results computed by the [Evaluate.ipynb](Evaluate.ipynb) notebook are available in the [results/](results/) directory.
Moreover, the weights resulting from the training of the deep learning-based synchronization algorithm are available [here](https://drive.google.com/file/d/1qw8YG5RieJB7qf-Pj_FfqCLHdZtTXpFi/view?usp=sharing), which allows reproducing the results from [[A]](https://arxiv.org/abs/2205.10805) without retraining the neural network.

## References

[A] [F. Aït Aoudia, J. Hoydis, S. Cammerer, M. Van Keirsbilck, and A. Keller, "Deep Learning-Based Synchronization for Uplink NB-IoT", 2022](https://arxiv.org/abs/2205.10805)

[B] [H. Chougrani, S. Kisseleff and S. Chatzinotas, "Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone Frequency Hopping Random Access in NB-IoT," in IEEE Internet of Things Journal, vol. 8, no. 9, pp. 7437-7449, 1 May1, 2021, doi: 10.1109/JIOT.2020.3039004](https://ieeexplore.ieee.org/abstract/document/9263250/)

## License

Copyright &copy; 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia License](LICENSE.txt).
