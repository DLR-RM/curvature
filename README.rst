Laplace Approximation for Bayesian Deep Learning
************************************************

Curvature Library
============
Curvature Library is an official code for the following papers.

**Estimating Model Uncertainty of Neural Networks in Sparse Information Form**
Jongseok Lee, Matthias Humt, Jianxiang Feng, Rudolph Triebel, ICML 2020.
(`paper <https://proceedings.icml.cc/static/paper_files/icml/2020/2525-Paper.pdf>`_)

**Bayesian Optimization Meets Laplace Approximation for Robotic Introspection**
Jongseok Lee, Matthias Humt, Rudolph Triebel, IROS 2020 Workshop.
(`paper <https://elib.dlr.de/137021/1/ICRA2020LTAWS_paper_2.pdf>`_)

**Learning Multiplicative Interactions with Bayesian Neural Networks for Visual-Inertial Odometry**
Kashmira Shinde, Jongseok Lee, Matthias Humt, Aydin Sezgin, Rudolph Triebel, ICML 2020 Workshop
(`paper <https://elib.dlr.de/135547/1/Learning%20Multiplicative%20Interactions%20with%20Bayesian%20Neural%20Networks%20for%20Visual-Inertial%20Odometry.pdf>`_)

Overview
============

This repository contains PyTorch implementations of several Laplace approximation methods (`LA <https://pdfs.semanticscholar.org/b0f2/433c088591d265891231f1c22424047f1bc1.pdf>`_) [1_].
It is similar to this `TensorFlow implementation <https://github.com/tensorflow/kfac>`_ which approximates the curvature of neural networks, except that our main purpose is approximate Bayesian inference instead of second-order optimization. 

The following approximations to the Fisher information matrix (IM) are supported with different fidelty-complexity trade-offs:

1. Diagonal (`DIAG <https://nyuscholars.nyu.edu/en/publications/improving-the-convergence-of-back-propagation-learning-with-secon>`_) [7_]
2. Kronecker Factored Approximate Curvature (`KFAC <https://openreview.net/pdf?id=Skdvd2xAZ>`_) [2_, 3_, 6_]
3. Eigenvalue corrected KFAC (`EFB <https://arxiv.org/pdf/1806.03884.pdf>`_) [4_]
4. Sparse Information Form (`INF <https://proceedings.icml.cc/static/paper_files/icml/2020/2525-Paper.pdf>`_)

The aim is to make LA easy to use while LA in itself is a practical approach, because trained networks can be used without any modification. Our implementation supports this plug-in-and-play principle, i.e. you can make already pretrained network Bayesian, and obtain calibrated uncertainty in deep neural network's predictions! Our library also features a Bayesian Optimization method for easier tuning of hyperparameters.

Installation
============

To install the module, clone or download the repository and run:

.. code-block:: console

    $ pip install .
    
To install the optional dependencies for plotting (``plot``), evaluation (``eval``), hyperparameter optimization (``hyper``) or data loading (``data``) run:

.. code-block:: console

    $ pip install .[extra]
    
where ``extra`` is the name of the optional depency (in brackets). To install multiple optional dependencies at once run e.g.:

.. code-block:: console

    $ pip install ".[plot, data, eval]"

Alternatively, you can install the following dependencies manually:

* ``numpy``
* ``scipy``
* ``torch``
* ``torchvision``
* ``tqdm``
* ``psutil``
* ``tabulate``

.. code-block:: console

    $ pip/conda install numpy scipy torchvision tqdm psutil
    $ pip install torch/conda install pytorch

To generate figures, install the following additional dependencies:

* ``matplotlib``
* ``seaborn``
* ``statsmodels``
* ``colorcet``

.. code-block:: console

    $ pip/conda install matplotlib seaborn statsmodels colorcet

Finally, to run the ImageNet scripts or use the ``datasets`` module, install ``scikit-learn`` and for the hyperparameter optimization script, install ``scikit-optimize``.

.. code-block:: console

    $ pip/conda install scikit-learn
    $ pip install scikit-optimize/conda install scikit-optimize -c conda-forge

Get Started
===========
A 60-seconds blitz to Laplace approximation (which you can also find `here <https://rmc-github.robotic.dlr.de/humt-ma/curvature/blob/master/curvature/test.py>`_).
For a more detailed example please have a look at the
`Jupyter notebook <https://rmc-github.robotic.dlr.de/humt-ma/curvature/blob/master/curvature/tutorial.ipynb>`_.

.. code-block:: python

    # Standard imports
    import torch
    import torchvision
    import tqdm

    # From the repository
    from fisher import KFAC
    from lenet5 import lenet5
    from sampling import invert_factors

    # Change this to 'cuda' if you have a working GPU.
    device = 'cpu'

    # We will use the provided LeNet-5 variant pre-trained on MNIST.
    model = lenet5(pretrained='mnist', device=device).to(device)

    # Standard PyTorch dataset location
    torch_data = "~/.torch/datasets"
    mnist = torchvision.datasets.MNIST(root=torch_data,
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
    train_data = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)

    # Decide which loss criterion and curvature approximation to use.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    kfac = KFAC(model)

    # Standard PyTorch training loop:
    for images, labels in tqdm.tqdm(train_data):
        logits = model(images.to(device))

        # We compute the 'true' Fisher information matrix (IM),
        # by taking the expectation over the model distribution.
        # To obtain the empirical IM, just use the labels from
        # the data distribution directly.
        dist = torch.distributions.Categorical(logits=logits)
        sampled_labels = dist.sample()

        loss = criterion(logits, sampled_labels)
        model.zero_grad()
        loss.backward()

        # We call 'estimator.update' here instead of 'optimizer.step'.
        kfac.update(batch_size=images.size(0))

    # Access and invert the curvature information to perform Bayesian inference.
    # 'Norm' (tau) and 'scale' (N) are the two hyperparameters of Laplace approximation.
    # See the tutorial notebook for for an in-depth example and explanation.
    factors = list(kfac.state.values())
    inv_factors = invert_factors(factors, norm=0.5, scale=1, estimator='kfac')

.. .. literalinclude:: ../../curvature/test.py

Reproducing the ImageNet results
================================
To reproduce the ImageNet results, first download the `ImageNet ILSVRC 2012 <http://www.image-net.org/download-images>`_
and the `out-of-domain <https://www.kaggle.com/c/painter-by-numbers/data>`_ data. This is required to compute the
IM approximations and in- and out-of-domain evaluations but not for network training, as we work with the pre-trained networks from the ``torchvision`` package.
All scripts use the same standard arguments as well as some script specific arguments.

.. code-block:: console

    $ python curvature/factors.py \
            --torch_dir=<TORCH> \
            --data_dir=<DATA_DIR> \
            --results_dir=<RESULTS> \
            --model=<MODEL> \
            --data=<DATA> \
            --estimator=<ESTIMATOR> \
            --batch_size=<BATCH> \
            --samples=<SAMPLES>

**Standard arguments**

* ``TORCH`` The location where torch datasets and ``torchvision`` model checkpoints are stored. Defaults to ``~/.torch``.
* ``DATA_DIR`` The parent directory of the ImageNet and out-of-domain data. The structure of this directory should look as follows:

.. code-block:: console

    .
    +-- DATA_DIR/
    |   +-- datasets/
        |   +-- imagenet/
            |   +-- data/
                |   +-- train/
                    |   +-- n01440764/
                    |   +-- n01443537/
                    |   +-- ...
                |   +-- val/
                    |   +-- n01440764/
                    |   +-- n01443537/
                    |   +-- ...
        |   +-- not_imagenet/
            |   +-- data/
                |   +-- train/
                    |   +-- img1.jpg
                    |   +-- img2.jpg
                    |   +-- ...

* ``RESULTS`` The location where results should be stored.
* ``MODEL`` The name of a pre-trained `torchvision` model (e.g. ``densenet121`` or ``resnet50``).
* ``DATA`` The dataset to use. Defaults to ``imagenet``.
* ``ESTIMATOR`` Which IM estimator to use. One of ``diag``, ``kfac``, ``efb`` or ``inf``. Defaults to ``kfac``.
* ``BATCH`` The batch size to use. Defaults to ``32``.
* ``SAMPLES`` 1. How many weight posterior samples to draw when evaluating. 2. How many samples to draw from the models output distribution when approximating the IM. Defaults to ``30`` and ``10`` respectively.

**Additional arguments**

* ``--norm`` First hyperparameter of Laplace approximation (``tau``). This times the identity matrix is added to the IM.
* ``--scale`` Second hyperparameter of Laplace approximation (``N``). The IM is scaled by this term.
* ``--device`` One of ``cpu`` or ``gpu``.
* ``--epochs`` Number of iterations over the entire dataset.
* ``--optimizer`` Which optimizer to use when searching for hyperparemeters. One of ``tree`` (random forest), ``gp`` (gaussian process), ``random`` (random search, default) or ``grid`` (grid search).
* ``--rank`` The rank of the INF approximation. Defaults to ``100``.
* ``--verbose`` Get a more verbose printout.
* ``--plot`` Plots results at the end of an evaluation.
* ``--stats`` Computes running statistics during evaluation.
* ``--calibration`` Make a calibration comparison visualization.
* ``--ood`` Make a out-of-domain comparison visualization.

For a complete list of all arguments and their meaning call one of the scripts including ``--help``.

.. code-block:: console

    $ python curvature/factors.py --help

**Example**

This is a short example of a complete computation cycle for ``DenseNet 121`` and the ``KFAC`` estimator.
Keep in mind that many arguments have sensible default values, so we do not need to call all of them explicitly.
This is also true for ``--norm`` and ``--scale``, which are set to the best value found by the hyperparameter
optimization automatically. ``--torch_dir``, ``--data_dir``, ``--results_dir`` and ``--model`` always have to be given though.

.. code-block:: console

    $ python curvature/factors.py --model densenet121 --estimator kfac --samples 1 --verbose
    $ python curvature/hyper.py --model densenet121 --estimator kfac --batch_size 128 --plot
    $ python curvature/evaluate.py --model densenet121 --estimator kfac --batch_size 128 --plot

Once this cycle has been completed for several architectures or estimators, they can be compared using the ``visualization.py`` script.

.. code-block:: console

    $ python curvature/visualize.py --model densenet121 --calibration
    $ python curvature/visualize.py --model densenet121 --ood

To use the INF IM approximation, first compute ``EFB`` (which also computes ``DIAG`` with no additional computational overhead).

.. code-block:: console

    $ python curvature/factors.py --model densenet121 --estimator efb --samples 10 --verbose
    $ python curvature/factors.py --model densenet121 --estimator inf --rank 100

**Hyperparameters**

These are the best hyperparamters for each model and estimator found by evaluating 100 random pairs. Because the IM is typically
scaled by the size of the dataset, the ``scale`` parameter is multiplied by this number. To achieve this,
set the ``--pre_scale`` argument to ``1281166`` when running the ImageNet scripts.

===========  ===========  ============  ===========  ============  ==========  ===========  ==========  ===========
Model          DIAG Norm    DIAG Scale    KFAC Norm    KFAC Scale    EFB Norm    EFB Scale    INF Norm    INF Scale
===========  ===========  ============  ===========  ============  ==========  ===========  ==========  ===========
ResNet18              71           165            1         18916           1       100000         254          206
ResNet50              16          7387           69         25771          11     75113871      145307           60
ResNet152             14     797219512         2765         10162      100000            1      100000            1
DenseNet121        72992            98         2312         12791           4    814681241        1105          146
DenseNet161           19         76911          260         17780          19    708281251      100000            1
===========  ===========  ============  ===========  ============  ==========  ===========  ==========  ===========

Content
=======
A short description of all the modules and scripts in the ``curvature`` directory.

**Main source**

* ``fisher.py`` Implements diagonal, KFAC, EFB and INF IM approximations.
* ``sampling.py`` Damping, inverting and matrix normal sampling.

**ImageNet experiments**

* ``datasets.py`` Unified framework to load standard benchmark datasets.
* ``factors.py`` Various Fisher information matrix approximations.
* ``hyper_factors`` Hyperparameter optimization, including grid and random search as well as Bayesian optimization.
* ``evaluate.py`` Evaluates weight posterior approximations for a chosen model on the ImageNet validation set.
* ``plot.py`` Reliability, entropy, confidence and eigenvalue histograms, inv. ECDF vs. predictive entropy etc. plots.
* ``visualize.py`` Unified visualization of results obtained by running ``evaluate.py``.

**Misc**

* ``utils.py`` Helper functions.
* ``lenet5.py`` Implementation of a LeNet-5 variant for testing.
* ``test.py`` Code featured in the `Get Started`_ section.

Citation
============

If you find this library useful, please cite us in the following ways::

    @inproceedings{lee2020estimating, 
    title={Estimating Model Uncertainty of Neural Networks in Sparse Information Form}, 
    author={Lee, Jongseok and Humt, Matthias and Feng, Jianxiang and Triebel, Rudolph}, 
    booktitle={International Conference on Machine Learning (ICML)}, 
    year={2020}, 
    organization={Proceedings of Machine Learning Research} 
    } 

    @article{humt2020bayesian, 
      title={Bayesian Optimization Meets Laplace Approximation for Robotic Introspection}, 
      author={Humt, Matthias and Lee, Jongseok and Triebel, Rudolph}, 
      journal={arXiv preprint arXiv:2010.16141}, 
      year={2020}
    }
    
    @article{shinde2020learning,
      title={Learning Multiplicative Interactions with Bayesian Neural Networks for Visual-Inertial Odometry},
      author={Shinde, Kashmira and Lee, Jongseok and Humt, Matthias and Sezgin, Aydin and Triebel, Rudolph},
      journal={arXiv preprint arXiv:2007.07630},
      year={2020}
    }


Bibliography
============

.. [1] MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. Neural computation, 4(3), 448-472.
.. [2] Martens, J., & Grosse, R. (2015, June). Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning (pp. 2408-2417).
.. [3] Grosse, R., & Martens, J. (2016, June). A kronecker-factored approximate fisher matrix for convolution layers. In International Conference on Machine Learning (pp. 573-582).
.. [4] Ritter, H., Botev, A., & Barber, D. (2018, January). A scalable laplace approximation for neural networks. In 6th International Conference on Learning Representations, ICLR 2018-Conference Track Proceedings (Vol. 6). International Conference on Representation Learning.
.. [5] George, T., Laurent, C., Bouthillier, X., Ballas, N., & Vincent, P. (2018). Fast approximate natural gradient descent in a kronecker factored eigenbasis. In Advances in Neural Information Processing Systems (pp. 9550-9560).
.. [6] Botev, A., Ritter, H., & Barber, D. (2017, August). Practical gauss-newton optimisation for deep learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 557-565). JMLR. org.
.. [7] Becker, S & Lecun, Y. (1988). Improving the convergence of back-propagation learning with second-order methods. In D. Touretzky, G. Hinton, & T. Sejnowski (Eds.), Proceedings of the 1988 Connectionist Models Summer School, San Mateo (pp. 29-37). Morgan Kaufmann. 
