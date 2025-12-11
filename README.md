# Coda Env
```
conda create -n usps python=3.8 -y
conda activate usps
conda install -c conda-forge glew
```

# Install USPS
``` 
pip install -e .
pip install -r requirements.txt
pip install setuptools==59.5.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
cd USPS/envs/realworldrl_suite
pip install -e . 
cd ../..
```

# Possible issues
- Error: ```AttributeError: module 'setuptools._distutils' has no attribute 'version'```
     - Solution: make sure you're using setuptools 59.5.0
- Error: ```ModuleNotFoundError: No module named 'envs'```
     - Solution: Add USPS to python path (should be implemented in ```train.py``` and ```test.py```)

# Citation

This repository uses code from the following paper. If you use this code in your research, please cite:

```bibtex
@inproceedings{
    zhang2023robust,
    title={Robust Reinforcement Learning in Continuous Control Tasks with Uncertainty Set Regularization},
    author={Yuan Zhang and Jianhong Wang and Joschka Boedecker},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=keAPCON4jHC}
}
```

