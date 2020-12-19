# Enhanced-NeoNav
This is the implementation of our RA-L paper [`Towards Target-Driven Visual Navigation in Indoor Scenes via Generative Imitation Learning`](https://arxiv.org/abs/2009.14509), training and evaluation on Active Vision Dataset (depth only). This is an enhanced version of [NeoNav](https://arxiv.org/abs/1906.07207)<br>


## Implementation
### Training
* The environment: Cuda 10.0, Python 3.6.4, PyTorch 1.0.1 
* Please download "depth_imgs.npy" file from the [AVD_Minimal](https://drive.google.com/file/d/1SmA-3cGwV12XKdGYdsBEJwxf1MYdE6-y/view?usp=sharing) and put the file in the current folder. 
* Our trained model can be downloaded from [HERE](https://drive.google.com/open?id=182D_0hP7orpJKyDDLlUyV4URwT3Rt0Ux).
* To train the navigation model from scratch, use "python3 cnetworkd.py".
    
### Testing
* To evaluate our model, please run "python3 eva_checkpointd1.py".

## Contact
To ask questions or report issues please open an issue on the [issues tracker](https://github.com/wqynew/Enhanced-NeoNav/issues).
## Citation
If you use this work in your research, please cite the paper:
```
@article{wu2020towards,
  title={Towards target-driven visual navigation in indoor scenes via generative imitation learning},
  author={Wu, Qiaoyun and Gong, Xiaoxi and Xu, Kai and Manocha, Dinesh and Dong, Jingxuan and Wang, Jun},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={1},
  pages={175--182},
  year={2020},
  publisher={IEEE}
}
```



