# Enhanced-NeoNav
This is the implementation of our RA-L paper [`Towards Target-Driven Visual Navigation in Indoor Scenes via Generative Imitation Learning`](https://arxiv.org/abs/2009.14509), training and evaluation on Active Vision Dataset (depth only). This is an enhanced version of [NeoNav](https://arxiv.org/abs/1906.07207)<br>

## Navigation Model
![](https://github.com/wqynew/NeoNav/raw/master/image/overview.png)
## Implementation
### Training
* The environment: Cuda 10.0, Python 3.6.4, PyTorch 1.0.1 
* Please download "depth_imgs.npy" file from the [AVD_Minimal](https://drive.google.com/file/d/1SmA-3cGwV12XKdGYdsBEJwxf1MYdE6-y/view?usp=sharing) and put the file in the current folder. 
* Our trained model can be downloaded from [HERE](https://drive.google.com/open?id=182D_0hP7orpJKyDDLlUyV4URwT3Rt0Ux).
* To train the navigation model from scratch, use "python3 cnetworkd.py".
    
### Testing
* To evaluate our model, please run "python3 eva_checkpointd1.py".

## Results
<div align="center">
  <table style="width:100%" border="0">
    <thead>
        <tr>
            <th>Start</th>
            <th>End</th>
            <th>Start</th>
            <th>End</th>
        </tr>
    </thead>
    <tbody>
       <tr>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s1.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t1.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s3.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t3.png'></td>
       </tr>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
       </tr>
       <tr>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s2.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t2.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s4.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t4.png'></td>
       </tr>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310007440101_001310000150101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_016_1_001610000060101_001610004220101.gif'></td>
       </tr>
    </tbody>
  </table>
</div>

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



