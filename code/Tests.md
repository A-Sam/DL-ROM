# Tests
## 2d 2d_cylinder_CFD
- cmd:
    ```
    └─(13:05:49 on master ✖ ✹ ✭)──> python3.8 main.py -N 100 -B 31 -d_set 2d_cylinder_CFD --train --simulate --test   1 ↵ ──(Sat,Jul01)─┘
    100 31
    Training will be done on: cuda
    shape u_flat= (2082, 25600)
    Data Loaded in Dataset: 2d_cylinder_CFD with shape 2082
    Training Set Shape= (1561, 80, 320)
    Validation Set Shape= (521, 80, 320)
    Training from scratch UNet3d..
    1451
    411
    ```
-
# TODO
- [ ] Write the algorithm for solving and stacking 10 frames
- [ ] in the algorithm draw and plot the stacking of 10 frames and the alternate stacking of 1 ,maybe compare results at the end
- [ ] explain + plot the PCA via SVD why it helps to show the difference between the gt vs preds
- [ ] add more math regarding stacking and how they are impacting the loss functions
- [ ] explain + plot the results for with some pbm + env settings
  - [ ] 2d cylidner CFD
  - [ ] 2d sq CFD
  - [ ] 2d airfoil
  - [ ] snoaa
- [ ] Transfer Learning
    - [ ] did transfer learning really improved learning time
    - [ ] did transfer learning improve accuracy
- [ ] Comment why scenarios get good accuracy and others high error e.g. for turbulance, spectral bias etc