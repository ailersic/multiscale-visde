# Stochastic Multiscale Models

This repository accompanies the paper "Learning Stochastic Multiscale Models by Variational Inference," currently under consideration for ICML 2025. It contains the training scripts and results for all test cases presented therein, including additional 2D test cases.

All cases were trained with Adam using an initial learning rate of 0.001 and an exponential decay schedule, such that the learning rate decreases by 10% every 1000 optimization steps. Details of training procedure for each case can be found in the "experiments" directory.

## 2D Cylinder Flow

2D cylinder flow dataset, taken from [Guenther et al.](https://cgl.ethz.ch/publications/papers/paperGun17c.php), with domain truncated from 640 x 80 to 320 x 80. Trained for 2000 epochs.
- Original resolution: 320 x 80
- Macroscale resolution: 32 x 8
- Microscale dimension: {0, ..., 5}

#### Baseline closure model (macroscale only): 12.4% error

#### Multiscale model (microscale state dim = 5, visualized below): 4.9% error

<p align="center">
  <img align="middle" src="./images/cylinder_2d/512_5_2000_0.001_1000/test_multiscale.gif" alt="Multiscale 2D Cylinder Flow Model" width="100%"/>
</p>

## 2D Burgers' Equation

2D Burgers' equation multi-trajectory dataset. Trained for 500 epochs.
- Original resolution: 128 x 128
- Macroscale resolution: 8 x 8
- Microscale dimension: {0, ..., 5}

#### Baseline closure model (macroscale only): 12.7% error

#### Multiscale model (microscale state dim = 5, visualized below): 1.3% error

<p align="center">
  <img align="middle" src="./images/burgers_2d/64_5_500_0.001_1000/test_multiscale.gif" alt="Multiscale 2D Burgers Model" width="100%"/>
</p>

## 1D Korteweg - de Vries Equation

Korteweg - de Vries equation multi-trajectory dataset. Trained for 500 epochs.
- Original resolution: 1000
- Macroscale resolution: 20
- Microscale dimension: {0, ..., 5}

#### Baseline closure model (macroscale only, visualized below): 31.7% error

<div align="center">

<table>
  <tr>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_0_500_0.001_1000_2/test_0_0_pred_vs_true.svg" alt="Closure 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.0</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_0_500_0.001_1000_2/test_0_1_pred_vs_true.svg" alt="Closure 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.25</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_0_500_0.001_1000_2/test_0_2_pred_vs_true.svg" alt="Closure 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.5</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_0_500_0.001_1000_2/test_0_3_pred_vs_true.svg" alt="Closure 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.75</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_0_500_0.001_1000_2/test_0_4_pred_vs_true.svg" alt="Closure 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 1.0</sub>
    </td>
  </tr>
</table>

</div>

#### Multiscale model (microscale state dim = 5, visualized below): 5.8% error

<div align="center">

<table>
  <tr>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_5_500_0.001_1000_2/test_0_0_pred_vs_true.svg" alt="Multiscale 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.0</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_5_500_0.001_1000_2/test_0_1_pred_vs_true.svg" alt="Multiscale 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.25</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_5_500_0.001_1000_2/test_0_2_pred_vs_true.svg" alt="Multiscale 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.5</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_5_500_0.001_1000_2/test_0_3_pred_vs_true.svg" alt="Multiscale 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.75</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/kdv_1d/20_5_500_0.001_1000_2/test_0_4_pred_vs_true.svg" alt="Multiscale 1D KdV Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 1.0</sub>
    </td>
  </tr>
</table>

</div>

## 1D Burgers' Equation

1D Burgers' equation multi-trajectory dataset. Trained for 500 epochs.
- Original resolution: 1000
- Macroscale resolution: 20
- Microscale dimension: {0, ..., 5}

#### Baseline closure model (macroscale only, visualized below): 10.5% error

<div align="center">

<table>
  <tr>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_0_500_0.001_1000_3/test_0_0_pred_vs_true.svg" alt="Closure 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.0</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_0_500_0.001_1000_3/test_0_1_pred_vs_true.svg" alt="Closure 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.25</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_0_500_0.001_1000_3/test_0_2_pred_vs_true.svg" alt="Closure 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.5</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_0_500_0.001_1000_3/test_0_3_pred_vs_true.svg" alt="Closure 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.75</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_0_500_0.001_1000_3/test_0_4_pred_vs_true.svg" alt="Closure 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 1.0</sub>
    </td>
  </tr>
</table>

</div>

#### Multiscale model (microscale state dim = 5, visualized below): 1.1% error

<div align="center">

<table>
  <tr>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_5_500_0.001_1000_3/test_0_0_pred_vs_true.svg" alt="Multiscale 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.0</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_5_500_0.001_1000_3/test_0_1_pred_vs_true.svg" alt="Multiscale 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.25</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_5_500_0.001_1000_3/test_0_2_pred_vs_true.svg" alt="Multiscale 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.5</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_5_500_0.001_1000_3/test_0_3_pred_vs_true.svg" alt="Multiscale 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 0.75</sub>
    </td>
    <td align="center" width="20%">
      <img src="./images/burgers_1d/20_5_500_0.001_1000_3/test_0_4_pred_vs_true.svg" alt="Multiscale 1D Burgers Model - Time = 0.0" width="100%" /><br/>
      <sub>Time = 1.0</sub>
    </td>
  </tr>
</table>

</div>
