<div align=center>
  <h1>
    Gaussian Splatting: Point-Based Radiance Fields
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs479-spring-2025/ target="_blank"><b>KAIST CS479: Machine Learning for 3D Data (Spring 2025)</b></a><br>
    Programming Assignment 4
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://dvelopery0115.github.io target="_blank"><b>Seungwoo Yoo</b></a>  (dreamy1534 [at] kaist.ac.kr)      
  </p>
</div>

<div align=center>
  <img src="./media/teaser.gif" width="400"/>
</div>

#### Due: TBD, 23:59 KST
#### Where to Submit: KLMS

## Abstract

Following the success of [Neural Radiance Fields (NeRF)](https://arxiv.org/abs/2003.08934) in novel view synthesis using implicit representations, researchers have actively explored adapting similar concepts to other 3D graphics primitives.
The most successful among them is [Gaussian Splatting (GS)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), a method based on a point-cloud-like representation known as Gaussian Splats.

Unlike simple 3D points that encode only position, Gaussian Splats store local volumetric information by associating each point with a covariance matrix, modeling a Gaussian distribution in 3D space.
These splats can be efficiently rendered by projecting and rasterizing them onto an image plane, enabling real-time applications as demonstrated in the paper.

In this assignment, we will explore the core principles of the Gaussian Splat rendering algorithm by implementing its key components.
As in our previous assignment on NeRF, we strongly encourage you to review the paper beforehand or while working on this assignment.

<details>
<summary><b>Table of Content</b></summary>
  
- [Abstract](#abstract)
- [Setup](#setup)
- [Code Structure](#code-structure)
- [Tasks](#tasks)
  - [Task 0. Download Data](#task-0-download-data)
  - [Task 1. World to NDC](#task-1-world-to-ndc)
  - [Task 2. Covariance Matrix Projection](#task-2-covariance-matrix-projection)
  - [Task 3. Volume Rendering of Projected Splats](#task-3-volume-rendering-of-projected-splats)
  - [Task 4. Qualitative \& Quantitative Evaluation](#task-4-qualitative--quantitative-evaluation)
- [What to Submit](#what-to-submit)
- [Grading](#grading)
- [Further Readings](#further-readings)
</details>

## Setup

To get started, clone this repository first.
```
git clone --recursive {PROJECT_URL}
```

We recommend creating a virtual environment using `conda`.
To create a `conda` environment, issue the following command:
```
conda create --name cs479-gs python=3.10
```
This should create a basic environment with Python 3.10 installed.
Next, activate the environment and install the dependencies using `pip`:
```
conda activate cs479-gs
```
The remaining dependencies are the ones related to PyTorch and they can be installed with the command:
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torchmetrics[image]
pip install imageio[ffmpeg]
pip install plyfile tyro==0.6.0 jaxtyping==0.2.37 typeguard==2.13.3
pip install simple-knn/.
```

Register the project root directory (i.e., `gs_renderer`) as an environment variable to help the Python interpreter search our files.
```
export PYTHONPATH=.
```

By default, the configuration is set to render `lego` scene. You can select different scenes by altering argument `Args` in `render.py`. Run the following command to render the scene:
```
python render.py
```
For now, running this command will result in an error, as the Gaussian Splat files have not been downloaded yet.  

All by-products made during rendering, including images, videos, and evaluation results, will be saved in an experiment directory under `outputs/{SCENE NAME}`.


## Code Structure
This codebase is organized as the following directory tree. We only list the core components for brevity:
```
gs_renderer
│
├── data                <- Directory for data files.
├── src
│   ├── camera.py       <- A light-weight data class for storing camera parameters.
│   ├── renderer.py     <- Main renderer implementation.
│   ├── scene.py        <- A light-weight data class for storing Gaussian Splat parameters.
│   └── sh.py           <- A utility for processing Spherical Harmonic coefficients.
├── render.py           <- Main script for rendering.
└── README.md           <- This file.
```

## Tasks

### Task 0. Download Data

Download the scene files (`data.zip`) from [here](https://drive.google.com/file/d/16z6kmnPgvPN-HVu0TpCloxvygUrjCzu6/view?usp=sharing) and extract them into the root directory.
After extraction, the `data` directory should be structured as follows:
```
data
│
├── cam_data.npz        <- Camera parameters.
├── chair.ply           <- "Chair" Scene.
├── drums.ply           <- "Drums" Scene.
├── ficus.ply           <- "Ficus" Scene.
├── hotdog.ply          <- "Hotdog" Scene.
├── lego.ply            <- "Lego" Scene.
├── materials.ply       <- "Materials" Scene.
├── mic.ply             <- "Mic" Scene.
└── ship.ply            <- "Ship" Scene.
```

### Task 1. World to NDC

Implement the coordinate transformation from world space to normalized device coordinates (NDC) in the `project_ndc` method of `renderer.py`.  

Given a homogeneous coordinate $\mathbf{p}$, performing the following matrix multiplication yields the coordinate of the point in the view space:

```math
\mathbf{p}_{\text{view}} = \mathbf{p} \mathbf{V}
```

where $\mathbf{V}$ is the view matrix (world-to-camera transformation).

To project the point onto the image plane, first perform the matrix multiplication

```math
\mathbf{p}_{\text{proj}} = \mathbf{p}_{\text{view}} \mathbf{P}
```

where $\mathbf{P}$ is the projection matrix (camera-to-clip transformation).
Then, divide the first three components of $\mathbf{p}_{\text{ndc}}$ by the fourth component to obtain the normalized device coordinates:

```math
\tilde{\mathbf{p}} = \frac{\mathbf{p}_{\text{proj}}}{\mathbf{p}_{\text{proj}, w}}
```

where $\mathbf{p}\_{\text{proj}, w}$ is the fourth component of $\mathbf{p}\_{\text{proj}}$.

Lastly, compute the binary mask indicating the points that are behind the near plane by checking whether the $z$-coordinate of $\mathbf{p}\_{\text{view}}$ is greater than $z\_{\text{near}}$.

### Task 2. Covariance Matrix Projection

Implement the projection of the covariance matrix onto the image plane in the `compute_cov_2d` method of `renderer.py`.

You are only allowed to modify the code inside the block marked with `TODO` in the method.
After transforming the centers of 3D Gaussian splats to the camera space, compute the Jacobian matrix of the world-to-camera and projective transformations.
Specifically, we can use the Jacobian matrix $\mathbf{J}$ of form:

```math
\mathbf{J} = \begin{bmatrix}
  \frac{f_x}{t_z} &      0          & -\frac{f_x t_x}{t_z^2} \\
  0               & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \\
  0               &      0          & 0
\end{bmatrix},
```

where $f\_x$ and $f\_y$ are the focal lengths, and $t\_x$, $t\_y$, and $t\_z$ represent the center coordinates of the 3D Gaussians in camera space.
We provide a tensor `J` initialized with zeros of the correct shape. You need to fill in the correct values in the tensor.

Next, compute the covariance matrix in the image plane by projecting the world-space covariance matrix using the Jacobian matrix:

```math
\boldsymbol{\Sigma}_{\text{2D}} = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma}_{\text{3D}} \mathbf{W}^T \mathbf{J}^T
```

where $\mathbf{W}$ is the rigid transformation componenet of the camera space to world space transformation.

### Task 3. Rendering Equation of Point-Based Radiance Fields

Finally, implement the rendering equation for point-based radiance fields in the `render` method of `renderer.py`, which computes pixel colors by blending the colors of 2D Gaussian splats stacked on the image plane.
Note that we assume that the center coordinates and covariance matrices of 2D Gaussian splats lie in the image space.

Due to memory constraint, the renderer divides an image into multiple tiles and processes each tile separately.
For each tile, the renderer computes the 2D Gaussian splats projected onto the tile and accumulates the colors of the splats.
The provided skeleton already implements this process, and you can use `in_mask` to identify the splats that should be used for rendering the current tile.

Implement the following four steps in the `render` method at the locations marked as `TODO`:

First, sort the Gaussians in ascending order based on their depth.  
Next, Compute the displacement vector $\mathbf{d}\_{i,j} \in \mathbb{R}^2$ between the center of $i$-th pixel in the current tile and the $j$-th Gaussian splat indicated by `in_mask`.  
Compute the Gaussian weight at the pixel center by evaluating the Gaussian distribution at the displacement vector:
  ```math
    \mathbf{w}_{i,j} = \exp (-\frac{1}{2} \mathbf{d}_{i,j}^T  \Sigma_{j}^{-1}  \mathbf{d}_{i,j} )
  ```
  where $\Sigma\_{j}$ is the covariance matrix of the $j$-th 2D Gaussian splat.

Lastly, Perform alpha blending to accumulate the colors of the splats, using the product of opacities and Gaussian weights from Step 3 to determine the final pixel colors. The color of the $i$-th pixel is computed as:
  ```math
  \mathbf{C}_i = \sum_{j} \mathbf{c}_j \tilde{\alpha}_j \Pi_{k < j} (1 - \tilde{\alpha}_k),
  ```
  where $\mathbf{c}\_j$ is the color of the $j$-th Gaussian splat, $\tilde{\alpha}\_j = \mathbf{w}\_{i,j} \alpha_{j}$ is the product of the Gaussian weight $\mathbf{w}\_{i,j}$ and the opacity $\alpha\_j$ of the $j$-th splat. Intuitively, the contribution of the $j$-th Gaussian splat depends on:
  1. Proximity: how close the $i$-th pixel is to the splat center in the image space, and
  2. Opacity: how opaque the splat is.

### Task 4. Qualitative \& Quantitative Evaluation

TBD.

> :bulb: **For details on grading, refer to section [Grading](#grading).**

## What to Submit

Compile the following files as a **ZIP** file named `{NAME}_{STUDENT_ID}.zip` and submit the file via Gradescope.
  
- The folder `gs_renderer` that contains every source code file;
- A folder named `{NAME}_{STUDENT_ID}_renderings` containing the rendered images (`.png` files) used for computing evaluation metrics;
- A text file named `{NAME}_{STUDENT_ID}.txt` containing **a comma-separated list of LPIPS, PSNR, and SSIM** from quantitative evaluation;

## Grading

**You will receive a zero score if:**
- **you do not submit,**
- **your code is not executable in the Python environment we provided, or**
- **you modify any code outside of the section marked with `TODO`.**
  
**Plagiarism in any form will also result in a zero score and will be reported to the university.**

**Your score will incur a 10% deduction for each missing item in the [Submission Guidelines](#submission-guidelines) section.**

Otherwise, you will receive up to **TODO** points from this assignment that count toward your final grade.

| Evaluation Criterion | LPIPS (↓) | PSNR (↑) | SSIM (↑) |
|---|---|---|---|
| **Success Condition \(100%\)** |  |  |  |
| **Success Condition \(50%)**   |  |  |  |

As shown in the table above, each evaluation metric is assigned up to 100 points. In particular,
- **LPIPS**
  - You will receive 100 points if the reported value is equal to or, *smaller* than the success condition \(100%)\;
  - Otherwise, you will receive 50 points if the reported value is equal to or, *smaller* than the success condition \(50%)\.
- **PSNR**
  - You will receive 100 points if the reported value is equal to or, *greater* than the success condition \(100%)\;
  - Otherwise, you will receive 50 points if the reported value is equal to or, *greater* than the success condition \(50%)\.
- **SSIM**
  - You will receive 100 points if the reported value is equal to or, *greater* than the success condition \(100%)\;
  - Otherwise, you will receive 50 points if the reported value is equal to or, *greater* than the success condition \(50%)\.

## Reference

- [torch-splatting](https://github.com/hbb1/torch-splatting)
