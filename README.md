# AIM 2025 Low-light RAW Video Denoising Challenge Baseline

This repository provides a baseline method for the [**AIM 2025 Low-light RAW Video Denoising Challenge**](https://www.codabench.org/competitions/8729/), part of the [AIM 2025 workshop](https://cvlai.net/aim/2025/). Using this code, you can train your denoising model on the provided RAW video sequences and generate a submission file ready for evaluation.

## Challenge Overview

Low-light RAW video denoising remains one of the most demanding and understudied tasks in computational imaging. Existing RAW denoising methods focus on well-lit or moderately lit scenes, but real-world applications often involve extreme low-light conditions with severe noise.  
This challenge encourages development of solutions that leverage temporal redundancies across frames while preserving fine spatial details (textures, edges) in raw sensor data. Submissions are evaluated on real-world sequences captured with modern smartphone sensors on a motion stage, paired with high-quality ground truth.

## Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/fearleesy/aim2025-raw-video-denoising.git
cd aim2025-raw-video-denoising

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare the dataset
#    You can find full details and data download instructions on the official challenge page

# 4. Train the baseline model
python3 train.py

# 6. Package submission
python3 submit.py
