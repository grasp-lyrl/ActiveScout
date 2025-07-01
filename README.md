# Active Scout: Multi-Target Tracking Using Neural Radiance Fields in Dense Urban Enviroments ([Paper](https://arxiv.org/abs/2406.07431))
Authors: Christopher D. Hsu and Pratik Chaudhari

## Abstract
We study pursuit-evasion games in highly occluded urban environments, e.g. tall buildings in a city, where a scout (quadrotor) tracks multiple dynamic targets on the ground. We show that we can build a neural radiance field (NeRF) representation of the city—online—using RGB and depth images from different vantage points. This representation is used to calculate the information gain to both explore unknown parts of the city and track the targets—thereby giving a completely
first-principles approach to actively tracking dynamic targets. We demonstrate, using a custom-built simulator using Open Street Maps data of Philadelphia and New York City, that we can explore and locate 20 stationary targets within 300 steps. This is slower than a greedy baseline, which does not use active perception. But for dynamic targets that actively hide behind
occlusions, we show that our approach maintains, at worst, a tracking error of 200m; the greedy baseline can have a tracking error as large as 600m. We observe a number of interesting
properties in the scout’s policies, e.g., it switches its attention to track a different target periodically, as the quality of the NeRF representation improves over time, the scout also becomes better in terms of target tracking.

## Example run in Philadelphia map

https://github.com/user-attachments/assets/37e5dc87-cee3-43ef-a13d-49c0e7fbadf9

## Installation

### 1. Set up Environment

```bash
conda create -n scout python=3.10
conda activate scout
python -m pip install --upgrade pip
```

### 2. Install PyTorch 2.0.1 with CUDA 11.8

```bash
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

### 3. Install tiny-cuda-nn

```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### 4. Install Other Requirements

```bash
conda install scikit-image scikit-learn PyYAML imageio tqdm scipy rich
pip install matplotlib opencv-python moderngl-window lpips pyqt6 osmium ipdb imgui pywavefront seaborn
```

### 5. Install rotorpy

```bash
pip install -e planning/rotorpy
```

## OpenStreetMaps and OSM2WORLD for a Custom  Map

**This portion is optional as we have provided the maps from the paper in `data/` **

1. **Install Java Runtime Environment (JRE) 17:**
   ```bash
   sudo apt update
   sudo apt install openjdk-17-jre
   ```

2. **Download OSM2WORLD:**  
   [OSM2WORLD Download Page](https://osm2world.org/download/)

3. **Open the Java Application:**
   ```bash
   java -jar OSM2World.jar
   ```

4. **Download a snippet of OSM:**  
   [OpenStreetMap Export](https://www.openstreetmap.org/export#map=16/39.9539/-75.1692)

5. **Import .osm data into OSM2WORLD and export as .obj file.**  
   - You can edit `<standard.properties>` for custom color textures.
   - Note the `coord_origin` in the generated .obj and update your scene params.
   - If OSM2WORLD says the map is too big, you can proceed, but downstream processes may be slow.

6. **(Optional) Use Blender to get surface normals for the .obj file.**  
   - Skipping this step will result in rendering without light properties.

7. **Convert .osm data to protobuf format for use with pyrosm and geodataframes:**
   ```bash
   osmium cat -o data/osm/philly_cc/philly_cc.osm.pbf data/osm/philly_cc/philly_cc.osm
   ```

## Configurations

1. **Update experiment configs in `scene_config/`.**
2. **Update NeRF training configs in `perception/nerf_scripts/configs/`,**

## Running an experiment

1. **Run the pipeline without NeRF using the ground truth map:**
   ```bash
   python scripts/pipeline.py --map_name <philly_cc|nyc_stuy>
   ```

2. **Run the pipeline while training a NeRF:**
   ```bash
   python scripts/nerfpipeline.py --map_name <philly_cc|nyc_stuy>
   ```

## Citation
```
@INPROCEEDINGS{10802565,
  author={Hsu, Christopher D. and Chaudhari, Pratik},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Active Scout: Multi-Target Tracking Using Neural Radiance Fields in Dense Urban Environments}, 
  year={2024},
  pages={10399-10406},
  keywords={Target tracking;Urban areas;Buildings;Games;Neural radiance field;Information filters;Probabilistic logic;Mutual information;Intelligent robots;Quadrotors},
  doi={10.1109/IROS58592.2024.10802565}}

or

@misc{hsu2024activescoutmultitargettracking,
      title={Active Scout: Multi-Target Tracking Using Neural Radiance Fields in Dense Urban Environments}, 
      author={Christopher D. Hsu and Pratik Chaudhari},
      year={2024},
      eprint={2406.07431},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2406.07431}, 
}
```
