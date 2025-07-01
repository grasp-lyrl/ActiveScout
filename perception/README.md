# Active Perception using Neural Radiance Fields ([Paper](https://arxiv.org/abs/2310.09892))
Authors: Siming He, Christopher D. Hsu∗, Dexter Ong∗, Yifei Simon Shao, Pratik Chaudhari

Forked and adapted from: [Github](https://github.com/grasp-lyrl/Active-Perception-using-Neural-Radiance-Fields)

## Abstract
We study active perception from first principles to argue that an autonomous agent performing active perception should maximize the mutual information that past observations posses about future ones. Doing so requires (a) a representation of the scene that summarizes past observations and the ability to update this representation to incorporate new observations (state estimation and mapping), (b) the ability to synthesize new observations of the scene (a generative model), and (c) the ability to select control trajectories that maximize predictive information (planning). This motivates a neural radiance field (NeRF)-like representation which captures photometric, geometric and semantic properties of the scene grounded. This representation is well-suited to synthesizing new observations from different viewpoints. And thereby, a sampling-based planner can be used to calculate the predictive information from synthetic observations along dynamically-feasible trajectories. We use active perception for exploring cluttered indoor environments and employ a notion of semantic uncertainty to check for the successful completion of an exploration task. We demonstrate these ideas via simulation in realistic 3D indoor environments.


## Citation
```
@inproceedings{siming2024active,
  title={Active perception using neural radiance fields},
  author={He, Siming and Hsu, Christopher D and Ong, Dexter and Shao, Yifei Simon and Chaudhari, Pratik},
  booktitle={2024 American Control Conference (ACC)},
  pages={4353--4358},
  year={2024},
  organization={IEEE}
}
```