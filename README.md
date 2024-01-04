# DGFDBenchmark
This is an open source lib called "DGFDBenchmark" for domain-generalization-based fault diagnosis

This lib is written by JIA Linshan from City University of Hong Kong.

Please cite our paper if this repo is useful for you.
@article{JIA2024106099,
title = {Causal Disentanglement Domain Generalization for time-series signal fault diagnosis},
journal = {Neural Networks},
pages = {106099},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106099},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024000133},
author = {Linshan Jia and Tommy W.S. Chow and Yixuan Yuan},
keywords = {Causal disentanglement, Domain generalization, Time-series fault diagnosis, DGFDBenchmark},
abstract = {Domain generalization-based fault diagnosis (DGFD) presents significant prospects for recognizing faults without the accessibility of the target domain. Previous DGFD methods have achieved significant progress; however, there are some limitations. First, most DGFG methods statistically model the dependence between time-series data and labels, and they are superficial descriptions to the actual data-generating process. Second, most of the existing DGFD methods are only verified on vibrational time-series datasets, which is insufficient to show the potential of domain generalization in the fault diagnosis area. In response to the above issues, this paper first proposes a DGFD method named Causal Disentanglement Domain Generalization (CDDG), which can reestablish the data-generating process by disentangling time-series data into the causal factors (fault-related representation) and no-casual factors (domain-related representation) with a structural causal model. Specifically, in CDDG, causal aggregation loss is designed to separate the unobservable causal and non-causal factors. Meanwhile, the reconstruction loss is proposed to ensure the information completeness of the disentangled factors. We also introduce a redundancy reduction loss to learn efficient features. The proposed CDDG is verified on five cross-machine vibrational fault diagnosis cases and three cross-environment acoustical anomaly detection cases by comparing it with eight state-of-the-art (SOTA) DGFD methods. We publicize the open-source time-series DGFD Benchmark containing CDDG and the eight SOTA methods. The code repository will be available at https://github.com/ShaneSpace/DGFDBenchmark.}


