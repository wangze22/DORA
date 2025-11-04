### Description
This project contains the GAN face generation test code used in the paper, including two-stage on-chip training and image generation for FID evaluation.  
Since the data in the paper was obtained using our ACIM hardware platform, this project only provides simulation code. 
The simulator's computational dataflow is consistent with the hardware platform, offering a comprehensive reference and approximate computational results.

---

### Folder Description

- [`c200_sdk/`](c200_sdk)  
  SDK related to the hardware test platform. Cannot run without the hardware; provided for reference only.

- [`cim_c200_backend/`](cim_c200_backend)  
  Code and dependencies for the simulator.

- [`cim_layers/`](cim_layers)  
  Code and dependencies for the simulator.

- [`cim_qn_train/`](cim_qn_train)  
  Code and dependencies for the simulator.

- [`cim_runtime_simple/`](cim_runtime_simple)  
  Code and dependencies for the simulator.

- [`cim_toolchain_utils/`](cim_toolchain_utils)  
  Code and dependencies for the simulator.

- [`cim_weight_mapper/`](cim_weight_mapper)  
  Code and dependencies for the simulator.

- [`dataset/`](dataset)  
  Contains the training dataset, consisting of 1,000 portrait images at 128×128 resolution.

- [`GAN_online_train/`](GAN_online_train)  
  Path containing the test scripts.

---

### How to Run
The project includes test cases for two architectures:  
1. **DORA** architecture for on-chip training and image generation.  
2. **Conventional ACIM** architecture for on-chip training and image generation, serving as the SOTA baseline.  

The test scripts for these two cases (can be run directly after environment setup) are:  

- [`GAN_online_train/main_DORA.py`](GAN_online_train/main_DORA.py)  
  On-chip training and image generation using the DORA architecture.  

- [`GAN_online_train/main_SOTA_conventional_ACIM.py`](GAN_online_train/main_SOTA_conventional_ACIM.py)  
  On-chip training and image generation using the conventional ACIM architecture.  

Each script takes under a minute to run on an RTX 4080 SUPER.