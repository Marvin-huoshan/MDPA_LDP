# MDPA_LDP

**This repository provides the code for our ACM CCS 2025 paper titled** *Mitigating Data Poisoning Attacks to Local Differential Privacy*

## Description
The goal of **MDPA_LDP** is to mitigate data poisoning attacks on Local Differential Privacy (LDP). By evaluating multiple detection and post-processing methods, it aims to improve the robustness and data privacy protection of the LDP.

## Contents

This repository contains five main Python files and one environment configuration file:


- `Fake_User_Detection.py`: Source code for evaluating different fake user detection methods, including Diffstats and FIAD.
- `Overall_Detection.py`: Source code for evaluating the overall detection method ASD.
- `Recovery.py`: Source code for evaluating different post-processing methods, including RSN, Norm-sub, Base-cut, LDPRecover, and Normalization.
- `running_time.py`: Source code for evaluating the running time of different fake user detection methods.
- `running_time_Overall.py`: Source code for evaluating the running time of the overall detection method.
- `generate_dataset.py`: Source code for generating and preprocessing all evaluation datasets
- `environment.yml`: Environment configuration file describing all required dependencies.

## Running example

### Environment Installation

1. **Install Miniconda** (a tool for managing Python versions and virtual environments):

   - Download the installer for your system from [Anaconda's download page](https://www.anaconda.com/download/success)  
     or use:
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     ```
   - Run the installer (replace `Miniconda3-latest-Linux-x86_64.sh` with the name of your downloaded file):
     ```bash
     sh Miniconda3-latest-Linux-x86_64.sh
     ```
   - Verify the installation:
     ```bash
     conda -V
     ```

2. **Create the Conda Environment:**

   Navigate to the directory containing `environment.yml` and run:
   ```bash
   conda env create -f environment.yml

3. **Activate the environment:**

    ```bash
     conda activate Post_processing_Attack_test
     ```

### Running the Experiments:

1. **Diffstat and FIAD:**
    
    ```bash
    python -m Fake_User_Detection --splits=4 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OUE
   ```
--split: The size of subsets in MGA-A.($r^\prime$)

--h_ao=0: Use APA attack (set to 1 for APA attack).

--vswhat: Iteration parameter for different values of $\epsilon$, $\beta$, or $r$.

--dataset: Specify the dataset to run (e.g., zipf).

--protocol: Specify the protocol to use (e.g., OUE).

Other parameters can be modified in the arg_parse() function.

2. **ASD:**

    ```bash
    python -m Overall_Detection --ratio=0.1 --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=GRR
    ```
--ratio: The ratio of fake users.

Other parameters are similar to those in Fake User Detection.

3. **Attack Recovery of LDP Post-processing:**

    ```bash
    python -m Recovery --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OUE
    ```
   
4. **Running time for Diffstats and FIAD:**
    
    ```bash
    python -m running_time --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=OUE
    ```
   
5. **Running time for ASD:**
    
    ```bash
    python -m running_time_Overall --ratio=0.1 --splits=10 --h_ao=0 --vswhat=epsilon --dataset=zipf --protocol=GRR
    ```
