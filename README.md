# Tractian Condition Monitoring Challenge

This project aimed to solve two conditions monitoring problems proposed by a TRACTIAN Challenge.
- Lubricant issues detection on bearings based on carpet noise identification
- Motor base bolt loseness detection

---

## Part 1. Lubricant issues detection

The fist problem involved the detection of lubricant issues that can cause distributed wear on the bearings. 

This anomalous condition manifests in the vibration spectrum by generating random noise (background noise), typically in the form of a 'carpet' as it approaches the natural frequencies. Carpet patterns can be perceived as a series of spectral peaks that are randomly close to each other, and its detection is of great importance in order to diagnose lubrication problems. Figure 1 shows an example of a carpet region, along with an illustration of what does not qualify as a carpet—namely, a series of regularly spaced spectral peaks. 

![Carpet Example Figure](assets/Carpet_example.png)

Figure 1. Example of a carpet region. (from TRACTIAN)

In order to solve detect carpet noises in the unlabelled data provided, this project proposed the human in the loop pipeline approach described bellow.

At which a unsupervised machine-learning techinique called Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is used to find high amplitude frequency bands located near each other that could be considered carpet noise regions.These initial regions are filtered using a relative energy based filter, which compares the root mean squared (RMS) of the frequency band selected with the overall signal RMS. If the RMS surpass a high percentage threshold previously defined, the final selected regions are classified as carpet regions.

-This unsupervised pipeline then generates:
- plots of each wave's carpet region selection process;

- a csv file with features calculated from all regions from a wave signal, wheter they are considered carpet regions or not;

- a plot of these features for all regions, highlighting the regions chosen

-In posession of this files an analyst can further refine the selected regions to train a supervised model approach.



## Part 2. Bolt loseness detection

The second problem involved the detection of losenes in bolts holding a motor to its base. Looseness is a common fault condition that can lead to excessive vibration. Structural looseness typically involves loose bolts in non-rotating components. It is important to address this issue promptly to prevent the development of additional failures caused by the resulting vibrations. 

-Here it was proposed a pipeline for diferent ML model selections:
The code available in part_2_train folder was used to generate data and train the different models using the parameters avalible from part_2 config.yaml

│-- train_part2/
│   │-- main.py
│   │-- utils.py
│   │-- pipeline.py
│   │-- models.py

-And feature extraction based on the following highly used vibration metrics were used: 
- Root Mean Squared (RMS)
- High pass RMS
- Peak amplitude
- Crest value
- Zero Crossing Rate
- Kurtosis

-This metrics were used both for acceleration and velocity signals. It was observed that although trends are not easily sppoted for acceleration signals the introduction of velocity is of great help in loseness condition monitoring.

---

## 3. Project Structure

The project is structured is shown bellow.It is structured in a way that each part of the challenge has one main file that holds the proposed solution. In the case of the first part, this is the Unsupervised_Carpet_predictor.py file and in the case of the second part it is the LosenessDetection.py file.Each file has its specific Part_N_config.yaml that holds hyperparameters and folders location holder to run the file.

```text
project/
│-- data/
│   │-- part_1/
│   │-- part_2/
│-- figures/
│-- train_part2/
│   │-- main.py
│   │-- utils.py
│   │-- pipeline.py
│   │-- models.py
│-- reports/
│   │-- Report_part_1.md
│   │-- Report_part_2.md
│-- Unsupervised_Carpet_Predictor.py
│-- Part1_config.yaml
│-- Loseness_Detection.py
│-- Part2_config.yaml
│-- README.md
│-- requirements.txt

In order to run those files it is first necessary to install the requirements availble in requirements.txt.

''' pip install requirements.txt