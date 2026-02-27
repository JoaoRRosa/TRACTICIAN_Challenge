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



The code available in the train_part2 folder was used to generate the dataset and train the different models using the hyperparameters defined in Part2_config.yaml.

train_part2/
│-- main.py
│-- utils.py
│-- pipeline.py
│-- models.py
Feature Extraction

Feature extraction was based on widely used vibration metrics:

Root Mean Square (RMS)

High-pass RMS

Peak amplitude

Crest factor

Zero Crossing Rate

Kurtosis

These metrics were computed for both acceleration and velocity signals.

Although trends are not always easily observed in acceleration signals alone, the inclusion of velocity-based features significantly improves looseness condition monitoring.

##3. Project Structure

The project is organized so that each part of the challenge has a main file containing the proposed solution:

Part 1 → Unsupervised_Carpet_Predictor.py

Part 2 → Loseness_Detection.py

Each part also has its own configuration file (Part1_config.yaml and Part2_config.yaml) containing hyperparameters and folder paths required to run the corresponding script.

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
Installation

Before running the scripts, install the required dependencies listed in requirements.txt:

pip install -r requirements.txt

If you want, I can also:

Improve the technical tone (make it more academic/formal)