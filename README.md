# Tractian Condition Monitoring Challenge

This project aimed to solve two conditions monitoring problems proposed by a TRACTIAN Challenge.
- Lubricant issues detection on bearings
- Motor base bolt loseness detection

---

## Part 1.

The fist problem involved the detection of lubricant issues that can cause distributed wear on the bearings. 

This anomalous condition manifests in the vibration spectrum by generating random noise (background noise), typically in the form of a 'carpet' as it approaches the natural frequencies. Carpet patterns can be perceived as a series of spectral peaks that are randomly close to each other, and its detection is of great importance in order to diagnose lubrication problems. Figure 1 shows an example of a carpet region, along with an illustration of what does not qualify as a carpet—namely, a series of regularly spaced spectral peaks. 

![Carpet Example Figure](assets/Carpet_example.png)

## 1. Overview

Provide a slightly more detailed explanation:
- Context / motivation
- High-level approach
- Expected outcome

---

## 2. Methodology

Describe:
- Data sources
- Preprocessing
- Feature engineering
- Models / algorithms used
- Evaluation strategy

---

## 3. Results

Explain your findings.

### Example Figure

![Example Figure](figures/example_plot.png)

Brief explanation of what the figure shows and why it matters.

You can add multiple figures:

![Another Figure](figures/another_plot.png)

---

## 4. Project Structure

```text
project/
│-- data/
│-- figures/
│-- src/
│   │-- main.py
│   │-- utils.py
│-- notebooks/
│-- README.md
│-- requirements.txt