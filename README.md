# SpEMSK

A self-navigated framework for motion detection and spoke-reordering musculoskeletal dynamic reconstruction in 3D Radial MRI.

---

## Related Paper

This repository contains the official research demonstration code accompanying our manuscript currently under review at *Medical Image Analysis (MedIA)*:

### **A self-navigated framework for motion detection and spoke-reordering musculoskeletal dynamic reconstruction in 3D Radial MRI**

---

## Authors

**Enping Lin\*, Fatih Calakli, Musa Tunç Arslan, Giovani Schulte Farina, Simon Keith Warfield**

Boston Children's Hospital  
Harvard Medical School

\* Corresponding author

---

## Overview

This repository provides a complete demonstration pipeline for our proposed self-navigated musculoskeletal (MSK) dynamic MRI motion sensing framework based on 3D radial sampling.

The purpose of this package is to clearly demonstrate the complete implementation procedure of the proposed framework described in our MedIA manuscript, including:

- Synthetic non-rigid MSK motion generation
- Golden-angle 3D radial trajectory generation
- Spoke-energy (SpE) motion extraction
- PCA-based motion surrogate generation and different selection
- Dynamic spoke reordering
- Motion-resolved reconstruction
- Quantitative visualization and comparison

To make the entire reconstruction workflow fully reproducible and easy to understand, we also provide our internally developed synthetic MSK motion simulation framework, which can generate different non-rigid motion patterns for evaluating motion-resolved reconstruction methods.

This package is intended to help readers clearly understand how the proposed reconstruction pipeline is implemented step-by-step.

---

## MATLAB Environment

This project was developed and tested using:

```text
MATLAB R2025a
