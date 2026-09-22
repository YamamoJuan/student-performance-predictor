# Data

## Source

- **Dataset**: UCI Machine Learning Repository — Student Performance (ID 320)
- **URL**: https://archive.ics.uci.edu/dataset/320/student+performance
- **File used**: `student-mat.csv` (Mathematics course)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation**: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

Original paper: P. Cortez and A. M. G. Silva, "Using Data Mining to Predict Secondary School Student Performance", 2008.

## Description

The dataset describes student achievement in secondary education at two Portuguese schools.
It contains 395 students and 33 attributes combining grades, demographics, and school/lifestyle
questionnaire responses. There are no missing values.

Two period grades (`G1`, `G2`) are intentionally **excluded** from modeling: they are earlier
grades in the same course and are near-copies of the final grade `G3`, so using them would be
data leakage (they are unavailable at the time a real prediction of the final grade would be made).

## Files

- `student-mat.csv` — raw dataset as downloaded from UCI (semicolon-separated).
