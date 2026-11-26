# Acknowledgements

AeroGuard AI is maintained in [GargCodesT18/AeroGuard-AI](https://github.com/GargCodesT18/AeroGuard-AI) as an individual UML501 course-based Machine Learning project at Thapar Institute of Engineering and Technology.

## Project Contributions

AeroGuard AI focuses on aircraft engine predictive maintenance and Remaining Useful Life (RUL) estimation. The project incorporates:

- Validated data loading across all four NASA C-MAPSS subsets.
- Exploratory data analysis and sensor visualization.
- Feature engineering and leakage-aware preprocessing.
- Random Forest, MLP and LSTM comparative analysis.
- Reproducible training and engine-wise evaluation.
- Capped and uncapped RUL performance analysis.
- Engine-risk classification and uncertainty calibration.
- Interactive dashboard and local model inference.
- Comprehensive experimental documentation and visualization.

## Technical Resources and Notices

The project utilizes publicly available machine learning resources, existing software components, scientific literature and established libraries, including PyTorch, scikit-learn, NumPy, Pandas, Flask and Chart.js.

Applicable third-party notices and original authorship rights are respected. Detailed provenance records are maintained separately. No repository-wide license is asserted.

The bundled Chart.js license notice is available in [chart.js.LICENSE.md](aerospace-dashboard/chart.js.LICENSE.md).

## Scientific References

- A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*. [NASA PCoE Dataset Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- A. Saxena, K. Goebel, D. Simon and N. Eklund (2008), *Damage Propagation Modeling for Aircraft Engine Prognostics*. [NASA Technical Report](https://ntrs.nasa.gov/citations/20090029214).
- L. Breiman (2001), *Random Forests*. Machine Learning.
- S. Hochreiter and J. Schmidhuber (1997), *Long Short-Term Memory*. Neural Computation.
- Y. Gal and Z. Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. [ICML Paper](https://proceedings.mlr.press/v48/gal16.html).

## Academic Purpose

AeroGuard AI is intended for academic learning, comparative experimentation and practical demonstration of Machine Learning techniques. The experimental results are based on simulated engine telemetry and are not intended for certified aviation maintenance decisions.

## Reproducibility

Model and scaler integrity hashes are recorded in [model-checksums.sha256](docs/model-checksums.sha256). Executed notebooks, evaluation metrics, trained models and supporting visualizations are maintained within the project repository.
