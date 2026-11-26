# Acknowledgements

AeroGuard AI is maintained in [GargCodesT18/AeroGuard-AI](https://github.com/GargCodesT18/AeroGuard-AI) as a UML501 course-based Machine Learning project.

## Upstream authorship and AeroGuard contributions

The inherited **AEROPREDICT** codebase and dashboard are credited to **Kunwarpreet Singh (`kunwar1908`), `DevAriwala2712` and Team No Signal**. The baseline is commit [`23efd614884e04ace31fcf338157bc1c6b982711`](https://github.com/kunwar1908/AEROPREDICT/commit/23efd614884e04ace31fcf338157bc1c6b982711), from the [upstream repository](https://github.com/kunwar1908/AEROPREDICT), itself a fork of [DevAriwala2712/AEROPREDICT](https://github.com/DevAriwala2712/AEROPREDICT).

AeroGuard improvements include validated four-subset data loading, consistent training-only scaling, reproducible training options, saved engine splits, capped/uncapped evaluation, derived risk metrics, empirical interval scaling, Random Forest/MLP comparisons, executed analysis notebooks and corrected dashboard measurements. This publication includes portable documentation and verified asset organization. Original components and contributions remain attributed. This credit does not imply upstream endorsement of AeroGuard AI or its experimental results.

## Licensing and notices

**Licensing status:** no repository-wide LICENSE or explicit redistribution permission was established for the inherited source. The prior repository audit reported no detected license in either upstream repository. The inherited `package.json` contained `ISC`, without a corresponding license text or established repository-wide scope; that unsupported claim has been removed from this fork's metadata. Obtain permission or a clear applicable upstream license before redistribution. [GitHub explains the effect of absent licensing.](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

The bundled Chart.js notice is preserved in [chart.js.LICENSE.md](aerospace-dashboard/chart.js.LICENSE.md). It applies to that vendor component and does not license the rest of the project.

## Scientific references

- A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository: [NASA PCoE dataset catalog](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- A. Saxena, K. Goebel, D. Simon and N. Eklund (2008), *Damage Propagation Modeling for Aircraft Engine Prognostics*: [NASA technical report](https://ntrs.nasa.gov/citations/20090029214).
- Y. Gal and Z. Ghahramani (2016), *Dropout as a Bayesian Approximation*: [ICML paper](https://proceedings.mlr.press/v48/gal16.html). This motivates MC dropout; it does not confer guaranteed coverage on this implementation.

Model and scaler integrity hashes are recorded in [model-checksums.sha256](docs/model-checksums.sha256).
