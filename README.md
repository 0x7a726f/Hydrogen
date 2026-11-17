# Hydrogen Line SDR & Analysis Suite

A C++17 software-defined radio receiver, lossless `CS16` I/Q recorder, and offline 21-cm Neutral Hydrogen (H I, `1420.405751768 MHz`) analysis suite.

## Overview

The suite compiles two primary executables from CMake:

- **`radio_suite_cli`** (Windows & Linux): Headless CLI for live/simulated IQ recording (`record`) and offline calibration/fitting (`analyze`).
- **`radio_suite`** (Windows x64 DirectX 11 + Dear ImGui): Desktop application combining real-time PSD + waterfall monitoring, timed CS16 capture, and interactive spectral analysis.

## Pipeline & Solvers

- **Real-Time RX (`DspPipeline`, `Waterfall`)**:
  - Hann-windowed complex FFT with AVX2/FMA/SSE4.1 acceleration (`SimdDsp`).
  - Linear-power exponential moving average (`P_smooth`) converted to full-scale dBFS (`0 dBFS` = full-scale complex tone), preserving unbiased noise power statistics.
  - Lock-free SPSC ring buffer between SDR capture and DSP/disk threads.
- **Offline Calibration & Artifact Removal (`StreamingProcessor`, `DCExcision`, `RFIFilter`)**:
  - Multi-threaded batched FFT accumulator (`StreamingProcessor`) yielding averaged power spectra and time-frequency waterfall histories.
  - Local oscillator DC notch excision (`DCExcision`).
  - Modified Z-score median-15 MAD RFI mitigation (`RFIFilter`) with H I line protection (`±190 kHz` around `1420.40575 MHz`).
- **Continuum Baseline Solvers (`BaselineFitter`)**:
  - `ripple_filter` (`RippleFilter`): $O(N \log N)$ frequency-domain Gaussian baseline estimator with H I window masking.
  - `als` (`Asymmetric Least Squares`): $O(N)$ macro-binned Whittaker-Eilers pentadiagonal Cholesky solver (`lambda`, `p`).
  - `poly` (`Poly`): Chebyshev-scaled sigma-clipped polynomial fit.
  - `savitzky_golay` (`SavitzkyGolay`): 2-pass outlier-clipped, moment-preserving degree-2 Savitzky-Golay polynomial filter with local-median H I gap anchoring.
  - `standing_wave` (`StandingWave`): Delay-domain FFT coaxial/dish reflection peak detector with sub-bin parabolic frequency refinement and 2-harmonic sinusoidal least-squares excision across line-free channels.
  - Local linear trend removal in linear brightness temperature $T_A$ space (`refine_local`).
- **21-cm Line Profile Models (`Astrophysics`)**:
  - `gaussian_ls`: 3-parameter Levenberg-Marquardt Gaussian fit (`A`, `v0`, `sigma`).
  - `double_gaussian`: 6-parameter two-component Levenberg-Marquardt decomposition with individual arm profiles.
  - `triple_gaussian`: 9-parameter three-component Levenberg-Marquardt decomposition for multi-arm Galactic sightlines (e.g. Local + Perseus + Outer arms or CNM + WNM halos).
  - `voigt`: 4-parameter equal-FWHM Pseudo-Voigt fit (`A`, `v0`, `sigma`, `eta`).
  - `gauss_hermite`: 5-parameter Gauss-Hermite fit (`A`, `v0`, `sigma`, `h3`, `h4`) with numerical FWHM measurement.
  - `moments`: Non-parametric spectral moments ($M_0$, $M_1$, $\sigma_{M_2} = \sqrt{M_2}$) and smoothed half-maximum FWHM.
  - `busy_function`: 6-parameter Generalized Busy Function (Westmeier et al. 2014) modeling flat-topped, double-horned, and asymmetric 21-cm profiles.
- **Derived 21-cm Physical & Kinematic Metrics (`AstrophysicalMetrics`)**:
  - Optically-thin ($N_{\text{HI}}$) and optically-thick corrected ($N_{\text{HI,thick}}$) column densities, peak optical depth $\tau_{\max}$, upper-bound Doppler kinetic temperature $T_{k,\max} = 21.855\,\text{FWHM}^2$, equivalent width $W_{\text{eq}}$, and full velocity widths $W_{50}$ and $W_{20}$.
- **Exports (`Exporter`, `PlotRenderer`)**:
  - Generates 5 diagnostic PNG plots (`1_hi_spectrum_calibrated.png` through `5_hi_master_dashboard.png`), `calibrated_spectrum.csv`, `analysis_report.json`, and `analysis_report.txt`.

## Build & Test (CMake)

### Requirements
- CMake `>= 3.16`, C++17 compiler (`g++`, `clang++`, or MSVC)
- Optional on Linux for hardware RTL-SDR: `librtlsdr` runtime (`librtlsdr.so.0` / `librtlsdr.so`)
- Windows GUI builds require Windows SDK (DirectX 11, Win32)

### Configure, Build, and Run CTest

```bash
cmake -S cpp_suite -B cpp_suite/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_suite/build
ctest --test-dir cpp_suite/build --output-on-failure
```

This builds binaries in `cpp_suite/bin/`:
- `cpp_suite/bin/radio_suite_cli` (or `radio_suite_cli.exe` on Windows)
- `cpp_suite/bin/radio_suite.exe` (Windows only)
- Test executables: `test_suite_state`, `test_end_to_end_bridge`, `test_cli_commands`, `test_performance_stress`, `test_dsp_optimizations`

## CLI Usage (`radio_suite_cli`)

### Record IQ Data
```bash
# Record 60s from RTL-SDR hardware
./cpp_suite/bin/radio_suite_cli record --fc 1420.5M --fs 2.4M -d 60 -g 25.4 -o recordings/obs_001.cs16

# Record 5s from internal virtual H I simulator
./cpp_suite/bin/radio_suite_cli record --virtual --fc 1420.5M --fs 2.4M -d 5 -o recordings/sim_test.cs16
```

### Analyze IQ Data
```bash
# Default pipeline (Ripple baseline + Single Gaussian fit, reads .json/.txt sidecar automatically)
./cpp_suite/bin/radio_suite_cli analyze recordings/sim_test.cs16 -o output/sim_test

# ALS baseline + Double Gaussian 2-arm decomposition
./cpp_suite/bin/radio_suite_cli analyze recordings/sim_test.cs16 --baseline als --line-method double_gaussian --tsys 100 -o output/sim_test_2arm

# Standing-Wave baseline + Triple Gaussian 3-arm decomposition with custom spin temperature
./cpp_suite/bin/radio_suite_cli analyze recordings/sim_test.cs16 --baseline standing_wave --line-method triple_gaussian --spin-temp 135 -o output/sim_test_3arm
```

## Repository Layout

```
Radio/
├── README.md
└── cpp_suite/
    ├── CMakeLists.txt
    ├── bin/                # Windows RTL-SDR DLLs & build output directory
    ├── include/            # Core C++17 headers & GUI panel interfaces
    ├── lib/                # Vendored Dear ImGui, KissFFT, RTL-SDR headers, stb_image_write
    ├── src/                # DSP, astrophysics, SDR I/O, plotting, CLI, and DX11 GUI sources
    └── tests/              # State, E2E bridge, CLI, performance, and DSP/math unit tests
```
