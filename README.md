# Hydrogen

C++17 RTL-SDR receiver, `CS16` I/Q recorder, and offline 21-cm Neutral Hydrogen (H I, 1420.40575 MHz) spectral analysis suite.

## Executables

- **`hydrogen_cli`** (Windows & Linux) — command-line recorder (`record`) and offline calibration/fitting pipeline (`analyze`).
- **`hydrogen`** (Windows x64, DirectX 11 + Dear ImGui) — desktop app with live spectrum/waterfall display, timed CS16 recording, and interactive offline analysis.

## Features

- **Live SDR capture**: RTL-SDR hardware support (plus a built-in virtual H I simulator for testing without hardware), lock-free SPSC ring buffer between RX and DSP/disk threads, and lossless `CS16` recording with `.json` / `.txt` metadata sidecars.
- **Offline calibration**: Multi-threaded 50%-overlapped Welch FFT accumulation, LO DC notch excision, MAD-based RFI mitigation, and five continuum baseline solvers (`ripple_filter`, `als`, `poly`, `savitzky_golay`, `standing_wave`).
- **21-cm line fitting**: Single/double/triple Gaussian, Pseudo-Voigt, Gauss-Hermite, spectral moments, and Busy Function models, along with column density ($N_{\text{HI}}$), optical depth, kinetic temperature upper bound, and $W_{50}/W_{20}$ width estimates.
- **Interactive GUI**: Cached Welch PSD so baseline and line-fit parameters update live without re-reading `.cs16` files from disk, with scroll/box zoom, pan, linked 2x2 dashboard axes, and one-click artifact export.
- **Exports**: Five diagnostic PNG plots (`1_hi_spectrum_calibrated.png` through `5_hi_master_dashboard.png`), `calibrated_spectrum.csv`, `analysis_report.json`, and `analysis_report.txt`.

## Building

Requires CMake >= 3.16 and a C++17 compiler (MSVC, GCC, or Clang). On Linux, `librtlsdr` is optional and loaded at runtime for hardware RX. Windows GUI builds require the Windows SDK (DirectX 11).

```bash
cmake -S cpp_suite -B cpp_suite/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_suite/build --parallel
ctest --test-dir cpp_suite/build --output-on-failure
```

Binaries are written to `cpp_suite/bin/`:
- `hydrogen_cli` (`hydrogen_cli.exe` on Windows)
- `hydrogen.exe` (Windows only)

Prebuilt Windows x64 binaries are attached to GitHub Releases.

## CLI Usage

### Record IQ data

```bash
# 60s capture from RTL-SDR hardware
./cpp_suite/bin/hydrogen_cli record --fc 1420.5M --fs 2.4M -d 60 -g 25.4 -o recordings/obs_001.cs16

# 5s capture from the built-in H I simulator
./cpp_suite/bin/hydrogen_cli record --virtual --fc 1420.5M --fs 2.4M -d 5 -o recordings/sim_test.cs16
```

### Analyze a recording

```bash
# Default pipeline (ripple_filter baseline + Gaussian fit; reads .json/.txt sidecars automatically)
./cpp_suite/bin/hydrogen_cli analyze recordings/sim_test.cs16 -o output/sim_test

# ALS baseline + two-component Gaussian fit
./cpp_suite/bin/hydrogen_cli analyze recordings/sim_test.cs16 --baseline als --line-method double_gaussian --tsys 100 -o output/sim_test_2arm

# Standing-wave baseline + three-component Gaussian fit
./cpp_suite/bin/hydrogen_cli analyze recordings/sim_test.cs16 --baseline standing_wave --line-method triple_gaussian --spin-temp 135 -o output/sim_test_3arm
```

## Layout

```text
Hydrogen/
├── README.md
└── cpp_suite/
    ├── CMakeLists.txt
    ├── bin/        # Windows RTL-SDR DLLs & build output
    ├── include/    # C++17 headers
    ├── lib/        # Vendored ImGui, KissFFT, librtlsdr headers, stb_image_write
    ├── src/        # DSP, astrophysics, SDR I/O, plotting, CLI, and DX11 GUI
    └── tests/      # Unit, CLI, E2E, and stress tests
```
