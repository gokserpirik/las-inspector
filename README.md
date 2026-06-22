# Automated LiDAR Quality Control Inspector (las-inspector) 🔍

A lightweight Python CLI tool designed to ingest standard ASPRS LAS/LAZ point cloud files, run automated quality control checks, and generate interactive, client-ready HTML reports complete with spatial density heatmaps and elevation profiles.

> ℹ️
> This is a portfolio demonstration and proof of concept. It is not intended for commercial engineering, large-scale production, or high-precision survey audits.

> ⚠️
> This repository serves as a showcase of using AI as a software engineering force multiplier, contrasting a basic prototyping script (`main.py`) with a custom, high-performance binary parsing CLI tool (`las_inspector.py`). The code and documentation were developed in partnership with AI to demonstrate low-level byte parsing and rapid CLI tool optimization.

---

## 📊 Dataset Credit & Sourcing

The sample dataset (`las2018.laz`) used for development was sourced from the [USGS National Map Lidar Explorer](https://viewer.nationalmap.gov/basic/).
* **Format:** `.laz` (Compressed LiDAR)
* **Scale Precision:** Millimeter (0.001m precision)

To run the pipeline with your own data:
1. Download a tile in `.las` or `.laz` format from the USGS portal.
2. Save it to the working directory.
3. Pass the file path as an argument to `las_inspector.py`.

---

## 🤔 Why Use This?

While professional civil engineering projects typically rely on feature-rich CAD suites (like Autodesk Civil 3D) or GIS software (like ArcGIS Pro) to inspect LiDAR files, this lightweight Python CLI workflow offers key practical advantages for specific use cases:

1. **Zero Licensing Cost:** It uses free, open-source libraries (`numpy`, `matplotlib`, `lazrs`). Anyone can inspect point clouds without purchasing expensive proprietary CAD/GIS software licenses.
2. **Speed & Low Overhead:** Heavy CAD programs can take minutes to load massive datasets and often freeze on standard laptops. This script loads, parses headers, and runs QC checks in seconds using custom binary decoding and memory-efficient NumPy arrays.
3. **Easy Automation & CI/CD Pipelines:** Because it is a simple terminal command-line tool, it can easily be scheduled (e.g., cron jobs), integrated into automated file pipelines, or run as an automated check (pre-commit hooks or GitHub Actions) on newly received survey tiles.
4. **Transparency & Customizability:** All QC thresholds (like density limits and classification ratios) are explicitly written in code. This provides a clear, reproducible calculation logic rather than a "black-box" proprietary algorithm.

---

## 📋 Sample Scenario: Quality Control (QC) Delivery Auditing

### The Context
A civil engineering firm is working on a municipal subdivision project. Every week, a drone surveying subcontractor delivers raw LiDAR point cloud tiles (`.laz` format) covering different sections of the project site.

### The Problem
* The firm must verify that the point cloud files meet the contractually required specifications (e.g., density of at least 2.0 pts/m², sufficient ground classification for digital terrain modeling, and no corrupt headers).
* Manually loading each tile into heavy GIS software to inspect metadata and run statistics takes hours of manual click-work every week.
* If a contractor submits a tile with incorrect classification or low density, the error might not be discovered until weeks later during detailed modeling, causing costly project delays.

### The Solution with this Script
1. **Data Collection:** The subcontractor uploads the weekly survey tiles to a shared directory.
2. **Execution:** An automated script triggers `python las_inspector.py [filename].laz --density` for each incoming file.
3. **Result:** The script automatically parses the binary structure, validates the file against density and classification heuristics, and outputs a clean text summary along with a standalone `qc_report.html` featuring embedded density and elevation plots.
4. **Value:** The project engineer receives an automated notification with the QC verdict (PASSED/FAILED) and a single-file interactive report. Subcontractor deliveries can be audited and accepted or rejected instantly, ensuring only high-quality data enters the design pipeline.

---

## 🛠️ How it Works

1. **Low-Level Header Decoding:** Reads the public header block of the LAS/LAZ file to extract scale factors, offsets, coordinate bounds, point records count, and coordinate reference system (CRS) WKT data.
2. **Binary Point Record Parsing:** Bypasses heavy wrapper libraries by decoding point records directly using Python's `struct` module and NumPy buffers to unpack X, Y, Z coordinates, intensities, return numbers, and classification bytes.
3. **Dynamic LAZ Decompression:** Transparently detects compressed `.laz` files and reads the LASzip VLR (Variable Length Record), using high-speed Rust-based `lazrs` bindings to decompress point data on the fly.
4. **Heuristic QC Warning Engine:** Scans point flags and statistics to trigger warnings/errors for low point density, lack of ground-classified points, withheld/synthetic flags, or potential flight-line overlap zones.
5. **Asset-Embedded Report Compilation:** Renders an interactive, responsive HTML5 report styled with a dark theme. It base64-encodes the generated density and elevation plots directly into the HTML file, creating a single, fully portable deliverable.

### 🧠 AI Co-Pilot Case Study: Baseline vs. Enhanced

This repository serves as a showcase of using AI as a force multiplier for software engineering, contrasting a basic prototyping script with a custom, high-performance binary parsing CLI tool.

| Feature | Baseline Script (`main.py`) | Enhanced CLI Tool (`las_inspector.py`) |
| :--- | :--- | :--- |
| **Dependencies** | Requires `laspy` high-level wrapper | Native binary decoding (uses Python `struct` & `numpy` buffers) |
| **Parsing Logic** | Wrapped abstraction | Low-level binary reading from file offsets (IEEE 754 doubles, packed bytes) |
| **Visualizations** | None | 2D Density Heatmap & Elevation profiles |
| **Outputs** | Console log + basic HTML snippet | Standalone, semantic HTML5 report styled with dark mode and modern typography |
| **Warnings** | Hardcoded checks | Configurable warning heuristics (low density, missing ground data) |

#### 📚 Technical Learnings & Code Evolution

Developing the enhanced CLI tool involved deep diving into the ASPRS LAS specification details:

1. **Bitwise Unpacking:**
   LiDAR point formats compress spatial attributes using bitfields. The parser extracts packed properties using bitwise masks.
   * For **LAS 1.2** (Formats 0-5), the classification and flags share a single byte at offset 15:
     ```python
     flag_synthetic = bool(byte & 0x20)
     flag_keypoint = bool(byte & 0x40)
     flag_withheld = bool(byte & 0x80)
     classification = byte & 0x1F  # 5-bit code
     ```
   * For **LAS 1.4** (Formats 6-10), attributes are expanded. Return Number and Number of Returns are packed together in byte 14, and Classification Flags are separated from the Classification byte:
     ```python
     # Byte 14: Return number (bits 0-3) & Number of returns (bits 4-7)
     return_number = byte_14 & 0x0F
     number_of_returns = (byte_14 >> 4) & 0x0F
     ```

2. **Byte Alignment Debugging:**
   During development, we resolved a critical coordinate misalignment where LAS 1.4 files (PDRF 6-10) were returning false-positive classification warnings. By analyzing byte sizes, we corrected the offsets (e.g. referencing byte 16 for the expanded 8-bit Classification code, and reading the 8-byte GPS Time double at offset 22 instead of 20).

3. **Memory Optimization Thoughts:**
   While the current version loads the binary file into a byte buffer for speed, dealing with multi-gigabyte point clouds requires mapping the file via `mmap` or streaming point packets in chunks.

---

## ⚙️ Configuration & Parameters

To customize the CLI run behavior and thresholds, adjust the command-line options or modify the warning heuristics within `las_inspector.py`:

### CLI Flags & Arguments
* **`input` (positional):** Path to the raw `.las` or `.laz` file.
* **`--output`, `-o`:** Name of the output HTML report (defaults to `qc_report.html`).
* **`--density`:** Generates a dual-plot PNG showing the 2D Point Density heatmap and 3D Elevation map.
* **`--overlap-check`:** Runs a heuristic check for flight line overlap zones.
* **`--max-points`:** Limits the number of points processed (useful for rapid testing on massive files).

### Modifying Heuristic Thresholds
Open `las_inspector.py` and modify the following conditional checks in `compute_stats()` to fit your project standards:

1. **Minimum Ground Classification:**
   ```python
   if 2 not in classifications or classifications[2] < len(pts) * 0.01:
       # Warns if Ground points (Class 2) are less than 1% of total points
   ```
2. **Point Density Warning:**
   ```python
   if hdr.point_density < 10:
       # Triggers warning if density drops below 10 points per square unit
   ```
3. **Withheld Point Tolerance:**
   ```python
   withheld_pct = (withheld / len(pts)) * 100
   if withheld_pct > 5:
       # Triggers warning if more than 5% of points are flagged as withheld
   ```

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10 or newer
* [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### Install & Run
1. Clone this repository.
2. Place your LAS/LAZ file in the folder.
3. Install dependencies and run:

```bash
# Using uv:
uv sync
source .venv/bin/activate
python las_inspector.py las2018.laz --density --output qc_report.html

# Or using standard pip:
pip install -r requirements.txt
python las_inspector.py las2018.laz --density --output qc_report.html
```

### Output Files & Visualizations

The script generates three key deliverables:

1. **`qc_report.html`**: A standalone, semantic HTML5 report with responsive dark-mode styling, Google Fonts typography, interactive cards, and a detailed summary of point count, CRS, and returns. If `--density` is enabled, the visualization is embedded directly into this file.
2. **`las2018_qc_summary.txt`**: A clean, lightweight text file summary saved directly to the working directory for quick inspection or piping to other tools.
3. **`report_density.png`**: (Only generated with `--density`) A dual-panel figure mapping point density per bin next to a terrain-colored elevation scatter plot.

![LiDAR Density and Elevation Plots](report_density.png)
