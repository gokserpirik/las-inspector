import laspy
import numpy as np

las = laspy.read("las2018.laz")

header = las.header
print(header.point_count)

crs = las.header.parse_crs()
print(crs)

print(f"Min Bounds: X={header.x_min}, Y={header.y_min}, Z={header.z_min}")
print(f"Max Bounds: X={header.x_max}, Y={header.y_max}, Z={header.z_max}")

print(np.unique(las.classification))

classes, counts = np.unique(las.classification, return_counts=True)
print(dict(zip(classes, counts)))

returns, return_counts = np.unique(las.return_number, return_counts=True)
print(dict(zip(returns, return_counts)))

area = (header.x_max - header.x_min) * (header.y_max - header.y_min)
print(header.point_count / area, "points per square unit")

withheld_count = np.sum(las.withheld)
print(f"Withheld points: {withheld_count}")

intensities = las.intensity
print(f"Intensity range: {np.min(intensities)} to {np.max(intensities)}")

qc_passed = (header.point_count / area >= 2.0) and (withheld_count == 0)
print(f"QC Status: {'PASSED' if qc_passed else 'FAILED'}")

with open("qc_summary.txt", "w") as f:
    f.write(f"QC Verdict: {'PASSED' if qc_passed else 'FAILED'}\nTotal Points: {header.point_count}\n")

html_data = f"<html><body style='font-family:Arial;'><h1>QC Report</h1><p>Verdict: {'PASSED' if qc_passed else 'FAILED'}</p></body></html>"
with open("report.html", "w") as html_out:
    html_out.write(html_data)
print("HTML report generated successfully!")
