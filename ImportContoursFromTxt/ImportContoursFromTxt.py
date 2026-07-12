"""This file acts as the main module for this script.
It is designed to be run from within Fusion 360's scripting environment.
To execute it:
    1. Open Fusion 360.
    2. Create a new Project (Button on the left)
    3. Choose a new Part Design (Top left) and create it
    4. Go to the 'Scripts and Add-Ins' menu (under the 'Tools' tab) (or Zusatzmodule -> Skripte und Zusatzmodule im Reiter Dienstprogramme)
    5. Click on Scripts
    6. Click on the "+" button at the top and add a script or Add-In from a folder
    7. Select "ImportContoursFromTxt" folder
    8. Click on the script and click "Run"
"""

import os
import re
import traceback
from collections import defaultdict

import adsk.core
# import adsk.fusion

# Initialize the global variables for the Application and UserInterface objects.
app = adsk.core.Application.get()
ui  = app.userInterface

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CONTOUR_FOLDER = r"D:\Studium\Physik\Bachelorarbeit\MSR-field_cancelation\Contours"

# Mapping von planeKey zu Fusion-BaseFace Namen
# Front = XY, Top = XZ, Right = YZ
PLANE_MAP = {
    "xy": "Front",   # XY-Plane
    "xz": "Top",     # XZ-Plane
    "yz": "Right",   # YZ-Plane
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def find_face_groups(folder):
    """
    Group .txt files by:
      Face1_Top_xy_contour1.txt
      -> prefix = "Face1_Top_xy", planeKey = "xy"
    Returns:
      { prefix: (planeKey, [list of full file paths]) }
    """
    groups = defaultdict(list)
    plane_keys = {}

    # Pattern: Face<number>_<name>_<xyz>_contour<number>.txt
    pattern = re.compile(r"^(Face\d+_.+?_([xyz]{2}))_contour\d+\.txt$", re.IGNORECASE)

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".txt"):
            continue
        m = pattern.match(fname)
        if not m:
            ui.messageBox(f"[skip] '{fname}' does not match the expected naming pattern")
            continue
        prefix, plane_key = m.group(1), m.group(2).lower()
        groups[prefix].append(os.path.join(folder, fname))
        plane_keys[prefix] = plane_key

    # Sort files for deterministic order:
    for prefix in groups:
        groups[prefix].sort()

    return {prefix: (plane_keys[prefix], files) for prefix, files in groups.items()}


def get_base_face(design, plane_name):
    """
    Gehört die BaseFace mit dem Namen 'plane_name' aus den Standard-Planes.
    plane_name: "Front", "Top", "Right"
    We use construction planes instead of baseFaces.
    """
    root_comp = design.rootComponent

    if plane_name == "Front":
        return root_comp.xYConstructionPlane
    elif plane_name == "Top":
        return root_comp.xZConstructionPlane
    elif plane_name == "Right":
        return root_comp.yZConstructionPlane
    else:
        return None


def sketch_spline_from_points(sketch, points):
    """
    points: List of adsk.core.Point3D (in 3D)
    Creates a sketch curve (fitted spline) from these points.
    """
    try:
        # Wrap Point3D objects into an ObjectCollection
        point_collection = adsk.core.ObjectCollection.create()
        for p in points:
            point_collection.add(p)

        spline = sketch.sketchCurves.sketchFittedSplines.add(point_collection)
        return spline
    except Exception as e:
        ui.messageBox(f"[warn] Could not create fitted spline from points: {e}")
        return None

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if app is None:
        ui.messageBox("Could not get Fusion application.")
        return

    # Use the currently active document instead of creating a new one
    doc = app.activeDocument
    if doc is None:
        ui.messageBox("No active document. Please open/Create a document first.")
        return

    # Use design instead of part (for this Fusion version)
    try:
        design = doc.design
    except Exception:
        ui.messageBox("Active document does not seem to be a Design document.")
        return

    component = design.rootComponent

    # Scan folder
    ui.messageBox(f"Scanning: {CONTOUR_FOLDER}")
    try:
        face_groups = find_face_groups(CONTOUR_FOLDER)
    except Exception as e:
        ui.messageBox(f"Failed to scan folder: {e}")
        app.log(f'Failed:\n{traceback.format_exc()}')
        return

    if not face_groups:
        ui.messageBox("No matching .txt files found. Check CONTOUR_FOLDER / naming.")
        return

    # Für jede Face-Gruppe:
    for prefix, (plane_key, files) in face_groups.items():
        plane_name = PLANE_MAP.get(plane_key)
        if plane_name is None:
            ui.messageBox(f"[skip] Unknown plane suffix '{plane_key}' for group '{prefix}'")
            continue

        # ui.messageBox(f"\n=== Face group: {prefix}  (plane: {plane_name}) ===")

        # 2. BaseFace für die Ebene holen
        base_face = get_base_face(design, plane_name)
        if base_face is None:
            ui.messageBox(f"  [fail] BaseFace '{plane_name}' not found. Check your Fusion setup.")
            continue

        # 3. Skizze auf der Ebene erstellen
        sketch = component.sketches.add(base_face)
        sketch.name = prefix

        # 4. Für jede Datei: Punkte lesen und Spline erstellen
        for file_path in files:

            points_3d = []
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts_line = line.split()
                        if len(parts_line) < 3:
                            continue
                        x, y, z = map(float, parts_line[:3])

                        # Wahl: planar auf die Ebene setzen
                        if plane_key == "xy":
                            z = 0.0
                        elif plane_key == "xz":
                            y = 0.0
                        elif plane_key == "yz":
                            x = 0.0

                        pts = adsk.core.Point3D.create(x, y, z)
                        points_3d.append(pts)
            except Exception as e:
                ui.messageBox(f"  [warn] Could not read file {file_path}: {e}")
                continue

            if len(points_3d) < 2:
                ui.messageBox(f"  [warn] Not enough points in {file_path}")
                continue

            # Spline aus Punkten in der Skizze erstellen
            sketch_spline_from_points(sketch, points_3d)

        ui.messageBox(f"  Sketch created for '{prefix}'.")

    ui.messageBox("\nDone.")


# If you want to run this script from Fusion:
def run(_context: str):
    """This function is called by Fusion when the script is run."""
    try:
        main()
    except:  # pylint: disable=bare-except
        app.log(f'Failed:\n{traceback.format_exc()}')
        ui.messageBox("Script failed. Check the Text Commands window for details.")
