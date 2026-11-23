# Parcel segmentation lab

Test whether **separated boxes and sealed clothing courier bags** can be segmented
into individual masks using your Gemini 2 or saved images. Start with parcels that
have visible gaps between them on an otherwise empty bin floor or table.

This first experiment uses an **empty reference scene**, not a trained parcel
recognition model. Changed foreground components are treated as candidate parcels.
It needs no GPU, model weights, ROS installation, TCP calibration, or robot connection.

## MacBook camera with a highlighted pickup preview

The live command now defaults to the MacBook/webcam source. Run:

```bash
cd /Users/shashankshende/Projects/parcel_segmentation_lab
source .venv/bin/activate
python run.py live --source webcam --mode rgb --grasp-preview
```

Point the camera at a plain work area and keep the laptop fixed. Choose an area
where the entire parcel is visible; a downward view is preferable. Select the
work area with **R**, capture it empty with **B**, then place separated parcels
and remove your hands. Press **F** to freeze/unfreeze the view, **S** to save the
current result, and **Q** to quit. Allow camera access if macOS asks. If index 0
is another connected camera, choose the built-in camera using `--camera-index`.

Run this command **in the IDE's integrated terminal or macOS Terminal**. A
background agent process can be denied camera access even when the IDE has it.
If access fails, check **System Settings → Privacy & Security → Camera** for
the application you launched Python from, then restart the command. The local
background launch was denied by macOS; live frames have not been verified.

The preview shows:

- A translucent colored mask for each foreground candidate.
- A **bright green, axis-aligned box around the complete parcel**, with a filled
  label such as `parcel 2`, matching the requested top-view sorting display.
- A **purple full-parcel box** labeled `target N` for the selected 2D candidate
  when grasp preview is enabled.
- A smaller **white rotated cup layout** only on that target: two circles for
  the contact footprints, an arrow for the cup-pair axis, and a white midpoint cross.
- Midpoint **u, v in original image pixels**, plus an angle in degrees measured
  **clockwise from image-right**. 0 degrees is horizontal and 90 is vertical.
- A message when no complete two-cup layout fits. Clipped masks are rejected.

The large parcel box and small cup-layout rectangle describe different things:
the former encloses the entire segmented object, while the latter shows where
the two cups would fit in image space. `parcel N` is a per-frame ID, not a proven
pick sequence. The crowded reference scene requires individual masks even where
parcels touch or overlap; the existing empty-reference/connected-components
baseline still requires separated parcels. Updating the display does not add
segmentation of touching or hidden parcels.

The angle describes the proposed **cup-pair direction**, not an independently
estimated rigid-object pose. For symmetric cups, angles 180 degrees apart are
equivalent. Frames are not mirrored by the application. Image perspective means
this angle must not be sent directly to the robot as base-frame yaw.

The reported hardware has 70 mm centre spacing and 20–30 mm cup diameters.
Default preview spacing/radius are 84/18 **pixels**, preserving the proportions
of a 70 mm spacing and a conservative 30 mm diameter at an **arbitrary illustrative
scale**. They are not a measured conversion from millimetres. Adjust the pixel
spacing and radius with trackbars or `--cup-spacing-px` / `--cup-radius-px`.
An RGB camera alone does not determine this image scale for unknown parcels.

The planner samples positions and rotations within each mask, checks both cup
circles plus an edge margin, and ranks them using the weaker contact's boundary
clearance with a small centering preference. This is **2D silhouette geometry**:
it cannot verify folds, vacuum sealing, surface tilt, mass distribution, or 3D
tool clearance. The rectangular outline visualizes the layout's extent; only
the two circular footprints plus margin are checked against the parcel mask.
Search is sampled, so an unsampled feasible pose can be missed.

This is a lightweight pickup visualization, not the NVIDIA
[FoundationPose](https://github.com/NVlabs/FoundationPose) model. That system
estimates/tracks 6D object pose; choosing a suction contact is a further task.
This prototype reports no depth, surface normal, robot XYZ, roll, or pitch.
The existing calibrated TCP must also be accounted for before future robot use.

Saved runs include **`grasp_overlay.png`** and **`grasp_preview.json`**, containing
each candidate's midpoint, both contact centres, image angle, layout corners,
rejection reason, and ranking measure. The JSON explicitly marks the scale as
uncalibrated and `robot_pose` as null. `overlay.png` remains the plain segmentation.

To see the same display on artificial parcels without opening a camera:

```bash
python run.py demo --mode rgb --grasp-preview --show
```

## Quick start: test without a camera

Run in a terminal:

```bash
cd /Users/shashankshende/Projects/parcel_segmentation_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py demo
```

If the environment is already installed, just activate it and run the demo. The
dependencies are pinned for Python 3.9–3.12; the local checks used Python 3.9.
The NumPy version varies by Python version to match the Orbbec SDK's requirements.

The demo generates three artificial, separated parcels and ground-truth masks.
It checks RGB, depth, and hybrid modes independently. Each should report `PASS`,
three candidates, and foreground intersection-over-union (IoU) of at least 0.95.
This verifies the pipeline on artificial data; it does not measure real parcel accuracy.

Inputs go in `data/demo/`. Each result gets a unique folder under `outputs/`.
Open its `overlay.png` to see each candidate in a different color. Optionally:

```bash
python run.py demo --show
```

## Test your own saved images

1. Hold the camera at one fixed observation pose; for a wrist-mounted camera,
   return to exactly that robot pose and keep it still during capture.
2. Save a color image of the **empty** bin/table as `data/background.png`.
3. Place 2–5 parcels with clear gaps. Remove hands and other objects.
4. Save `data/parcels.png` at the same camera pose, resolution, and lighting.
5. Run:

```bash
python run.py image \
  --background data/background.png \
  --image data/parcels.png \
  --expected-count 3 \
  --select-roi \
  --show
```

Change `3` to the number of parcels you actually placed. In the ROI window, drag
a rectangle inside the bin walls, then press Enter/Space. Escape cancels. An
incorrect count returns exit code 1 and still saves all results for inspection.
Exit code 2 means an input/configuration error.

For a terminal without a desktop, omit `--show`/`--select-roi`, and optionally pass
`--roi X Y WIDTH HEIGHT` in the **original image's pixel coordinates**. No images
are silently resized to make input dimensions match.

## Live Gemini 2 test

Install the optional official Orbbec Python binding in the same environment:

```bash
python -m pip install -r requirements-camera.txt
python run.py live --source orbbec --mode depth
```

The package is named `pyorbbecsdk2`; the imported Python module is `pyorbbecsdk`.
Use a wheel compatible with your operating system, architecture, Python, and
camera firmware. If pip cannot find a matching wheel, follow the official
[Orbbec installation documentation](https://orbbec.github.io/pyorbbecsdk/).
The camera requirements also pin compatible PyObjC dependencies on macOS with
Python 3.9, avoiding an unsupported source build pulled by the SDK's examples.
On Linux, follow its USB/udev setup. Close Orbbec Viewer and other camera users
before starting. The script does not install drivers or update firmware.

The adapter starts RGB and depth, requests frame synchronization, uses the SDK's
**depth-to-color AlignFilter**, and converts raw depth with `get_depth_scale()`
to millimetres. It never substitutes a resize for geometric alignment.

With the robot/camera at a stationary observation pose:

| Key | Action |
| --- | --- |
| **R** | Select a rectangular ROI inside the bin walls; press Enter to accept. |
| **B** | Capture the empty reference from the last 9 frames. Empty the bin and hold it still with hands clear before pressing. |
| **S** | Save the displayed frame, its reference, depth arrays, masks, crops, settings, and report. |
| **Q / Esc** | Quit and release the camera. |

Capture the empty reference first, then add separated parcels. Do not press B
again while parcels are present: that would make those parcels part of the background.
Recapture the empty reference after any camera movement or significant lighting change.
Saved reference folders also let you keep the empty capture independently.

The left preview shows candidate masks and IDs. The right preview shows the
binary foreground mask: white is foreground; black is background. Trackbars tune
the RGB difference threshold, depth difference threshold, and minimum component area.

For a different connected camera index:

```bash
python run.py live --source orbbec --camera-index 1 --mode hybrid
```

For an ordinary RGB webcam, without the Orbbec SDK:

```bash
python run.py live --source webcam --camera-index 0 --mode rgb
```

Desktop previews require a graphical session. On macOS, camera access may require
permission for the terminal or editor running Python.

## Reprocess a saved RGB-D capture

Every saved live result includes its own matching background and aligned depth.
Replace `outputs/run_NAME` below with your actual saved directory:

```bash
python run.py image \
  --background outputs/run_NAME/background.png \
  --image outputs/run_NAME/scene.png \
  --background-depth outputs/run_NAME/background_depth_mm.npy \
  --depth outputs/run_NAME/depth_mm.npy \
  --mode depth \
  --depth-change-mm 15 \
  --expected-count 3
```

Read the recorded ROI in `instances.json` and pass it with `--roi` when replaying
a cropped experiment. Inputs can also be `.npy` depth arrays or single-channel
uint16 PNG depth images **already expressed in mm**. Zero/NaN means missing depth.
An Orbbec raw 16-bit PNG may need its SDK depth scale applied first; a colored
depth screenshot is not a distance image. Matching array sizes alone do not prove alignment.

## Choose and tune a mode

| Mode | Foreground rule | Useful when |
| --- | --- | --- |
| `rgb` | Color differs from the empty reference in OpenCV's 8-bit Lab space. | Saved color images or webcam; parcels contrast with the bin. |
| `depth` | Valid distance is at least the threshold closer than the empty reference. | Gemini 2; parcels resemble the bin color or cast shadows. |
| `hybrid` | Depth where both readings are valid; RGB fallback only where depth is missing. | Depth has holes but color contrast is useful. |

Defaults are starting values, not calibrated sensor tolerances. `--depth-change-mm 15`
means a **15 mm change along the camera depth axis**, not true parcel height above
a tilted bin floor. Thin bags may need a lower threshold, provided measured depth
noise allows it. RGB mode can detect thin bags when depth cannot resolve them.

If parcels are missed, lower the relevant threshold or minimum area. If noise or
shadows become candidates, raise the threshold, improve lighting, use depth, or
restrict the ROI. If bags resemble the floor, try depth. If one parcel fragments
into several masks, inspect contrast/depth holes and adjust acquisition or thresholds.

The default 3-pixel morphological opening removes small specks. Use `--open-kernel 0`
for thin details. There is no dilation, closing, convex hull, or automatic filling
that could silently connect neighboring objects. Gaps must still be large enough
to survive image resolution, blur, and noise. Touching/overlapping/stacked parcels
can merge into one component; this baseline does not separate hidden instances.

## Inspect the result

| File | Contents |
| --- | --- |
| `overlay.png` | Colored masks, candidate IDs, boundaries, and bounding boxes. |
| `foreground.png` | Binary union of retained masks. |
| `labels.png` | uint16 instance IDs; 0 is background, 1..N are candidates. |
| `masks/parcel_001.png` | One full-resolution binary mask per candidate. |
| `crops/parcel_001.png` | Cropped color image with mask as PNG transparency. |
| `instances.json` | Count, pixel boxes/areas/centroids, optional depth statistics, settings, input metadata, and warnings. |
| `scene.png`, `background.png` | Exact color inputs for replay. |
| `depth_mm.npy`, `background_depth_mm.npy` | Matching aligned distance arrays when depth was supplied. |

IDs are assigned per frame; they are not persistent tracking IDs. Centroids and
boxes are image measurements, **not suction targets or robot coordinates**.
Objects touching the ROI boundary are retained and flagged for inspection.
Background subtraction also detects hands, tools, and other newly added objects;
it does not identify their semantic class.

For the first real test, capture 10–20 scenes with known counts and inspect every
mask, not just the total count. Check that each parcel has one complete mask,
gaps remain black, and no floor/wall/shadow is included. Real annotated masks are
needed to measure boundary accuracy. Record boxes, white/grey/black bags, shiny
film, thin bags, different orientations, and the smallest expected gap.

After this separated-parcel baseline works on real images, the next development
step is segmentation of touching/stacked parcels. Suction contact planning uses
your 70 mm cup spacing plus the still-needed cup diameters and tool geometry;
it is a separate stage.

## Automated checks

```bash
python -m unittest discover -s tests -v
```

These check individual masks against known shapes, empty scenes, same-color
objects in depth, shadow rejection, invalid depth, ROI clipping, small gaps,
touching-object limitations, saved-file consistency, and command exit codes.
Camera acquisition needs verification on your connected Gemini 2.

Local verification: all 21 automated checks passed with Python 3.9.6, NumPy 1.26.4,
and OpenCV 4.11.0. The artificial three-parcel demo passed in all three modes
(RGB foreground IoU 0.9814; depth/hybrid 0.9999). Orbbec SDK 2.1.3 installed and
its required adapter APIs imported successfully. The two-cup preview also passed
checks for rotated and concave masks, per-cup footprint containment, clipped and
too-narrow masks, translation consistency, and explicitly absent robot poses.
The generated overlay was visually inspected. A physical camera stream and the
desktop camera preview still require a camera-authorized application context.

Implementation references:
[OpenCV connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html),
[Orbbec Python SDK](https://github.com/orbbec/pyorbbecsdk), and
[Orbbec alignment example](https://github.com/orbbec/pyorbbecsdk/blob/v2-main/examples/beginner/03_color_and_depth_aligned.py).
