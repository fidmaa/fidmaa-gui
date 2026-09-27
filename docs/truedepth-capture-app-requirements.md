# TrueDepth capture app — requirements for FIDMAA analysis

Audience: the developer/agent building the iOS capture app. The photos it
produces are analysed by `portrait-analyser` and FIDMAA GUI.

## Why

On iPhone 17 Pro the built-in Camera app stores portrait depth with
`AVDepthData.depthDataAccuracy == .relative` (checked in both Portrait and
Photo mode). Validated against a ChArUco board (2026-09-27, IMG_2336–2343),
that depth underestimates distance by ~7 % at 28–37 cm and 17–28 % at
50–75 cm, varying from photo to photo; every millimetre measurement scales with
it. iPhone 14 Camera photos were `.absolute`. With the correct per-photo
distance, mouth-opening measurements agreed within ~2 mm across 40–84 cm.

The app must deliver **absolute** metric depth from the TrueDepth (Face ID)
sensor, independent of the Camera app.

## Capture

1. Device: `AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front)`.
2. `AVCapturePhotoOutput` with:
   - `isDepthDataDeliveryEnabled = true` (output and per-photo settings);
   - `isPortraitEffectsMatteDeliveryEnabled = true`;
   - `enabledSemanticSegmentationMatteTypes = [.teeth, .skin, .hair]` —
     **required**: teeth detection and neck analysis depend on these mattes;
   - `isCameraCalibrationDataDeliveryEnabled = true` where supported.
3. After capture, read `photo.depthData`:
   - **Record `depthDataAccuracy`, `depthDataQuality`, `isDepthDataFiltered`.**
   - If accuracy is not `.absolute`, show a clear warning in the UI and store
     the flag — never silently accept relative depth.
4. Prefer delivering **unfiltered** depth (`isDepthDataFiltered = false` in the
   photo settings) as well as, or instead of, the filtered map: filtering
   smooths edges, which corrupts depth at the incisal edges next to the mouth
   cavity.
5. Working distance ~20–50 cm (TrueDepth's reliable range). Show the live
   distance to the face if possible.
6. No GPS/location metadata (patient photos). Filename should carry a study
   ID supplied by the operator.

## Output format

Two options — agree on one before implementation:

- **A (preferred): lossless float depth.** Keep the HEIC from
  `photo.fileDataRepresentation()` for the colour image and mattes, and write
  the depth map losslessly next to it (e.g. 32-bit float TIFF or `.npy`, in
  metres, `kCVPixelFormatType_DepthFloat32`), plus a JSON sidecar with
  accuracy, quality, filtered flag, depth map size, and the calibration below.
  `portrait-analyser` will get a loader for this format.
- **B: Camera-app-compatible HEIC.** Encode depth as 8-bit disparity with
  `FloatMinValue`/`FloatMaxValue` XMP exactly like the Camera app, so the
  existing `load_image()` works unchanged. Costs precision (256 levels).

Either way, store from `depthData.cameraCalibrationData`:
`intrinsicMatrix`, `intrinsicMatrixReferenceDimensions`, `pixelSize`,
`lensDistortionCenter`, `lensDistortionLookupTable`,
`inverseLensDistortionLookupTable`, `extrinsicMatrix`. The analysis will use
the stored focal length instead of the iPhone 14 calibration polynomial.

## Acceptance test

Photograph the ChArUco board (5×7, 25 mm squares, 18 mm markers,
DICT_6X6_250) held next to the face at ~20, 30, 40, 50 cm. The board pose
gives the true distance independently of depth; the app's depth at the board
must agree within ~2 % (Camera app today: 7–28 % short).
