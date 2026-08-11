# Changelog

All notable changes to FIDMAA GUI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Added an `Area (uniform)` region selector for surfaces without a single
  meaningful extremum. It samples deterministic corresponding positions over
  both disks and bypasses candidate-depth ranking.
- Restored absolute `Highest` and `Lowest` region selectors while retaining
  pose-resistant `Local peak` and `Local valley` selectors.
- Added a visible skin-matte-to-stable-depth neck overlay on the main image and
  zoom views, including raw silhouette edges, corrected depth points, and the
  measured neck arc.
- Added a configurable contour step (1–8 raw depth levels), defaulting to every
  fourth level, to reduce visual density without changing measurement data.
- Neck output now distinguishes the curved 3D surface polyline from its direct
  Euclidean chord and reports median-filtered straight-row surface vectors.
- Added the `surface_vector_filtered` measurement alongside the existing direct
  and surface vector measurements.
- Added surface length results for sampling intervals `N=2` through `N=7` to
  the measurement panel.
- Added a same-size 3x3 median filter for raw disparity data, bilinear sampling
  at fractional coordinates, and invalid zero-depth detection.
- Added evenly distributed profile samples that retain both measurement
  endpoints and produce the same result regardless of point order.

### Changed

- Changed measurement defaults to a 20 px patch radius and 5% candidate pool,
  and increased the supported profile count from 10 to 30 (default remains
  10).

### Fixed

- Remove the experimental cursor-depth-band display and explicitly calculate
  contour boundaries from a 3×3 median map to suppress isolated depth noise.
- Pass the semantic hair matte into neck-edge selection and increase automatic
  front-arc sampling from 10 to 25 points.
- Bound automatic neck-row selection below the lowest FaceMesh landmark using
  face scale, and expose the selected anatomical Y search band in the output.
- Position teeth-centroid markers inside the aspect-fitted teeth-map rectangle,
  including its letterbox offsets, instead of treating the entire widget as
  image content.
- Changed pixel-to-millimetre conversion to measure X/Y coordinates from the
  image centre (the principal-point approximation) instead of the top-left
  corner.
- Removed the depth-dependent lateral displacement that distorted distances
  between points at different camera depths while preserving the calibrated
  pixel scale for points at the same depth.

### Tests

- Added regression tests for median filtering, bilinear depth sampling,
  sampling intervals, flat-surface stability, point-order independence, and
  invalid depth handling.
- Added regression tests confirming that the image centre maps to X=Y=0 at
  every depth, equal-depth measurements retain their calibrated scale, and
  motion along the optical axis does not acquire a phantom X/Y component.
