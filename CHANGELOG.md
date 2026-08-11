# Changelog

All notable changes to FIDMAA GUI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Restored absolute `Highest` and `Lowest` region selectors while retaining
  pose-resistant `Local peak` and `Local valley` selectors.
- Added a visible skin-matte-to-stable-depth neck overlay on the main image and
  zoom views, including raw silhouette edges, corrected depth points, and the
  measured neck arc.
- Added a configurable contour step (1–8 raw depth levels), defaulting to every
  third level, to reduce visual density without changing measurement data.
- Added the `surface_vector_filtered` measurement alongside the existing direct
  and surface vector measurements.
- Added surface length results for sampling intervals `N=2` through `N=7` to
  the measurement panel.
- Added a same-size 3x3 median filter for raw disparity data, bilinear sampling
  at fractional coordinates, and invalid zero-depth detection.
- Added evenly distributed profile samples that retain both measurement
  endpoints and produce the same result regardless of point order.

### Fixed

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
