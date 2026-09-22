# Automatic Regular 8 capture geometry calibration

RAW Regular 8 capture automatically samples the existing preview and full
sprocket detections. It estimates the image-support geometry used by
postprocess P15 separately from the capture-ROI geometry used by P24. No
processing algorithm or transport-control constant is changed.

The calibrator requires at least 18 valid, non-stationary samples. It freezes
only when the most recent 12 samples satisfy the robust pitch, size, alignment,
and optical sanity limits. Robust medians and MAD values are written to the
project's `metadata.json` as `capture_geometry`. The object uses schema version
1 and includes `calibration_method: "regular8_capture_geometry_v1"` and
`frozen: true`.

During capture the operator receives `regular8_geometry_status` websocket
events. A status of `calibrating` means the capture is proceeding while the
sample window is collected; `calibrated` means the geometry is frozen and will
be consumed automatically by postprocess. If the capture ends before a stable
window is available, capture remains valid and metadata records
`automatic_capture_calibration_failed`; postprocess then uses its normal
Regular 8 defaults unless a reviewed geometry override is supplied.

Super 8 does not instantiate this calibrator and its capture behavior is
unchanged. The calibrator is read-only with respect to frames and does not
modify raw DNGs, debug images, or processing outputs.
