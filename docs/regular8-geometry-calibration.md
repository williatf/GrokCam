# Automatic Regular 8 capture geometry calibration

RAW Regular 8 capture automatically samples the existing preview and full
sprocket detections. It estimates the image-support geometry used by
postprocess P15 separately from the capture-ROI geometry used by P24. No
processing algorithm or transport-control constant is changed.

The calibrator requires at least 18 valid, non-stationary samples. It freezes
only when the most recent 12 samples satisfy the robust pitch, size, alignment,
and optical sanity limits. Frozen geometry statistics and their frame list are
calculated from those same 12 samples. The resulting object must also pass the
frozen-geometry sanity validator before it is written to the project's
`metadata.json` as `capture_geometry`. It uses schema version 1 and includes
`calibration_method: "regular8_capture_geometry_v1"` and `frozen: true`.

During capture the operator receives `regular8_geometry_status` websocket
events. A status of `calibrating` means the capture is proceeding while the
sample window is collected; `calibrated` means the geometry is frozen and will
be consumed automatically by postprocess. If the capture ends before a stable
window is available, capture remains valid and metadata records
`automatic_capture_calibration_failed`; postprocess then uses its normal
Regular 8 defaults unless a reviewed geometry override is supplied.

Frozen geometry is stored with the project and reused when that project is
selected for a later capture. An unfinished calibration is capture-local and
starts over after a process restart. Changes to optics or camera alignment do
not automatically invalidate frozen project geometry; select a new project
when those conditions change until an explicit recalibration workflow is added.

Super 8 does not instantiate this calibrator and its capture behavior is
unchanged. The calibrator is read-only with respect to frames and does not
modify raw DNGs, debug images, or processing outputs.
