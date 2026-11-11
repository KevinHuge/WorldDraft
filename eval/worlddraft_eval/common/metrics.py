"""Current benchmark metrics selected for reporting."""

ENTITY_METRICS = ['DF_count_global_accuracy', 'DF_count_local_accuracy', 'bbox_iou']
MOTION_METRICS = ['track_matched_coverage', 'ISR', 'EventTimingCoverage']
CAMERA_METRICS = ['RotErr_deg_mean', 'TransErr_rel_mean', 'CamMC_rel_mean']
RF_METRICS = ['RF_total', *['RF_' + a for a in ('color', 'material', 'texture', 'environment', 'lighting')]]
AP_METRICS = ['AP_total', *['AP_' + a for a in ('color', 'material', 'texture', 'environment', 'lighting')]]
REPORT_METRICS = ENTITY_METRICS + MOTION_METRICS + CAMERA_METRICS + RF_METRICS + AP_METRICS + ['SL_color']
# Per-video QA columns and the camera metrics; RAW metrics are not rescaled to 0-100.
QA_KEYS = ['DF_count_global_accuracy', 'DF_count_local_accuracy', 'RF_total', 'AP_total', 'SL_color']
QA_KEYS += [f'{m}_{a}' for m in ('RF', 'AP') for a in ('color', 'material', 'texture', 'environment', 'lighting')]
CAMERA = CAMERA_METRICS
RAW = set(CAMERA + ['clip_text_visual_alignment'])
