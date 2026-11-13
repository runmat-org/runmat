use super::super::moving_window_contract::{
    moving_descriptor, moving_gpu_sample_points_extension, moving_integer_metadata,
};
use crate::*;
moving_descriptor!("movmedian", "\"includenan\"");
moving_gpu_sample_points_extension!(
    "movmedian-gpu-sample-points",
    "movmedian with SamplePoints on a GPU input is a RunMat extension",
    "RunMat:compatibility:MovmedianGpuSamplePointsExtension"
);
pub const GPU_LARGE_WINDOW_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "movmedian-gpu-large-window",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "movmedian with a window longer than 31 on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MovmedianGpuLargeWindowExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [GPU_SAMPLE_POINTS_EXTENSION, GPU_LARGE_WINDOW_EXTENSION];
moving_integer_metadata!(
        "M = movmedian(A, k, dim, nanflag, Name, Value)",
        ExactInteger,
        PreserveInput,
        GatherFallback,
        "Every window orders exact same-class integer values and uses the native median midpoint rule; exact typed endpoint fills retain class, while GPU SamplePoints and windows longer than 31 are separately mode-gated gather/reupload extensions."
    );
