use super::super::moving_window_contract::{
    moving_descriptor, moving_gpu_sample_points_extension, moving_integer_metadata,
};
use crate::*;
moving_descriptor!("movmin", "\"omitnan\"");
moving_gpu_sample_points_extension!(
    "movmin-gpu-sample-points",
    "movmin with SamplePoints on a GPU input is a RunMat extension",
    "RunMat:compatibility:MovminGpuSamplePointsExtension"
);
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_SAMPLE_POINTS_EXTENSION];
moving_integer_metadata!(
        "M = movmin(A, k, dim, nanflag, Name, Value)",
        ExactInteger,
        PreserveInput,
        GatherFallback,
        "Every window selects its minimum directly from exact same-class integer values and exact typed endpoint fills; integer overflow is inapplicable, while GPU SamplePoints is a mode-gated gather/reupload extension."
    );
