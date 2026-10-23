use super::super::moving_window_contract::{
    moving_descriptor, moving_gpu_sample_points_extension, moving_integer_metadata,
};
use crate::*;
moving_descriptor!("movmax", "\"omitnan\"");
moving_gpu_sample_points_extension!(
    "movmax-gpu-sample-points",
    "movmax with SamplePoints on a GPU input is a RunMat extension",
    "RunMat:compatibility:MovmaxGpuSamplePointsExtension"
);
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_SAMPLE_POINTS_EXTENSION];
moving_integer_metadata!(
        "M = movmax(A, k, dim, nanflag, Name, Value)",
        ExactInteger,
        PreserveInput,
        GatherFallback,
        "Every window selects its maximum directly from exact same-class integer values and exact typed endpoint fills; integer overflow is inapplicable, while GPU SamplePoints is a mode-gated gather/reupload extension."
    );
