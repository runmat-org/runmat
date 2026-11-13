use super::super::moving_window_contract::{
    moving_gpu_sample_points_extension, moving_stdvar_descriptor, moving_stdvar_integer_metadata,
};
use crate::*;
moving_stdvar_descriptor!("movvar");
moving_gpu_sample_points_extension!(
    "movvar-gpu-sample-points",
    "movvar with SamplePoints on a GPU input is a RunMat extension",
    "RunMat:compatibility:MovvarGpuSamplePointsExtension"
);
pub const TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "movvar-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "movvar with a typed-integer normalization or dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:MovvarTypedIntegerControlExtension"),
    };
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [GPU_SAMPLE_POINTS_EXTENSION, TYPED_INTEGER_CONTROL_EXTENSION];
moving_stdvar_integer_metadata!("movvar", "M = movvar(A, k, args...) with integer A");
