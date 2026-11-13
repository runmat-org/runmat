use super::super::moving_window_contract::{
    moving_descriptor, moving_gpu_sample_points_extension, moving_integer_metadata,
};
use crate::*;
moving_descriptor!("movprod", "\"includenan\"");
moving_gpu_sample_points_extension!(
    "movprod-gpu-sample-points",
    "movprod with SamplePoints on a GPU input is a RunMat extension",
    "RunMat:compatibility:MovprodGpuSamplePointsExtension"
);
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_SAMPLE_POINTS_EXTENSION];
moving_integer_metadata!(
        "M = movprod(A, k, dim, nanflag, Name, Value)",
        FloatingPoint,
        Double,
        GatherFallback,
        "Integer data is materialized once into the documented double computation and output domain; endpoint shape rules and missing-value spellings share that rule, and GPU SamplePoints is a mode-gated gather/reupload extension."
    );
