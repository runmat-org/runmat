use super::super::moving_window_contract::{
    moving_descriptor, moving_gpu_sample_points_extension, moving_integer_metadata,
};
use crate::*;
moving_descriptor!("movmean", "\"includenan\"");
moving_gpu_sample_points_extension!(
    "movmean-gpu-sample-points",
    "movmean with SamplePoints on a GPU input is a RunMat extension",
    "RunMat:compatibility:MovmeanGpuSamplePointsExtension"
);
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_SAMPLE_POINTS_EXTENSION];
moving_integer_metadata!(
        "M = movmean(A, k, dim, nanflag, Name, Value)",
        FloatingPoint,
        Double,
        GatherFallback,
        "Integer data is materialized once into the documented double mean and output domain; endpoint shape rules and missing-value spellings share that rule, and GPU SamplePoints is a mode-gated gather/reupload extension."
    );
