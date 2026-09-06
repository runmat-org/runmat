use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AccelerationInferenceRule {
    Gather,
    GpuArray,
}
