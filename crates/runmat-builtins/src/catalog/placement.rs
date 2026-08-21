use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPortability {
    NativeAndWasm,
    NativeOnly,
    WasmHostBridge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinAcceleratorPolicy {
    Forbidden,
    Optional,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinResidencyPolicy {
    Host,
    PreserveInputs,
    ProduceResident,
    GatherToHost,
    Dynamic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinFusionPolicy {
    Never,
    Candidate,
    Boundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPlacementContract {
    pub portability: BuiltinPortability,
    pub accelerator: BuiltinAcceleratorPolicy,
    pub residency: BuiltinResidencyPolicy,
    pub fusion: BuiltinFusionPolicy,
}
