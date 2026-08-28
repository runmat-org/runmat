use super::{
    NativeBlockId, NativeBlockParameter, NativeFrameState, NativeInstruction, NativeLocalId,
    NativeMirSite, NativeSourceLocation, NativeValueId,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum NativeEdgeArgument {
    Value(NativeValueId),
    LoopIteration { local: NativeLocalId },
    CaughtException { local: NativeLocalId },
    AwaitResult { local: NativeLocalId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEdge {
    pub target: NativeBlockId,
    pub arguments: Vec<NativeEdgeArgument>,
    pub side_effect_epoch: NativeValueId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NativeTerminatorKind {
    Goto {
        edge: NativeEdge,
    },
    Branch {
        condition: NativeValueId,
        then_edge: NativeEdge,
        else_edge: NativeEdge,
    },
    Switch {
        discriminant: NativeValueId,
        cases: Vec<(NativeValueId, NativeEdge)>,
        otherwise: NativeEdge,
    },
    For {
        iterable: NativeValueId,
        binding: NativeLocalId,
        body: NativeEdge,
        exit: NativeEdge,
    },
    ParFor {
        region: runmat_types::ParallelRegionId,
        iterable: NativeValueId,
        maximum_workers: Option<NativeValueId>,
        binding: NativeLocalId,
        body: NativeEdge,
        exit: NativeEdge,
    },
    Spmd {
        region: runmat_types::ParallelRegionId,
        header: Vec<NativeValueId>,
        body: NativeEdge,
        exit: NativeEdge,
    },
    TryCatch {
        try_edge: NativeEdge,
        catch_edge: NativeEdge,
        catch_binding: Option<NativeLocalId>,
    },
    Return {
        values: Vec<NativeValueId>,
    },
    Await {
        future: NativeValueId,
        result: Option<runmat_mir::MirPlace>,
        resume: NativeEdge,
    },
    Unreachable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTerminator {
    pub site: NativeMirSite,
    pub source: NativeSourceLocation,
    pub class: runmat_mir::NativeLoweringClass,
    pub safepoint: Option<super::NativeSafepointId>,
    pub kind: NativeTerminatorKind,
    pub frame_state: NativeFrameState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRegionBoundaryKind {
    Entry,
    Exit,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRegionValueBinding {
    pub value: runmat_types::RegionValueId,
    pub ssa: NativeValueId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRegionGuard {
    pub contract: runmat_types::RegionGuardContract,
    pub value: Option<NativeValueId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRegionBoundary {
    pub region: runmat_types::RegionId,
    pub point: runmat_types::ProgramPointId,
    pub kind: NativeRegionBoundaryKind,
    pub live_values: Vec<NativeRegionValueBinding>,
    pub guards: Vec<NativeRegionGuard>,
    pub frame_state: NativeFrameState,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBlock {
    pub id: NativeBlockId,
    pub parameters: Vec<NativeBlockParameter>,
    pub side_effect_epoch: NativeValueId,
    pub region_boundaries: Vec<NativeRegionBoundary>,
    pub instructions: Vec<NativeInstruction>,
    pub terminator: NativeTerminator,
}
