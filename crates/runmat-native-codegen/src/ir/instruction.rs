use super::{
    NativeFrameState, NativeInstructionId, NativeMirSite, NativeOutput, NativeSafepointId,
    NativeSourceLocation, NativeValueId,
};
use runmat_types::{CapabilitySet, EffectSet};
use serde::{Deserialize, Serialize};

/// Semantic payload after MIR operands have been mapped to SSA inputs.
///
/// Retaining the canonical MIR operation metadata avoids a private inference or
/// duplicate semantic table. Generated code consumes `inputs`; payload operands
/// are identities used to validate/reconstruct exact runtime slow paths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "payload")]
pub enum NativeOperation {
    Rvalue {
        value: runmat_mir::MirRvalue,
        result: NativeRvalueResult,
    },
    Statement(runmat_mir::MirStmtKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "arity")]
pub enum NativeRvalueResult {
    Assignment,
    MultiAssignment(u32),
    Discard,
    Terminator,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInstruction {
    pub id: NativeInstructionId,
    pub site: NativeMirSite,
    pub source: NativeSourceLocation,
    pub class: runmat_mir::NativeLoweringClass,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub inputs: Vec<NativeValueId>,
    pub outputs: Vec<NativeOutput>,
    pub effect_epoch_output: Option<NativeValueId>,
    /// Canonical constructs conditionally embedded by the MIR operation (at
    /// present, short-circuit right-hand temporaries), in evaluation order.
    pub embedded_constructs: Vec<runmat_mir::MirConstructKind>,
    pub operation: NativeOperation,
    pub safepoint: Option<NativeSafepointId>,
    pub frame_state: Option<NativeFrameState>,
}
