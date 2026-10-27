use super::{NativeLocalId, NativeValueId};
use runmat_types::{ProgramPointId, ProgramSourceId, ProgramSpan};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSourceLocation {
    pub source: ProgramSourceId,
    pub span: ProgramSpan,
}

pub type NativeSitePhase = runmat_mir::MirOperationPhase;
pub type NativeMirSite = runmat_mir::MirOperationSite;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFrameLocal {
    pub local: NativeLocalId,
    pub value: NativeValueId,
}

/// Exact state materializable at a native safepoint or control transfer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFrameState {
    pub point: ProgramPointId,
    pub source: NativeSourceLocation,
    pub locals: Vec<NativeFrameLocal>,
    pub operands: Vec<NativeValueId>,
    pub side_effect_epoch: NativeValueId,
}
