use super::{NativeBlock, NativeBlockId, NativeLocalId, NativeMirSite};
use runmat_types::{ProgramFunctionId, ProgramSourceId};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFunctionAbi {
    pub fixed_inputs: Vec<NativeLocalId>,
    pub varargin: Option<NativeLocalId>,
    pub fixed_outputs: Vec<NativeLocalId>,
    pub varargout: Option<NativeLocalId>,
    pub implicit_nargin: Option<NativeLocalId>,
    pub implicit_nargout: Option<NativeLocalId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFunction {
    pub id: ProgramFunctionId,
    pub source: ProgramSourceId,
    pub name: String,
    pub abi: NativeFunctionAbi,
    pub local_count: u32,
    pub entry: NativeBlockId,
    pub blocks: Vec<NativeBlock>,
    pub expected_sites: Vec<NativeMirSite>,
}
