use super::{NativeBlock, NativeBlockId, NativeIndexExpression, NativeLocalId, NativeMirSite};
use runmat_types::{
    BindingId, FunctionArgDefaultValue, FunctionArgSizeSpec, FunctionArgValidator,
    ProgramFunctionId, ProgramSourceId,
};
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

/// Executor-neutral validation and default metadata for one fixed input.
///
/// The semantic constraint vocabulary is owned by `runmat-types`; Native IR
/// adds only the exact local identity needed by a native execution host.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFunctionArgumentValidation {
    pub input: NativeLocalId,
    pub size: Option<FunctionArgSizeSpec>,
    pub class_name: Option<String>,
    pub validators: Vec<FunctionArgValidator>,
    pub default_value: Option<FunctionArgDefaultValue>,
}

/// Executor-neutral metadata for one canonical MIR local.
///
/// Native artifacts retain semantic binding identity and names instead of VM
/// slots. This lets every host, including the browser/WASM host, implement
/// workspace, global, and persistent behavior without depending on bytecode
/// frame layout.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLocalMetadata {
    pub id: NativeLocalId,
    pub binding: Option<BindingId>,
    pub name: Option<String>,
    pub kind: NativeLocalKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeLocalKind {
    Parameter,
    Output,
    Binding,
    Temporary,
    Capture,
}

impl From<&runmat_mir::MirLocalKind> for NativeLocalKind {
    fn from(kind: &runmat_mir::MirLocalKind) -> Self {
        match kind {
            runmat_mir::MirLocalKind::Parameter => Self::Parameter,
            runmat_mir::MirLocalKind::Output => Self::Output,
            runmat_mir::MirLocalKind::Binding => Self::Binding,
            runmat_mir::MirLocalKind::Temporary => Self::Temporary,
            runmat_mir::MirLocalKind::Capture => Self::Capture,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFunction {
    pub id: ProgramFunctionId,
    pub source: ProgramSourceId,
    pub name: String,
    pub abi: NativeFunctionAbi,
    pub argument_validations: Vec<NativeFunctionArgumentValidation>,
    pub locals: Vec<NativeLocalMetadata>,
    pub index_expressions: Vec<NativeIndexExpression>,
    pub entry: NativeBlockId,
    pub blocks: Vec<NativeBlock>,
    pub expected_sites: Vec<NativeMirSite>,
}

impl NativeFunction {
    pub fn local_count(&self) -> usize {
        self.locals.len()
    }

    pub fn local(&self, local: NativeLocalId) -> Option<&NativeLocalMetadata> {
        self.locals.get(local.0 as usize)
    }

    pub fn index_expression(&self, local: NativeLocalId) -> Option<&NativeIndexExpression> {
        self.index_expressions
            .binary_search_by_key(&local, |expression| expression.local)
            .ok()
            .map(|index| &self.index_expressions[index])
    }
}
