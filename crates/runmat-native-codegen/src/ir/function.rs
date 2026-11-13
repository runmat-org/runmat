use super::{NativeBlock, NativeBlockId, NativeLocalId, NativeMirSite};
use runmat_types::{
    BindingId, CapabilitySet, FunctionArgDefaultValue, FunctionArgSizeSpec, FunctionArgValidator,
    FunctionId, ProgramSourceId,
};
use serde::{Deserialize, Serialize};

mod continuation_geometry;
pub use continuation_geometry::NativeExpressionContinuationGeometry;

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

/// Deterministic semantic-binding lookup for a Native frame.
///
/// `locals` remain indexed by Native local identity. This separately sorted
/// view lets callable issuance resolve a callee capture in the creating frame
/// without rescanning all locals or confusing the two frames' local ordinals.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBindingLocal {
    pub binding: BindingId,
    pub local: NativeLocalId,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeLexicalBindingKind {
    Owned,
    Inherited,
    Transit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLexicalBinding {
    pub binding: BindingId,
    pub local: NativeLocalId,
    pub kind: NativeLexicalBindingKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeInheritedStorageKind {
    Global,
    Persistent,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInheritedStorageBinding {
    pub binding: BindingId,
    pub local: NativeLocalId,
    pub owner: FunctionId,
    pub name: String,
    pub owner_function: String,
    pub kind: NativeInheritedStorageKind,
}

/// One callable capture in canonical compiler order.
///
/// The disposition identifies the single value authority for the binding.
/// A MIR `Capture` local is also the positional transport identity for every
/// callable capture, including bindings whose values remain owned by a live
/// lexical activation or canonical storage; that local classification is not
/// a competing value authority. `ByValue` and `TransitSnapshot` captures seed
/// invocation locals, while inherited lexical and canonical-storage captures
/// remain owned by their live authorities.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeClosureCapture {
    pub binding: BindingId,
    pub local: NativeLocalId,
    pub disposition: NativeClosureCaptureDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeClosureCaptureDisposition {
    /// Ordinary immutable capture installed from the callable snapshot.
    ByValue,
    /// Nested-function binding read from its live ancestor activation.
    Lexical,
    /// Anonymous function's own by-value snapshot. Its retained activation is
    /// independently carried for lexically nested descendants.
    TransitSnapshot,
    /// Global or persistent binding resolved through canonical storage.
    InheritedStorage,
}

impl From<&runmat_mir::MirLocalKind> for NativeLocalKind {
    fn from(kind: &runmat_mir::MirLocalKind) -> Self {
        match kind {
            runmat_mir::MirLocalKind::Parameter => Self::Parameter,
            runmat_mir::MirLocalKind::Output => Self::Output,
            runmat_mir::MirLocalKind::Binding => Self::Binding,
            runmat_mir::MirLocalKind::IncomingWorkspace => Self::Binding,
            // Named runtime lookup uses the same refreshable storage as an
            // incoming workspace binding; MIR retains its assignment facts.
            runmat_mir::MirLocalKind::ExternalWorkspace => Self::Binding,
            runmat_mir::MirLocalKind::Temporary => Self::Temporary,
            runmat_mir::MirLocalKind::Capture => Self::Capture,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFunction {
    pub id: FunctionId,
    /// Stable portable identity used only by program points and source observations.
    pub portable_id: runmat_types::ProgramFunctionId,
    /// SHA-256 of the canonical MIR (function metadata, body) pair. This is
    /// source identity, not a hash of generated machine code.
    pub mir_body_digest: runmat_execution::Digest,
    pub source: ProgramSourceId,
    pub name: String,
    pub source_role: runmat_types::FunctionSourceRole,
    /// Compiler/session coordinate of the lexical parent, if this is nested.
    pub lexical_parent: Option<FunctionId>,
    pub class_method_owner: Option<runmat_types::ClassMethodOwner>,
    /// Whole-function requirements derived by canonical MIR analysis. Native
    /// executors use this contract when a function becomes an async target;
    /// they do not reconstruct semantics from lowered instructions.
    pub capabilities: CapabilitySet,
    /// Canonical capture policy derived from the same source MIR body.
    pub spawn_safety: runmat_types::SpawnSafetyFact,
    pub abi: NativeFunctionAbi,
    pub argument_validations: Vec<NativeFunctionArgumentValidation>,
    pub locals: Vec<NativeLocalMetadata>,
    pub binding_locals: Vec<NativeBindingLocal>,
    /// Compiler-ordered capture ABI of the callable value. Shared lexical and
    /// canonical-storage captures remain in this list even though their Native
    /// frame input comes from the corresponding live authority rather than a
    /// by-value `Capture` local.
    pub closure_captures: Vec<NativeClosureCapture>,
    /// Projected once from canonical MIR binding ownership, not rediscovered
    /// from display names or native frame snapshots.
    pub lexical_bindings: Vec<NativeLexicalBinding>,
    pub inherited_storage_bindings: Vec<NativeInheritedStorageBinding>,
    pub creates_lexical_environment: bool,
    pub entry: NativeBlockId,
    pub blocks: Vec<NativeBlock>,
    pub expression_continuation_geometry: NativeExpressionContinuationGeometry,
    /// Exact analysis-certified requirements for every lowered MIR site.
    pub operation_contracts: Vec<runmat_mir::MirOperationContract>,
    /// Ordered execution projection of `operation_contracts`, retained by the
    /// resume engine. Verification requires byte-for-byte site equality; this
    /// is not an independent semantic registry.
    pub expected_sites: Vec<NativeMirSite>,
}

impl NativeFunction {
    pub fn local_count(&self) -> usize {
        self.locals.len()
    }

    pub fn local(&self, local: NativeLocalId) -> Option<&NativeLocalMetadata> {
        self.locals.get(local.0 as usize)
    }

    pub fn local_for_binding(&self, binding: BindingId) -> Option<NativeLocalId> {
        self.binding_locals
            .binary_search_by_key(&binding, |entry| entry.binding)
            .ok()
            .map(|index| self.binding_locals[index].local)
    }
}
