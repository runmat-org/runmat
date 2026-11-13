use runmat_types::{
    CallableFact, CapabilitySet, ClassDeclaration, EffectSet, MemberName, ProgramFunctionId,
    ValueFact,
};
use serde::{Deserialize, Serialize};

/// The semantic input read by a capture, not the presence of a redundant VM
/// copy. Anonymous snapshots and shared lexical cells have distinct authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionCaptureSource {
    ByValue,
    Lexical(runmat_types::BindingId),
    CanonicalStorage(runmat_types::BindingId),
    Unknown,
}

/// One ordered compiler capture contract, pairing the actual source with its
/// value facts. The generic CallableFact capture vector is its value projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCaptureAnalysis {
    pub source: FunctionCaptureSource,
    pub value: ValueFact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionConvergence {
    Exact,
    Widened,
    DynamicRecursion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionAnalysis {
    pub function: ProgramFunctionId,
    pub callable: CallableFact,
    pub captures: Vec<FunctionCaptureAnalysis>,
    pub outputs: Vec<ValueFact>,
    pub effects: EffectSet,
    /// Compiler proof that every suspension origin is deterministic data-local
    /// work. Missing/old facts fail closed; MaySuspend remains in `effects`.
    #[serde(default)]
    pub cooperative_suspension: bool,
    pub capabilities: CapabilitySet,
    pub convergence: FunctionConvergence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassPropertyAnalysis {
    pub property: MemberName,
    pub fact: ValueFact,
    pub has_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassAnalysis {
    pub declaration: ClassDeclaration,
    pub properties: Vec<ClassPropertyAnalysis>,
    pub methods: Vec<ProgramFunctionId>,
}
