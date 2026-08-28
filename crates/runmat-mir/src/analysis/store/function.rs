use runmat_types::{
    CallableFact, CapabilitySet, ClassDeclaration, EffectSet, MemberName, ProgramFunctionId,
    ValueFact,
};
use serde::{Deserialize, Serialize};

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
    pub outputs: Vec<ValueFact>,
    pub effects: EffectSet,
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
