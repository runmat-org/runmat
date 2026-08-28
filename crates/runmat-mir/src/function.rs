use runmat_hir::{
    CapturedBinding, ClassId, FunctionArgumentValidation, FunctionId, FunctionKind,
    FunctionModifiers, FunctionName, Span,
};
use runmat_types::ProgramSourceId;
use serde::{Deserialize, Serialize};

/// Immutable frontend metadata required by MIR-owned semantic analysis.
///
/// Executable control flow remains in [`crate::MirBody`]. Keeping this compact
/// record beside it means analysis never needs a reverse query into HIR and does
/// not duplicate class or argument-validation rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirFunctionMetadata {
    pub source: ProgramSourceId,
    pub name: FunctionName,
    pub parent: Option<FunctionId>,
    pub enclosing_class: Option<ClassId>,
    pub kind: FunctionKind,
    pub argument_validations: Vec<FunctionArgumentValidation>,
    pub captures: Vec<CapturedBinding>,
    pub modifiers: FunctionModifiers,
    pub span: Span,
}
