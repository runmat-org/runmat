pub use runmat_builtins::Type;

mod diagnostic;
mod error;
mod hir;
mod ids;
pub(crate) mod inference;
pub(crate) mod lowering;
mod lowering_context;
pub mod remapping;
mod span;
mod validation;

pub mod compatibility {
    pub use crate::hir::{
        CompatibilityHirClassMember as HirClassMember, CompatibilityHirExpr as HirExpr,
        CompatibilityHirExprKind as HirExprKind, CompatibilityHirLValue as HirLValue,
        CompatibilityHirProgram as HirProgram, CompatibilityHirStmt as HirStmt,
        CompatibilityLoweringResult as LoweringResult,
    };
    pub use crate::inference::function_outputs::infer_function_output_types;
    pub use crate::inference::function_vars::infer_function_variable_types;
    pub use crate::inference::globals::infer_global_variable_types;
    pub use crate::lowering::ctx::lower_compatibility as lower;
}

pub use diagnostic::{
    HirDiagnostic, HirDiagnosticNote, HirDiagnosticSeverity, HirDiagnosticSpan,
    HirDiagnosticSuggestion,
};
pub use error::{set_error_namespace, SemanticError};
pub use hir::{
    AggregateKind, AssignmentCreationPolicy, AssignmentShapePolicy, AsyncValueFact, BindingName,
    BindingOwner, BindingResolution, BindingRole, BindingStorage, BuiltinId, CallKind,
    CallResolution, CallSyntax, CallableFallbackPolicy, CallableIdentity, CapturedBinding,
    ClassArgumentBlock, ClassEnumeration, ClassEvent, ClassKind, ClassMethod, ClassProperty,
    ClassResolution, CommandArgument, CommandOptionName, CompatibilityMode, DefPath,
    DefPathSegment, DimFact, DimSymbol, EmptyArrayRole, EntrypointName, EntrypointOrigin,
    EntrypointPolicy, EnvironmentEffect, EvaluationContext, ExpansionSemantics, FunctionAbi,
    FunctionHandleTarget, FunctionKind, FunctionModifiers, FunctionName, FunctionResolution,
    FutureFact, FutureStateFact, HirAssembly, HirBinding, HirBlock, HirCall, HirCallableRef,
    HirClass, HirCommandCall, HirEntrypoint, HirExpr, HirExprKind, HirFunction, HirImport,
    HirModule, HirPlace, HirStmt, HirStmtKind, ImportResolution, IndexComponent, IndexKind,
    IndexResultContext, IndexingSemantics, LoopIterationSemantics, LoweringResult, MemberAccess,
    MemberName, MethodAttributes, MethodId, MethodName, NumericClass, NumericDomain, OperatorKind,
    OutputTarget, OutputTargetList, PackageName, PlaceMutation, PlaceMutationKind,
    PropertyAttributes, QualifiedName, ReferenceKind, ReferenceResolution, RequestedOutputCount,
    RequestedOutputs, SemanticIndex, ShapeFact, SourceUnitKind, SpawnSafetyFact, SpawnSafetyReason,
    StringLiteral, SymbolName, SyntheticName, TaskHandleFact, TensorElementDomainFact,
    TensorTypeFact, TypeFact, ValueFlowFact, WorkspaceEffect, WorkspaceExportPolicy,
    WorkspaceVisibility, AWAIT_EXTENSION_NAME, DISCARD_OUTPUT_NAME, FEVAL_BUILTIN_NAME,
    NARGIN_BUILTIN_NAME, NARGOUT_BUILTIN_NAME, SPAWN_EXTENSION_NAME,
    TEST_CLASS_REGISTRATION_BUILTIN_NAME,
};
pub use ids::{
    BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, SourceId, StmtId, VarId,
};
pub use lowering::lower;
pub use lowering_context::LoweringContext;
pub use span::{merge_span, Span};
