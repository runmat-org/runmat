pub use runmat_builtins::Type;

mod diagnostic;
mod error;
mod hir;
mod ids;
pub(crate) mod lowering;
mod lowering_context;
mod span;

pub use diagnostic::{
    HirDiagnostic, HirDiagnosticNote, HirDiagnosticSeverity, HirDiagnosticSpan,
    HirDiagnosticSuggestion,
};
pub use error::{set_error_namespace, HirError};
pub use hir::{
    AssignmentCreationPolicy, AssignmentShapePolicy, AsyncValueFact, BindingName, BindingOwner,
    BindingResolution, BindingRole, BindingStorage, BuiltinId, CallKind, CallResolution,
    CallSyntax, CallableFallbackPolicy, CallableIdentity, CapturedBinding, ClassArgumentBlock,
    ClassEnumeration, ClassEvent, ClassKind, ClassMethod, ClassProperty, ClassResolution,
    CommandArgument, DefPath, DefPathSegment, DimFact, DimSymbol, EmptyArrayRole, EntrypointName,
    EntrypointOrigin, EntrypointPolicy, EnvironmentEffect, ExpansionSemantics, FunctionAbi,
    FunctionArgDefaultValue, FunctionArgDim, FunctionArgSizeSpec, FunctionArgValidator,
    FunctionArgumentValidation, FunctionHandleTarget, FunctionKind, FunctionModifiers,
    FunctionName, FunctionResolution, FutureFact, FutureStateFact, HirAssembly, HirBinding,
    HirBlock, HirCall, HirCallableRef, HirClass, HirCommandCall, HirEntrypoint, HirExpr,
    HirExprKind, HirFunction, HirImport, HirIndex, HirModule, HirPlace, HirStmt, HirStmtKind,
    ImportResolution, IndexComponent, IndexKind, IndexResultContext, IndexingSemantics,
    LoweringResult, MemberAccess, MemberName, MethodAttributes, MethodId, MethodName, NumericClass,
    NumericDomain, OperatorKind, OutputTarget, OutputTargetList, PackageName, PlaceMutation,
    PlaceMutationKind, PropertyAttributes, QualifiedName, ReferenceKind, ReferenceResolution,
    RequestedOutputCount, ShapeFact, SourceUnitKind, SpawnSafetyFact, SpawnSafetyReason,
    StringLiteral, SymbolName, TaskHandleFact, TensorElementDomainFact, TensorTypeFact, TypeFact,
    ValueFlowFact, WorkspaceEffect, WorkspaceExportPolicy, WorkspaceVisibility,
    AWAIT_EXTENSION_NAME, DISCARD_OUTPUT_NAME, FEVAL_BUILTIN_NAME, NARGINCHK_BUILTIN_NAME,
    NARGIN_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME, NARGOUT_BUILTIN_NAME, SPAWN_EXTENSION_NAME,
    TEST_CLASS_REGISTRATION_BUILTIN_NAME,
};
pub use ids::{BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, SourceId, StmtId};
pub use lowering::lower;
pub use lowering_context::LoweringContext;
pub use span::{merge_span, Span};
