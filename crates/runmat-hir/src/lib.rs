pub use runmat_builtins::Type;
pub use runmat_parser::{IntegerLiteral, IntegerLiteralClass};

mod diagnostic;
mod error;
mod hir;
mod ids;
pub(crate) mod lowering;
mod lowering_context;
pub mod parallel;
mod span;
pub mod testing;

pub use diagnostic::{
    HirDiagnostic, HirDiagnosticNote, HirDiagnosticSeverity, HirDiagnosticSpan,
    HirDiagnosticSuggestion,
};
pub use error::{set_error_namespace, HirError};
pub use hir::{
    AssignmentCreationPolicy, AssignmentShapePolicy, BindingName, BindingOwner, BindingResolution,
    BindingRole, BindingStorage, BuiltinId, CallKind, CallResolution, CallSyntax,
    CallableFallbackPolicy, CallableIdentity, CapturedBinding, ClassArgumentBlock,
    ClassPropertyDefault, ClassResolution, CommandArgument, DefPath, DefPathSegment,
    EmptyArrayRole, EntrypointName, EntrypointOrigin, EntrypointPolicy, EnvironmentEffect,
    ExpansionSemantics, FunctionAbi, FunctionArgumentValidation, FunctionHandleResolution,
    FunctionHandleTarget, FunctionKind, FunctionModifiers, FunctionName, FunctionResolution,
    HirAssembly, HirBinding, HirBlock, HirCall, HirCallableRef, HirClass, HirCommandCall,
    HirEntrypoint, HirExpr, HirExprKind, HirFunction, HirImport, HirIndex, HirModule, HirPlace,
    HirScriptSection, HirStmt, HirStmtKind, ImportResolution, IndexComponent, IndexKind,
    IndexResultContext, IndexingSemantics, LoweringResult, MemberName, MethodId, MethodName,
    OperatorKind, OutputTarget, OutputTargetList, PackageName, PlaceMutation, PlaceMutationKind,
    QualifiedName, ReferenceKind, ReferenceResolution, RequestedOutputCount, SourceUnitKind,
    StringLiteral, SymbolName, WorkspaceEffect, WorkspaceExportPolicy, WorkspaceVisibility,
    ASSIGNIN_BUILTIN_NAME, AWAIT_EXTENSION_NAME, DISCARD_OUTPUT_NAME, EVALC_BUILTIN_NAME,
    EVALIN_BUILTIN_NAME, EVAL_BUILTIN_NAME, FEVAL_BUILTIN_NAME, NARGINCHK_BUILTIN_NAME,
    NARGIN_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME, NARGOUT_BUILTIN_NAME, RUNTESTS_BUILTIN_NAME,
    RUN_BUILTIN_NAME, SPAWN_EXTENSION_NAME, TEST_CLASS_REGISTRATION_BUILTIN_NAME,
};
pub use ids::{BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, SourceId, StmtId};
pub use lowering::lower;
pub use lowering_context::{FunctionOutputArity, LoweringContext};
pub use runmat_types::{
    ClassDeclaration, ClassKind, EnumerationDeclaration, EventDeclaration, InheritanceDeclaration,
    MemberAccess, MethodAttributes, MethodDeclaration, PropertyAttributes, PropertyDeclaration,
    SemanticAttribute, SpawnSafetyFact, SpawnSafetyReason,
};
pub use span::{merge_span, Span};
