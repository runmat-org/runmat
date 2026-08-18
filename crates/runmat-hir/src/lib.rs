pub use runmat_builtins::Type;
pub use runmat_parser::{IntegerLiteral, IntegerLiteralClass};

pub fn integer_literal_to_int_value(literal: &IntegerLiteral) -> runmat_builtins::IntValue {
    use runmat_builtins::IntValue;

    match literal.class() {
        IntegerLiteralClass::Int8 => IntValue::I8(literal.bits() as i8),
        IntegerLiteralClass::Int16 => IntValue::I16(literal.bits() as i16),
        IntegerLiteralClass::Int32 => IntValue::I32(literal.bits() as i32),
        IntegerLiteralClass::Int64 => IntValue::I64(literal.bits() as i64),
        IntegerLiteralClass::UInt8 => IntValue::U8(literal.bits() as u8),
        IntegerLiteralClass::UInt16 => IntValue::U16(literal.bits() as u16),
        IntegerLiteralClass::UInt32 => IntValue::U32(literal.bits() as u32),
        IntegerLiteralClass::UInt64 => IntValue::U64(literal.bits()),
    }
}

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
    FunctionArgDefaultValue, FunctionArgDim, FunctionArgRangeInclusivity, FunctionArgSizeSpec,
    FunctionArgValidationLiteral, FunctionArgValidator, FunctionArgumentValidation,
    FunctionHandleTarget, FunctionKind, FunctionModifiers, FunctionName, FunctionResolution,
    FutureFact, FutureStateFact, HirAssembly, HirBinding, HirBlock, HirCall, HirCallableRef,
    HirClass, HirCommandCall, HirEntrypoint, HirExpr, HirExprKind, HirFunction, HirImport,
    HirIndex, HirModule, HirPlace, HirStmt, HirStmtKind, ImportResolution, IndexComponent,
    IndexKind, IndexResultContext, IndexingSemantics, LoweringResult, MemberAccess, MemberName,
    MethodAttributes, MethodId, MethodName, NumericClass, NumericDomain, OperatorKind,
    OutputTarget, OutputTargetList, PackageName, PlaceMutation, PlaceMutationKind,
    PropertyAttributes, QualifiedName, ReferenceKind, ReferenceResolution, RequestedOutputCount,
    ShapeFact, SourceUnitKind, SpawnSafetyFact, SpawnSafetyReason, StringLiteral, SymbolName,
    TaskHandleFact, TensorElementDomainFact, TensorTypeFact, TypeFact, ValueFlowFact,
    WorkspaceEffect, WorkspaceExportPolicy, WorkspaceVisibility, ASSIGNIN_BUILTIN_NAME,
    AWAIT_EXTENSION_NAME, DISCARD_OUTPUT_NAME, EVALC_BUILTIN_NAME, EVALIN_BUILTIN_NAME,
    EVAL_BUILTIN_NAME, FEVAL_BUILTIN_NAME, NARGINCHK_BUILTIN_NAME, NARGIN_BUILTIN_NAME,
    NARGOUTCHK_BUILTIN_NAME, NARGOUT_BUILTIN_NAME, RUNTESTS_BUILTIN_NAME, RUN_BUILTIN_NAME,
    SPAWN_EXTENSION_NAME, TEST_CLASS_REGISTRATION_BUILTIN_NAME,
};
pub use ids::{BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, SourceId, StmtId};
pub use lowering::lower;
pub use lowering_context::{FunctionOutputArity, LoweringContext};
pub use span::{merge_span, Span};
