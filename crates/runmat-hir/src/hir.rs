use crate::{
    BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, SourceId, Span, StmtId, Type,
    VarId,
};
use runmat_parser as parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Canonical semantic HIR product for one compiled source set.
///
/// The assembly owns the tables for modules, functions, classes, bindings, and
/// entrypoints. Table IDs are local to this assembly; stable identities for
/// packages/modules/items are represented separately by `DefPath` and qualified
/// name types.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct HirAssembly {
    pub modules: Vec<HirModule>,
    pub functions: Vec<HirFunction>,
    pub classes: Vec<HirClass>,
    pub bindings: Vec<HirBinding>,
    pub entrypoints: Vec<HirEntrypoint>,
}

/// Source unit metadata plus references to module-owned semantic items.
///
/// Top-level functions, classes, and synthetic script entry functions live in
/// assembly tables and are referenced here by local IDs rather than embedded as
/// statement variants.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirModule {
    pub id: ModuleId,
    pub name: QualifiedName,
    pub source_id: SourceId,
    pub source_unit: SourceUnitKind,
    pub imports: Vec<HirImport>,
    pub top_level_functions: Vec<FunctionId>,
    pub classes: Vec<ClassId>,
    pub synthetic_entry_function: Option<FunctionId>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum SourceUnitKind {
    ScriptFile,
    FunctionFile,
    ClassFile,
    PackageFunctionFile,
    ClassFolderMethodFile,
    ReplSubmission,
    NotebookCell,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirEntrypoint {
    pub id: EntrypointId,
    pub name: Option<EntrypointName>,
    pub target: FunctionId,
    pub origin: EntrypointOrigin,
    pub policy: EntrypointPolicy,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum EntrypointOrigin {
    ProjectDeclared,
    SourcePath,
    ReplSubmission,
    NotebookCell,
    HostSynthetic,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct EntrypointPolicy {
    pub workspace_export: WorkspaceExportPolicy,
    pub top_level_await: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum WorkspaceExportPolicy {
    ExportTopLevelBindings,
    ReturnFunctionOutputs,
    HostDefined,
}

/// Uniform executable representation for MATLAB functions and generated entrypoints.
///
/// Named functions, nested functions, anonymous functions, class methods, and
/// synthetic script entrypoints all use this shape. Parameters, outputs, locals,
/// and captures reference semantic `BindingId`s; VM slots are intentionally not
/// represented in core HIR.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirFunction {
    pub id: FunctionId,
    pub module: ModuleId,
    pub parent: Option<FunctionId>,
    pub enclosing_class: Option<ClassId>,
    pub name: FunctionName,
    pub kind: FunctionKind,
    pub params: Vec<BindingId>,
    pub outputs: Vec<BindingId>,
    pub abi: FunctionAbi,
    pub locals: Vec<BindingId>,
    pub captures: Vec<CapturedBinding>,
    pub modifiers: FunctionModifiers,
    pub body: HirBlock,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionKind {
    Named,
    Anonymous,
    SyntheticEntrypoint,
    ClassMethod { is_static: bool },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct FunctionModifiers {
    pub isolated: bool,
    pub is_async: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct FunctionAbi {
    pub fixed_inputs: Vec<BindingId>,
    pub varargin: Option<BindingId>,
    pub fixed_outputs: Vec<BindingId>,
    pub varargout: Option<BindingId>,
    pub implicit_nargin: Option<BindingId>,
    pub implicit_nargout: Option<BindingId>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct CapturedBinding {
    pub binding: BindingId,
    pub from_function: FunctionId,
}

/// Semantic binding identity for a name owned by a module or function.
///
/// Bindings model MATLAB lexical/global/persistent storage and workspace
/// visibility. They are not VM slot numbers and should be used instead of the
/// legacy `VarId` in semantic HIR.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirBinding {
    pub id: BindingId,
    pub owner: BindingOwner,
    pub name: BindingName,
    pub role: BindingRole,
    pub storage: BindingStorage,
    pub workspace_visibility: WorkspaceVisibility,
    pub declared_span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum BindingOwner {
    Module(ModuleId),
    Function(FunctionId),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum BindingRole {
    Parameter,
    Output,
    Local,
    ModuleBinding,
    ImplicitAns,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum BindingStorage {
    Lexical,
    Global,
    Persistent,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum WorkspaceVisibility {
    Hidden,
    TopLevel,
    ModuleVisible,
    ImplicitAns,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirBlock {
    pub statements: Vec<HirStmt>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirStmt {
    pub id: StmtId,
    pub kind: HirStmtKind,
    pub span: Span,
}

impl HirStmt {
    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirStmtKind {
    ExprStmt(HirExpr, bool),
    Assign(HirPlace, HirExpr, bool),
    MultiAssign(OutputTargetList, HirExpr, bool),
    If {
        cond: HirExpr,
        then_body: HirBlock,
        elseif_blocks: Vec<(HirExpr, HirBlock)>,
        else_body: Option<HirBlock>,
    },
    While {
        cond: HirExpr,
        body: HirBlock,
    },
    For {
        binding: BindingId,
        range: HirExpr,
        body: HirBlock,
        semantics: LoopIterationSemantics,
    },
    Switch {
        expr: HirExpr,
        cases: Vec<(HirExpr, HirBlock)>,
        otherwise: Option<HirBlock>,
    },
    TryCatch {
        try_body: HirBlock,
        catch_binding: Option<BindingId>,
        catch_body: HirBlock,
    },
    Global(Vec<BindingId>),
    Persistent(Vec<BindingId>),
    Break,
    Continue,
    Return,
    Import(HirImport),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirExpr {
    pub id: ExprId,
    pub kind: HirExprKind,
    pub span: Span,
}

impl HirExpr {
    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirExprKind {
    Number(String),
    String(StringLiteral),
    Constant(SymbolName),
    Binding(BindingId),
    Unary(OperatorKind, Box<HirExpr>),
    Binary(Box<HirExpr>, OperatorKind, Box<HirExpr>),
    Tensor(Vec<Vec<HirExpr>>),
    Cell(Vec<Vec<HirExpr>>),
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
    End,
    Index(Box<HirExpr>, IndexingSemantics),
    Member(Box<HirExpr>, MemberName),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    Call(HirCall),
    CommandCall(HirCommandCall),
    FunctionHandle(FunctionHandleTarget),
    AnonymousFunction(FunctionId),
    MetaClass(QualifiedName),
    Await(Box<HirExpr>),
    Spawn(Box<HirExpr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirPlace {
    Binding(BindingId),
    Member(Box<HirExpr>, MemberName),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    Index(Box<HirExpr>, IndexingSemantics),
    IndexCell(Box<HirExpr>, IndexingSemantics),
}

/// Call expression with a semantic callee reference and source syntax marker.
///
/// Unresolved or dynamic calls remain explicit, so the HIR no longer relies on a
/// string-only `FuncCall` variant as the primary call representation.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirCall {
    pub callee: HirCallableRef,
    pub args: Vec<HirExpr>,
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirCallableRef {
    Function(FunctionId),
    ClassConstructor(ClassId),
    Builtin(BuiltinId),
    Imported(DefPath),
    DynamicExpr(Box<HirExpr>),
    Unresolved(QualifiedName),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum CallSyntax {
    Plain,
    Method,
    DottedInvoke,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirCommandCall {
    pub command: HirCallableRef,
    pub args: Vec<CommandArgument>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum CommandArgument {
    Word(SymbolName),
    StringLiteral(StringLiteral),
    OptionToken(CommandOptionName),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirImport {
    pub path: QualifiedName,
    pub wildcard: bool,
    pub span: Span,
}

/// Semantic class metadata owned by the assembly.
///
/// Methods reference `HirFunction` entries by `FunctionId`; class definitions are
/// not durable statement variants in the semantic model.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirClass {
    pub id: ClassId,
    pub module: ModuleId,
    pub name: QualifiedName,
    pub super_class: Option<ClassId>,
    pub kind: ClassKind,
    pub properties: Vec<ClassProperty>,
    pub methods: Vec<ClassMethod>,
    pub events: Vec<ClassEvent>,
    pub enumerations: Vec<ClassEnumeration>,
    pub arguments: Vec<ClassArgumentBlock>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ClassKind {
    Value,
    Handle,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassProperty {
    pub name: MemberName,
    pub attributes: PropertyAttributes,
    pub default: Option<HirExpr>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassMethod {
    pub function: FunctionId,
    pub name: MethodName,
    pub is_static: bool,
    pub attributes: MethodAttributes,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassEvent {
    pub name: SymbolName,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassEnumeration {
    pub name: SymbolName,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassArgumentBlock {
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct PropertyAttributes {
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub is_transient: bool,
    pub is_hidden: bool,
    pub access: MemberAccess,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct MethodAttributes {
    pub access: MemberAccess,
    pub is_hidden: bool,
    pub is_abstract: bool,
    pub is_sealed: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub enum MemberAccess {
    #[default]
    Public,
    Private,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct OutputTargetList {
    pub targets: Vec<OutputTarget>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum OutputTarget {
    Place(HirPlace),
    Discard,
    VarargoutExpansion,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum RequestedOutputCount {
    Zero,
    One,
    Exactly(usize),
    AtLeast(usize),
    UnknownDynamic,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum IndexKind {
    Paren,
    Brace,
    Dot,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum IndexComponent {
    Colon,
    End { dim: Option<usize>, offset: isize },
    Expr(HirExpr),
    Logical(HirExpr),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct IndexingSemantics {
    pub kind: IndexKind,
    pub components: Vec<IndexComponent>,
    pub result_context: IndexResultContext,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum IndexResultContext {
    ReadSingle,
    ReadCommaList,
    AssignmentTarget,
    DeletionTarget,
    FunctionArgumentExpansion,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionHandleTarget {
    Function(FunctionId),
    Builtin(BuiltinId),
    Method(MethodId),
    Anonymous(FunctionId),
    DefPath(DefPath),
    DynamicName(SymbolName),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum WorkspaceEffect {
    None,
    ReadsWorkspace,
    CreatesBinding,
    ClearsBinding,
    ClearsFunctionCache,
    MutatesGlobal,
    MutatesPersistent,
    LoadsExternalBindings,
    DynamicEval,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum EnvironmentEffect {
    PathMutation,
    WorkingDirectoryMutation,
    FunctionCacheInvalidation,
    DynamicLookupInvalidation,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum EmptyArrayRole {
    EmptyValue,
    ConcatenationIdentity,
    DeletionMarker,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ExpansionSemantics {
    ExactShape,
    ScalarExpansion,
    ImplicitExpansion,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum OperatorKind {
    UnaryPlus,
    UnaryMinus,
    Not,
    Add,
    Subtract,
    MatrixMultiply,
    ElementwiseMultiply,
    MatrixPower,
    ElementwisePower,
    Mldivide,
    Mrdivide,
    ElementwiseDivide,
    ElementwiseLeftDivide,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    ShortCircuitAnd,
    ShortCircuitOr,
    ElementwiseAnd,
    ElementwiseOr,
    Transpose,
    ConjugateTranspose,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum NumericClass {
    Double,
    Single,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum AggregateKind {
    Struct,
    Cell,
    ObjectArray(ClassId),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LoopIterationSemantics {
    ForColumns,
    WhileCondition,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum CompatibilityMode {
    MatlabStrict,
    RunMatExtended,
    Interactive,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum EvaluationContext {
    Statement,
    Expression,
    AssignmentRhs { targets: OutputTargetList },
    FunctionArgument,
    ConcatElement,
    CellElement,
    IndexOperand,
    CommandArgument,
    NameValueArgument,
    LineSpecArgument,
    DynamicFieldName,
    ForRange,
    Condition,
    SwitchExpression,
    CaseExpression,
    ReturnValue { requested_outputs: RequestedOutputs },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RequestedOutputs {
    pub count: RequestedOutputCount,
    pub targets: OutputTargetList,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ValueFlowFact {
    NoValue,
    Single(TypeFact),
    CommaList(Vec<TypeFact>),
    UnknownList,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum PlaceMutationKind {
    BindOrAssign,
    IndexedAssign,
    CellAssign,
    MemberAssign,
    Delete,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct PlaceMutation {
    pub place: HirPlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum AssignmentCreationPolicy {
    ExistingOnly,
    CreateBinding,
    CreateArrayByIndex,
    CreateStructFieldPath,
    Overloaded,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum AssignmentShapePolicy {
    Exact,
    ScalarExpansion,
    MatlabCompatible,
    Overloaded,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ReferenceKind {
    Binding(BindingId),
    Function(FunctionId),
    Builtin(BuiltinId),
    Class(ClassId),
    Package(QualifiedName),
    Imported(DefPath),
    RuntimeClass(QualifiedName),
    Dynamic,
    Unresolved,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum CallKind {
    DirectFunction(FunctionId),
    Builtin(BuiltinId),
    Constructor(ClassId),
    StaticMethod {
        class: ClassId,
        method: MethodId,
    },
    InstanceMethod {
        receiver: Box<HirExpr>,
        method: MethodId,
    },
    PackageFunction(DefPath),
    FunctionHandle,
    Dynamic,
    OverloadedOperator,
    OverloadedIndexing,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct SemanticIndex {
    pub bindings: Vec<BindingResolution>,
    pub functions: Vec<FunctionResolution>,
    pub classes: Vec<ClassResolution>,
    pub imports: Vec<ImportResolution>,
    pub references: Vec<ReferenceResolution>,
    pub calls: Vec<CallResolution>,
    pub mutations: Vec<PlaceMutation>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct BindingResolution {
    pub name: BindingName,
    pub binding: BindingId,
    pub owner: BindingOwner,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct FunctionResolution {
    pub name: FunctionName,
    pub function: FunctionId,
    pub parent: Option<FunctionId>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassResolution {
    pub name: QualifiedName,
    pub class: ClassId,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ImportResolution {
    pub import: HirImport,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ReferenceResolution {
    pub name: SymbolName,
    pub kind: ReferenceKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct CallResolution {
    pub name: QualifiedName,
    pub callee: HirCallableRef,
    pub kind: CallKind,
    pub requested_outputs: RequestedOutputCount,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum SpawnSafetyFact {
    SpawnSafe,
    RequiresIsolation,
    NotSpawnSafe { reason: SpawnSafetyReason },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum SpawnSafetyReason {
    MutableLexicalCapture,
    NonSendableRuntimeHandle,
    UnsynchronizedSharedMutation,
    UnknownDynamicCapture,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum AsyncValueFact {
    Future(FutureFact),
    TaskHandle(TaskHandleFact),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct FutureFact {
    pub output: Box<TypeFact>,
    pub state: FutureStateFact,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FutureStateFact {
    Lazy,
    Awaited,
    Unknown,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct TaskHandleFact {
    pub output: Box<TypeFact>,
    pub spawn_safety: SpawnSafetyFact,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum TypeFact {
    Never,
    Unknown,
    Logical,
    Numeric {
        class: NumericClass,
        domain: NumericDomain,
    },
    Tensor(TensorTypeFact),
    String,
    CharArray,
    Cell,
    Struct,
    ClassInstance(ClassId),
    ClassRef(ClassId),
    Function(FunctionId),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct TensorTypeFact {
    pub element: TensorElementDomainFact,
    pub shape: ShapeFact,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum TensorElementDomainFact {
    Unknown,
    Logical,
    Numeric {
        class: NumericClass,
        domain: NumericDomain,
    },
    Char,
    Object(ClassId),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum NumericDomain {
    Real,
    Complex,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ShapeFact {
    Unreachable,
    Unknown,
    Scalar,
    Ranked { rank: usize },
    Shaped { dims: Vec<DimFact> },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DimFact {
    Known(usize),
    Symbolic(DimSymbol),
    Unknown,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct DimSymbol(pub String);

/// Stable qualified semantic identity for a package/module/item path.
///
/// Unlike local `ModuleId`/`FunctionId`/`ClassId`/`BindingId` values, a `DefPath`
/// is intended to describe the same semantic item across compiler products when
/// the source/project identity is unchanged.
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct DefPath {
    pub package: PackageName,
    pub module: QualifiedName,
    pub item: Vec<DefPathSegment>,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub enum DefPathSegment {
    Function(SymbolName),
    Class(SymbolName),
    Method(SymbolName),
    Property(SymbolName),
    Entrypoint(EntrypointName),
    Synthetic(SyntheticName),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct QualifiedName(pub Vec<SymbolName>);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct SymbolName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct BindingName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct FunctionName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct EntrypointName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct MemberName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct MethodName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct PackageName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct SyntheticName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct BuiltinId(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct MethodId(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct CommandOptionName(pub String);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct StringLiteral(pub String);

// Legacy HIR retained only for the old lowering/inference pipeline during migration.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct LegacyHirExpr {
    pub kind: LegacyHirExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LegacyHirExprKind {
    Number(String),
    String(String),
    Var(VarId),
    Constant(String),
    Unary(parser::UnOp, Box<LegacyHirExpr>),
    Binary(Box<LegacyHirExpr>, parser::BinOp, Box<LegacyHirExpr>),
    Tensor(Vec<Vec<LegacyHirExpr>>),
    Cell(Vec<Vec<LegacyHirExpr>>),
    Index(Box<LegacyHirExpr>, Vec<LegacyHirExpr>),
    IndexCell(Box<LegacyHirExpr>, Vec<LegacyHirExpr>),
    Range(
        Box<LegacyHirExpr>,
        Option<Box<LegacyHirExpr>>,
        Box<LegacyHirExpr>,
    ),
    Colon,
    End,
    Member(Box<LegacyHirExpr>, String),
    MemberDynamic(Box<LegacyHirExpr>, Box<LegacyHirExpr>),
    DottedInvoke(Box<LegacyHirExpr>, String, Vec<LegacyHirExpr>),
    MethodCall(Box<LegacyHirExpr>, String, Vec<LegacyHirExpr>),
    AnonFunc {
        params: Vec<VarId>,
        body: Box<LegacyHirExpr>,
    },
    FuncHandle(String),
    FuncCall(String, Vec<LegacyHirExpr>),
    MetaClass(String),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LegacyHirStmt {
    ExprStmt(LegacyHirExpr, bool, Span),
    Assign(VarId, LegacyHirExpr, bool, Span),
    MultiAssign(Vec<Option<VarId>>, LegacyHirExpr, bool, Span),
    AssignLValue(LegacyHirLValue, LegacyHirExpr, bool, Span),
    If {
        cond: LegacyHirExpr,
        then_body: Vec<LegacyHirStmt>,
        elseif_blocks: Vec<(LegacyHirExpr, Vec<LegacyHirStmt>)>,
        else_body: Option<Vec<LegacyHirStmt>>,
        span: Span,
    },
    While {
        cond: LegacyHirExpr,
        body: Vec<LegacyHirStmt>,
        span: Span,
    },
    For {
        var: VarId,
        expr: LegacyHirExpr,
        body: Vec<LegacyHirStmt>,
        span: Span,
    },
    Switch {
        expr: LegacyHirExpr,
        cases: Vec<(LegacyHirExpr, Vec<LegacyHirStmt>)>,
        otherwise: Option<Vec<LegacyHirStmt>>,
        span: Span,
    },
    TryCatch {
        try_body: Vec<LegacyHirStmt>,
        catch_var: Option<VarId>,
        catch_body: Vec<LegacyHirStmt>,
        span: Span,
    },
    Global(Vec<(VarId, String)>, Span),
    Persistent(Vec<(VarId, String)>, Span),
    Break(Span),
    Continue(Span),
    Return(Span),
    Function {
        name: String,
        params: Vec<VarId>,
        outputs: Vec<VarId>,
        body: Vec<LegacyHirStmt>,
        has_varargin: bool,
        has_varargout: bool,
        span: Span,
    },
    ClassDef {
        name: String,
        super_class: Option<String>,
        members: Vec<LegacyHirClassMember>,
        span: Span,
    },
    Import {
        path: Vec<String>,
        wildcard: bool,
        span: Span,
    },
}

impl LegacyHirExpr {
    pub fn span(&self) -> Span {
        self.span
    }
}

impl LegacyHirStmt {
    pub fn span(&self) -> Span {
        match self {
            LegacyHirStmt::ExprStmt(_, _, span)
            | LegacyHirStmt::Assign(_, _, _, span)
            | LegacyHirStmt::MultiAssign(_, _, _, span)
            | LegacyHirStmt::AssignLValue(_, _, _, span)
            | LegacyHirStmt::Global(_, span)
            | LegacyHirStmt::Persistent(_, span)
            | LegacyHirStmt::Break(span)
            | LegacyHirStmt::Continue(span)
            | LegacyHirStmt::Return(span) => *span,
            LegacyHirStmt::If { span, .. }
            | LegacyHirStmt::While { span, .. }
            | LegacyHirStmt::For { span, .. }
            | LegacyHirStmt::Switch { span, .. }
            | LegacyHirStmt::TryCatch { span, .. }
            | LegacyHirStmt::Function { span, .. }
            | LegacyHirStmt::ClassDef { span, .. }
            | LegacyHirStmt::Import { span, .. } => *span,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LegacyHirClassMember {
    Properties {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
    Methods {
        attributes: Vec<parser::Attr>,
        body: Vec<LegacyHirStmt>,
    },
    Events {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
    Enumeration {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
    Arguments {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LegacyHirLValue {
    Var(VarId),
    Member(Box<LegacyHirExpr>, String),
    MemberDynamic(Box<LegacyHirExpr>, Box<LegacyHirExpr>),
    Index(Box<LegacyHirExpr>, Vec<LegacyHirExpr>),
    IndexCell(Box<LegacyHirExpr>, Vec<LegacyHirExpr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct LegacyHirProgram {
    pub body: Vec<LegacyHirStmt>,
    #[serde(default)]
    pub var_types: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct LoweringResult {
    pub assembly: HirAssembly,
    pub semantic_index: SemanticIndex,
    pub hir: LegacyHirProgram,
    pub variables: HashMap<String, usize>,
    pub functions: HashMap<String, LegacyHirStmt>,
    pub var_types: Vec<Type>,
    pub var_names: HashMap<VarId, String>,
    /// Temporary compatibility placeholder during Plan 1 migration.
    ///
    /// Lowering no longer computes legacy analysis products. Tests and temporary
    /// consumers that still need these facts should call the legacy inference
    /// helpers explicitly until Plan 2 replaces them with semantic analysis.
    pub inferred_globals: HashMap<VarId, Type>,
    /// See `inferred_globals`.
    pub inferred_function_envs: HashMap<String, HashMap<VarId, Type>>,
    /// See `inferred_globals`.
    pub inferred_function_returns: HashMap<String, Vec<Type>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    #[test]
    fn semantic_assembly_owns_core_items() {
        let module = ModuleId(0);
        let function = FunctionId(0);
        let binding = BindingId(0);
        let entrypoint = EntrypointId(0);

        let assembly = HirAssembly {
            modules: vec![HirModule {
                id: module,
                name: QualifiedName(vec![SymbolName("demo".into())]),
                source_id: SourceId(0),
                source_unit: SourceUnitKind::ScriptFile,
                imports: vec![],
                top_level_functions: vec![],
                classes: vec![],
                synthetic_entry_function: Some(function),
            }],
            functions: vec![HirFunction {
                id: function,
                module,
                parent: None,
                enclosing_class: None,
                name: FunctionName("demo_entry".into()),
                kind: FunctionKind::SyntheticEntrypoint,
                params: vec![],
                outputs: vec![],
                abi: FunctionAbi {
                    fixed_inputs: vec![],
                    varargin: None,
                    fixed_outputs: vec![],
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![binding],
                captures: vec![],
                modifiers: FunctionModifiers::default(),
                body: HirBlock { statements: vec![] },
                span: span(),
            }],
            classes: vec![],
            bindings: vec![HirBinding {
                id: binding,
                owner: BindingOwner::Function(function),
                name: BindingName("x".into()),
                role: BindingRole::Local,
                storage: BindingStorage::Lexical,
                workspace_visibility: WorkspaceVisibility::TopLevel,
                declared_span: span(),
            }],
            entrypoints: vec![HirEntrypoint {
                id: entrypoint,
                name: Some(EntrypointName("demo".into())),
                target: function,
                origin: EntrypointOrigin::SourcePath,
                policy: EntrypointPolicy {
                    workspace_export: WorkspaceExportPolicy::ExportTopLevelBindings,
                    top_level_await: false,
                },
            }],
        };

        assert_eq!(assembly.modules[0].synthetic_entry_function, Some(function));
        assert_eq!(assembly.entrypoints[0].target, function);
        assert_eq!(assembly.functions[0].locals, vec![binding]);
        assert!(matches!(
            assembly.bindings[0].workspace_visibility,
            WorkspaceVisibility::TopLevel
        ));
    }

    #[test]
    fn function_abi_can_reuse_shared_input_output_binding() {
        let binding = BindingId(0);
        let abi = FunctionAbi {
            fixed_inputs: vec![binding],
            varargin: None,
            fixed_outputs: vec![binding],
            varargout: None,
            implicit_nargin: Some(BindingId(1)),
            implicit_nargout: Some(BindingId(2)),
        };

        assert_eq!(abi.fixed_inputs[0], abi.fixed_outputs[0]);
        assert_eq!(abi.implicit_nargin, Some(BindingId(1)));
        assert_eq!(abi.implicit_nargout, Some(BindingId(2)));
    }

    #[test]
    fn captures_reference_original_binding_identity() {
        let parent = FunctionId(0);
        let child = FunctionId(1);
        let binding = BindingId(0);

        let capture = CapturedBinding {
            binding,
            from_function: parent,
        };

        let child_function = HirFunction {
            id: child,
            module: ModuleId(0),
            parent: Some(parent),
            enclosing_class: None,
            name: FunctionName("inner".into()),
            kind: FunctionKind::Named,
            params: vec![],
            outputs: vec![],
            abi: FunctionAbi {
                fixed_inputs: vec![],
                varargin: None,
                fixed_outputs: vec![],
                varargout: None,
                implicit_nargin: None,
                implicit_nargout: None,
            },
            locals: vec![],
            captures: vec![capture],
            modifiers: FunctionModifiers::default(),
            body: HirBlock { statements: vec![] },
            span: span(),
        };

        assert_eq!(child_function.parent, Some(parent));
        assert_eq!(child_function.captures[0].binding, binding);
        assert_eq!(child_function.captures[0].from_function, parent);
    }

    #[test]
    fn semantic_facts_capture_value_flow_and_mutation_context() {
        let binding = BindingId(0);
        let mutation = PlaceMutation {
            place: HirPlace::Binding(binding),
            kind: PlaceMutationKind::BindOrAssign,
            creation_policy: AssignmentCreationPolicy::CreateBinding,
            shape_policy: AssignmentShapePolicy::MatlabCompatible,
        };
        let value = ValueFlowFact::Single(TypeFact::Tensor(TensorTypeFact {
            element: TensorElementDomainFact::Numeric {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            },
            shape: ShapeFact::Shaped {
                dims: vec![DimFact::Known(1), DimFact::Symbolic(DimSymbol("n".into()))],
            },
        }));

        assert!(matches!(mutation.place, HirPlace::Binding(id) if id == binding));
        assert!(matches!(
            value,
            ValueFlowFact::Single(TypeFact::Tensor(TensorTypeFact {
                shape: ShapeFact::Shaped { .. },
                ..
            }))
        ));
    }

    #[test]
    fn def_path_is_distinct_from_local_table_ids() {
        let function_id = FunctionId(0);
        let path = DefPath {
            package: PackageName("pkg".into()),
            module: QualifiedName(vec![SymbolName("pkg".into()), SymbolName("demo".into())]),
            item: vec![DefPathSegment::Function(SymbolName("f".into()))],
        };

        assert_eq!(function_id, FunctionId(0));
        assert_eq!(path.package.0, "pkg");
        assert!(matches!(path.item[0], DefPathSegment::Function(_)));
    }

    #[test]
    fn async_facts_distinguish_lazy_futures_from_spawned_tasks() {
        let future = AsyncValueFact::Future(FutureFact {
            output: Box::new(TypeFact::Unknown),
            state: FutureStateFact::Lazy,
        });
        let task = AsyncValueFact::TaskHandle(TaskHandleFact {
            output: Box::new(TypeFact::Unknown),
            spawn_safety: SpawnSafetyFact::SpawnSafe,
        });

        assert!(matches!(
            future,
            AsyncValueFact::Future(FutureFact {
                state: FutureStateFact::Lazy,
                ..
            })
        ));
        assert!(matches!(
            task,
            AsyncValueFact::TaskHandle(TaskHandleFact {
                spawn_safety: SpawnSafetyFact::SpawnSafe,
                ..
            })
        ));
    }
}
