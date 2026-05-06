use crate::{
    BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, SourceId, Span, StmtId, Type,
    VarId,
};
use runmat_parser as parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct HirAssembly {
    pub modules: Vec<HirModule>,
    pub functions: Vec<HirFunction>,
    pub classes: Vec<HirClass>,
    pub bindings: Vec<HirBinding>,
    pub entrypoints: Vec<HirEntrypoint>,
}

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
    pub body: SemanticHirBlock,
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
pub struct SemanticHirBlock {
    pub statements: Vec<SemanticHirStmt>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SemanticHirStmt {
    pub id: StmtId,
    pub kind: HirStmtKind,
    pub span: Span,
}

impl SemanticHirStmt {
    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirStmtKind {
    ExprStmt(SemanticHirExpr, bool),
    Assign(SemanticHirPlace, SemanticHirExpr, bool),
    MultiAssign(OutputTargetList, SemanticHirExpr, bool),
    If {
        cond: SemanticHirExpr,
        then_body: SemanticHirBlock,
        elseif_blocks: Vec<(SemanticHirExpr, SemanticHirBlock)>,
        else_body: Option<SemanticHirBlock>,
    },
    While {
        cond: SemanticHirExpr,
        body: SemanticHirBlock,
    },
    For {
        binding: BindingId,
        range: SemanticHirExpr,
        body: SemanticHirBlock,
        semantics: LoopIterationSemantics,
    },
    Switch {
        expr: SemanticHirExpr,
        cases: Vec<(SemanticHirExpr, SemanticHirBlock)>,
        otherwise: Option<SemanticHirBlock>,
    },
    TryCatch {
        try_body: SemanticHirBlock,
        catch_binding: Option<BindingId>,
        catch_body: SemanticHirBlock,
    },
    Global(Vec<BindingId>),
    Persistent(Vec<BindingId>),
    Break,
    Continue,
    Return,
    Import(HirImport),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SemanticHirExpr {
    pub id: ExprId,
    pub kind: SemanticHirExprKind,
    pub span: Span,
}

impl SemanticHirExpr {
    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum SemanticHirExprKind {
    Number(String),
    String(StringLiteral),
    Constant(SymbolName),
    Binding(BindingId),
    Unary(OperatorKind, Box<SemanticHirExpr>),
    Binary(Box<SemanticHirExpr>, OperatorKind, Box<SemanticHirExpr>),
    Tensor(Vec<Vec<SemanticHirExpr>>),
    Cell(Vec<Vec<SemanticHirExpr>>),
    Range(
        Box<SemanticHirExpr>,
        Option<Box<SemanticHirExpr>>,
        Box<SemanticHirExpr>,
    ),
    Colon,
    End,
    Index(Box<SemanticHirExpr>, IndexingSemantics),
    Member(Box<SemanticHirExpr>, MemberName),
    MemberDynamic(Box<SemanticHirExpr>, Box<SemanticHirExpr>),
    Call(SemanticHirCall),
    CommandCall(SemanticHirCommandCall),
    FunctionHandle(FunctionHandleTarget),
    AnonymousFunction(FunctionId),
    MetaClass(QualifiedName),
    Await(Box<SemanticHirExpr>),
    Spawn(Box<SemanticHirExpr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum SemanticHirPlace {
    Binding(BindingId),
    Member(Box<SemanticHirExpr>, MemberName),
    MemberDynamic(Box<SemanticHirExpr>, Box<SemanticHirExpr>),
    Index(Box<SemanticHirExpr>, IndexingSemantics),
    IndexCell(Box<SemanticHirExpr>, IndexingSemantics),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SemanticHirCall {
    pub callee: HirCallableRef,
    pub args: Vec<SemanticHirExpr>,
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirCallableRef {
    Function(FunctionId),
    ClassConstructor(ClassId),
    Builtin(BuiltinId),
    Imported(DefPath),
    DynamicExpr(Box<SemanticHirExpr>),
    Unresolved(QualifiedName),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum CallSyntax {
    Plain,
    Method,
    DottedInvoke,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct SemanticHirCommandCall {
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
    pub default: Option<SemanticHirExpr>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassMethod {
    pub function: FunctionId,
    pub name: MethodName,
    pub is_static: bool,
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
    pub is_constant: bool,
    pub is_dependent: bool,
    pub is_transient: bool,
    pub is_hidden: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct OutputTargetList {
    pub targets: Vec<OutputTarget>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum OutputTarget {
    Place(SemanticHirPlace),
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
    Expr(SemanticHirExpr),
    Logical(SemanticHirExpr),
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

// Compatibility aliases for the pre-Plan-0 lowering/inference API. The semantic
// model above is the new target surface; these aliases keep existing tests and
// consumers compiling until Plans 1-3 migrate them to semantic HIR/MIR products.
pub type HirExpr = LegacyHirExpr;
pub type HirExprKind = LegacyHirExprKind;
pub type HirStmt = LegacyHirStmt;
pub type HirClassMember = LegacyHirClassMember;
pub type HirLValue = LegacyHirLValue;
pub type HirProgram = LegacyHirProgram;

#[derive(Debug, Clone)]
pub struct LoweringResult {
    pub hir: LegacyHirProgram,
    pub variables: HashMap<String, usize>,
    pub functions: HashMap<String, LegacyHirStmt>,
    pub var_types: Vec<Type>,
    pub var_names: HashMap<VarId, String>,
    pub inferred_globals: HashMap<VarId, Type>,
    pub inferred_function_envs: HashMap<String, HashMap<VarId, Type>>,
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
                body: SemanticHirBlock { statements: vec![] },
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
            body: SemanticHirBlock { statements: vec![] },
            span: span(),
        };

        assert_eq!(child_function.parent, Some(parent));
        assert_eq!(child_function.captures[0].binding, binding);
        assert_eq!(child_function.captures[0].from_function, parent);
    }
}
