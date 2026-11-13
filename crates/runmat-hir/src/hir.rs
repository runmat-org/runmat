use crate::{BindingId, ClassId, EntrypointId, ExprId, FunctionId, ModuleId, Span, StmtId};
pub use runmat_types::{
    AssignmentCreationPolicy, AssignmentShapePolicy, BindingName, BuiltinId, CallSyntax,
    CallableFallbackPolicy, CallableIdentity, CapturedBinding, ClassDeclaration, ClassIdentity,
    DefPath, DefPathSegment, EntrypointName, EnvironmentEffect, FunctionAbi,
    FunctionArgumentValidation, FunctionKind, FunctionModifiers, FunctionName, IndexKind,
    IndexResultContext, MemberName, MethodId, MethodName, OperatorKind, PackageName,
    PlaceMutationKind, QualifiedName, RequestedOutputCount, SemanticAttribute, StringLiteral,
    SymbolName, WorkspaceEffect, ASSIGNIN_BUILTIN_NAME, AWAIT_EXTENSION_NAME, DISCARD_OUTPUT_NAME,
    EVALC_BUILTIN_NAME, EVALIN_BUILTIN_NAME, EVAL_BUILTIN_NAME, FEVAL_BUILTIN_NAME,
    NARGINCHK_BUILTIN_NAME, NARGIN_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME, NARGOUT_BUILTIN_NAME,
    RUNTESTS_BUILTIN_NAME, RUN_BUILTIN_NAME, SPAWN_EXTENSION_NAME,
    TEST_CLASS_REGISTRATION_BUILTIN_NAME,
};
use runmat_types::{IntegerLiteral, ProgramSourceId};
use serde::{Deserialize, Serialize};

/// Canonical semantic HIR product for one compiled source set.
///
/// The assembly owns the tables for modules, functions, classes, bindings, and
/// entrypoints. Table IDs are local to this assembly; stable identities for
/// packages/modules/items are represented separately by `DefPath` and qualified
/// name types.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct HirAssembly {
    /// Original declarations for retained functions outside this unit's ID namespace.
    #[serde(default)]
    pub bound_function_modifiers: std::collections::BTreeMap<FunctionId, FunctionModifiers>,
    /// Exact current-function binding selected for implicit statement results.
    /// Reusing an incoming or explicit binding preserves its original role and
    /// storage; consumers must not rediscover this authority from its name.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub statement_result_bindings: std::collections::BTreeMap<FunctionId, BindingId>,
    /// Compiler-admitted program root, including function-only and empty
    /// sources with no synthetic executable entrypoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_source: Option<ProgramSourceId>,
    pub modules: Vec<HirModule>,
    pub functions: Vec<HirFunction>,
    pub classes: Vec<HirClass>,
    pub bindings: Vec<HirBinding>,
    pub entrypoints: Vec<HirEntrypoint>,
}

impl HirAssembly {
    /// Classify from the typed source unit and function graph, not a display
    /// name or the spelling of a source path.
    pub fn function_source_role(&self, function: &HirFunction) -> runmat_types::FunctionSourceRole {
        use runmat_types::FunctionSourceRole as Role;
        match &function.kind {
            FunctionKind::Anonymous => Role::Anonymous,
            FunctionKind::SyntheticEntrypoint => Role::Synthetic,
            FunctionKind::ClassMethod { .. } => Role::Method,
            FunctionKind::Named if function.parent.is_some() => Role::Nested,
            FunctionKind::Named => {
                let Some(module) = self.modules.get(function.module.0) else {
                    return Role::Unknown;
                };
                if matches!(module.source_unit, SourceUnitKind::FunctionFile)
                    && module.top_level_functions.first() == Some(&function.id)
                {
                    Role::Primary
                } else if module.top_level_functions.contains(&function.id) {
                    Role::Local
                } else {
                    Role::Unknown
                }
            }
        }
    }
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
    pub source_id: ProgramSourceId,
    pub source_unit: SourceUnitKind,
    pub imports: Vec<HirImport>,
    pub top_level_functions: Vec<FunctionId>,
    pub classes: Vec<ClassId>,
    pub synthetic_entry_function: Option<FunctionId>,
    pub script_sections: Vec<HirScriptSection>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirScriptSection {
    pub ordinal: u32,
    pub title: String,
    pub marker_span: Span,
    pub body_span: Span,
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
    pub argument_validations: Vec<FunctionArgumentValidation>,
    pub locals: Vec<BindingId>,
    pub captures: Vec<CapturedBinding>,
    pub modifiers: FunctionModifiers,
    pub body: HirBlock,
    pub span: Span,
}

/// Semantic binding identity for a name owned by a module or function.
///
/// Bindings model MATLAB lexical/global/persistent storage and workspace
/// visibility. They are not VM slot numbers.
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

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum BindingOwner {
    Module(ModuleId),
    Function(FunctionId),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum BindingRole {
    Parameter,
    Output,
    Local,
    /// A top-level binding with a value supplied by the admitted incoming
    /// workspace snapshot, not a newly declared or unresolved local.
    IncomingWorkspace,
    ExternalWorkspace,
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
    SequenceAssign {
        target: HirSequenceTarget,
        value: HirExpr,
        suppressed: bool,
    },
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
    },
    ParFor {
        region: runmat_types::ParallelRegionId,
        binding: BindingId,
        range: HirExpr,
        maximum_workers: Option<HirExpr>,
        body: HirBlock,
    },
    Spmd {
        region: runmat_types::ParallelRegionId,
        header: crate::parallel::SpmdHeader<HirExpr>,
        body: HirBlock,
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

    /// An argument already marked by HIR's comma-separated-list policy.
    /// Merely returning several values does not make an ordinary expression
    /// an expansion; its actual argument sequence context must request it.
    pub fn expands_call_argument_sequence(&self) -> bool {
        match &self.kind {
            HirExprKind::Index(_, indexing) => {
                indexing.kind == IndexKind::Brace
                    && indexing.result_context == IndexResultContext::FunctionArgumentExpansion
            }
            HirExprKind::Member { sequence_use, .. }
            | HirExprKind::MemberDynamic { sequence_use, .. } => {
                *sequence_use == runmat_types::SequenceUse::ExpandAll
            }
            HirExprKind::SubscriptChain(chain) => {
                chain.sequence_use == runmat_types::SequenceUse::ExpandAll
                    && chain.terminal_expands_sequence()
            }
            _ => false,
        }
    }

    /// Whether an unsuppressed expression has a value for statement echo.
    /// Builtin output policy belongs to the semantic call, not a CLI renderer.
    pub fn emits_statement_output(&self) -> bool {
        match &self.kind {
            HirExprKind::Call(call) => {
                call.requested_outputs.known_count() != Some(0)
                    && !matches!(&call.callee, HirCallableRef::Builtin(builtin)
                        if runmat_builtins::suppresses_auto_output(builtin.0.as_str()))
            }
            HirExprKind::CommandCall(_) => false,
            _ => true,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirExprKind {
    Number(String),
    ImaginaryNumber(String),
    IntegerLiteral(IntegerLiteral),
    String(StringLiteral),
    Constant(SymbolName),
    Binding(BindingId),
    Unary(OperatorKind, Box<HirExpr>),
    Binary(Box<HirExpr>, OperatorKind, Box<HirExpr>),
    Tensor(Vec<Vec<HirExpr>>),
    Cell(Vec<Vec<HirExpr>>),
    StructLiteral(Vec<(MemberName, HirExpr)>),
    ObjectLiteral {
        class_name: QualifiedName,
        fields: Vec<(MemberName, HirExpr)>,
    },
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
    End,
    Index(Box<HirExpr>, IndexingSemantics),
    /// One syntactic indexing chain retained as a unit so object dispatch can
    /// give the authoritative receiver the complete ordered subscript path.
    SubscriptChain(HirSubscriptChain),
    Member {
        base: Box<HirExpr>,
        member: MemberName,
        sequence_use: runmat_types::SequenceUse,
    },
    MemberDynamic {
        base: Box<HirExpr>,
        member: Box<HirExpr>,
        sequence_use: runmat_types::SequenceUse,
    },
    WorkspaceFirstStaticProperty {
        workspace_name: SymbolName,
        class_name: ClassIdentity,
        property: MemberName,
    },
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

/// A bracketed destination whose cardinality is determined by its evaluated
/// aggregate rather than by the number of syntactic targets.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirSequenceTarget {
    Member {
        base: Box<HirExpr>,
        member: MemberName,
    },
    DynamicMember {
        base: Box<HirExpr>,
        member: Box<HirExpr>,
    },
    CellContents {
        base: Box<HirExpr>,
        indexing: IndexingSemantics,
    },
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
    #[serde(default)]
    pub workspace_first_name: Option<SymbolName>,
    #[serde(default)]
    pub bare_identifier: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirCallableRef {
    Function(FunctionId),
    ExternalFunction {
        function: FunctionId,
        display_name: String,
    },
    Builtin(BuiltinId),
    Imported(DefPath),
    SuperConstructor {
        current_class: ClassIdentity,
        super_class: QualifiedName,
    },
    SuperMethod {
        current_class: ClassIdentity,
        super_class: QualifiedName,
        method: SymbolName,
    },
    DynamicExpr(Box<HirExpr>),
    Unresolved(QualifiedName),
}

impl HirCallableRef {
    pub fn is_feval_builtin_like(&self) -> bool {
        match self {
            HirCallableRef::Builtin(id) => id.0 == FEVAL_BUILTIN_NAME,
            HirCallableRef::Unresolved(name) if name.0.len() == 1 => {
                name.0[0].0 == FEVAL_BUILTIN_NAME
            }
            _ => false,
        }
    }

    pub fn identity(&self) -> Option<CallableIdentity> {
        match self {
            HirCallableRef::Function(function) => Some(CallableIdentity::BoundFunction(*function)),
            HirCallableRef::ExternalFunction {
                function,
                display_name,
            } => Some(CallableIdentity::ExternalFunction {
                function: *function,
                display_name: display_name.clone(),
            }),
            HirCallableRef::Builtin(builtin) => Some(CallableIdentity::Builtin(builtin.clone())),
            HirCallableRef::Imported(path) => Some(CallableIdentity::Imported(path.clone())),
            HirCallableRef::SuperConstructor { .. } | HirCallableRef::SuperMethod { .. } => None,
            HirCallableRef::DynamicExpr(_) => None,
            HirCallableRef::Unresolved(name) => {
                if name.0.len() == 1 {
                    Some(CallableIdentity::DynamicName(name.0[0].clone()))
                } else {
                    Some(CallableIdentity::ExternalName(name.clone()))
                }
            }
        }
    }
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
    pub module: ModuleId,
    pub declaration: ClassDeclaration,
    pub property_defaults: Vec<ClassPropertyDefault>,
    pub arguments: Vec<ClassArgumentBlock>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassPropertyDefault {
    pub property: MemberName,
    pub value: HirExpr,
}

impl HirClass {
    pub fn property_default(&self, property: &MemberName) -> Option<&HirExpr> {
        self.property_defaults
            .iter()
            .find(|default| default.property == *property)
            .map(|default| &default.value)
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct ClassArgumentBlock {
    pub declared_attributes: Vec<SemanticAttribute>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct OutputTargetList {
    pub requested_outputs: RequestedOutputCount,
    pub targets: Vec<OutputTarget>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum OutputTarget {
    Place(HirPlace),
    Sequence(HirSequenceTarget),
    Discard,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum IndexComponent {
    Colon,
    End { dim: Option<usize>, offset: isize },
    Expr(HirExpr),
    Logical(HirExpr),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirSubscriptChain {
    pub root: Box<HirExpr>,
    pub steps: Vec<HirSubscriptStep>,
    pub sequence_use: runmat_types::SequenceUse,
    pub context: runmat_types::ObjectIndexingContext,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirSubscriptStep {
    Index(IndexingSemantics),
    Member(MemberName),
    DynamicMember(HirExpr),
    DottedInvoke {
        member: MemberName,
        arguments: Vec<HirDottedInvokeArgument>,
        /// A lexically resolved function is eligible only after receiver
        /// method/member dispatch. Never re-resolve its name at execution.
        semantic_fallback: Option<runmat_types::CallableIdentity>,
    },
}

/// A dotted invocation has call arguments, not ordinary index selectors.
/// Receiver-relative `end` and `:` retain their source-level forms here;
/// a Value may carry a comma-separated-list expansion request.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirDottedInvokeArgument {
    Colon,
    End { offset: isize },
    Value(HirExpr),
}

impl HirSubscriptChain {
    pub fn terminal_expands_sequence(&self) -> bool {
        matches!(
            self.steps.last(),
            Some(
                HirSubscriptStep::Member(_)
                    | HirSubscriptStep::DynamicMember(_)
                    | HirSubscriptStep::Index(IndexingSemantics {
                        kind: IndexKind::Brace,
                        ..
                    })
            )
        )
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct IndexingSemantics {
    pub kind: IndexKind,
    pub components: Vec<IndexComponent>,
    pub result_context: IndexResultContext,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionHandleTarget {
    Function(FunctionId),
    ExternalFunction {
        function: FunctionId,
        display_name: String,
    },
    Builtin(BuiltinId),
    Anonymous(FunctionId),
    DefPath(DefPath),
    DynamicName(SymbolName),
}

impl FunctionHandleTarget {
    pub fn identity(&self) -> CallableIdentity {
        match self {
            FunctionHandleTarget::Function(function) => CallableIdentity::BoundFunction(*function),
            FunctionHandleTarget::ExternalFunction {
                function,
                display_name,
            } => CallableIdentity::ExternalFunction {
                function: *function,
                display_name: display_name.clone(),
            },
            FunctionHandleTarget::Builtin(builtin) => CallableIdentity::Builtin(builtin.clone()),
            FunctionHandleTarget::Anonymous(function) => {
                CallableIdentity::AnonymousFunction(*function)
            }
            FunctionHandleTarget::DefPath(path) => CallableIdentity::Imported(path.clone()),
            FunctionHandleTarget::DynamicName(name) => CallableIdentity::DynamicName(name.clone()),
        }
    }
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
pub struct PlaceMutation {
    pub place: HirPlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
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
pub struct HirIndex {
    pub bindings: Vec<BindingResolution>,
    pub functions: Vec<FunctionResolution>,
    pub classes: Vec<ClassResolution>,
    pub imports: Vec<ImportResolution>,
    pub references: Vec<ReferenceResolution>,
    pub calls: Vec<CallResolution>,
    pub function_handles: Vec<FunctionHandleResolution>,
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
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
    pub span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ProgramSourceId>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct FunctionHandleResolution {
    pub name: QualifiedName,
    pub target: FunctionHandleTarget,
    pub span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ProgramSourceId>,
}

#[derive(Debug, Clone)]
pub struct LoweringResult {
    pub assembly: HirAssembly,
    pub hir_index: HirIndex,
}

#[cfg(test)]
mod tests;
