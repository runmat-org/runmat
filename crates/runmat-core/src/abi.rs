use runmat_builtins::Value;
use runmat_hir::{BindingName, DefPath, EntrypointId, Span};
use uuid::Uuid;

use crate::execution::ExecutionProfiling;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SourceInput {
    Path(String),
    Text { name: String, text: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntrypointSelector {
    SourcePath(String),
    Named(String),
    Id(EntrypointId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostExecutionPolicy {
    pub materialization: MaterializationPolicy,
    pub top_level_await: bool,
}

impl Default for HostExecutionPolicy {
    fn default() -> Self {
        Self {
            materialization: MaterializationPolicy::MetadataOnly,
            top_level_await: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializationPolicy {
    MetadataOnly,
    Preview { limit: usize },
    HostValue,
    PreserveProvider,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkspaceHandle(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolverHandle(pub Uuid);

#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub source: SourceInput,
    pub entrypoint: EntrypointSelector,
    pub compatibility: runmat_hir::CompatibilityMode,
    pub host_policy: HostExecutionPolicy,
    pub inputs: RuntimeFlow,
    pub requested_outputs: runmat_hir::RequestedOutputCount,
    pub workspace: WorkspaceHandle,
    pub resolver: ResolverHandle,
}

#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    pub flow: RuntimeFlow,
    pub workspace_delta: WorkspaceDelta,
    pub display_events: Vec<DisplayEvent>,
    pub diagnostics: Vec<RuntimeDiagnostic>,
    pub effects: Vec<ObservedEffect>,
    pub suspension: Option<Suspension>,
    pub profiling: Option<ExecutionProfiling>,
}

impl Default for ExecutionOutcome {
    fn default() -> Self {
        Self {
            flow: RuntimeFlow::NoValue,
            workspace_delta: WorkspaceDelta::default(),
            display_events: Vec::new(),
            diagnostics: Vec::new(),
            effects: Vec::new(),
            suspension: None,
            profiling: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeFlow {
    NoValue,
    Single(Value),
    OutputList(Vec<Value>),
    CommaList(Vec<Value>),
    DynamicList(DynamicListHandle),
}

impl RuntimeFlow {
    pub fn is_no_value(&self) -> bool {
        matches!(self, Self::NoValue)
    }

    pub fn durable_workspace_value(&self) -> Option<&Value> {
        match self {
            Self::Single(value) => Some(value),
            Self::NoValue | Self::OutputList(_) | Self::CommaList(_) | Self::DynamicList(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicListHandle(pub Uuid);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SourceIdentity {
    Interactive { session: Uuid },
    PathAndContentHash { path: String, hash: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkspaceBindingKey {
    Interactive {
        session: Uuid,
        name: BindingName,
    },
    SourceBinding {
        source: SourceIdentity,
        def_path: DefPath,
        binding: BindingName,
    },
    Global {
        scope: GlobalScopeKey,
        name: BindingName,
    },
    Persistent {
        function: DefPath,
        name: BindingName,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GlobalScopeKey {
    Session(Uuid),
    Package(String),
}

#[derive(Debug, Clone)]
pub struct WorkspaceBindingValue {
    pub key: WorkspaceBindingKey,
    pub value: Value,
}

#[derive(Debug, Clone, Default)]
pub struct WorkspaceDelta {
    pub upserts: Vec<WorkspaceBindingValue>,
    pub removals: Vec<WorkspaceBindingKey>,
    pub full_snapshot_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}

#[derive(Debug, Clone)]
pub struct DisplayEvent {
    pub label: DisplayLabel,
    pub value: Value,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayLabel {
    Binding(BindingName),
    Literal(String),
    Anonymous,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDiagnostic {
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservedEffect {
    Workspace(WorkspaceEffectKind),
    Environment(EnvironmentEffectKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceEffectKind {
    Load,
    Clear,
    AssignIn,
    EvalIn,
    DeclareGlobal,
    DeclarePersistent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentEffectKind {
    ChangeDirectory,
    MutatePath,
    ClearFunctionCache,
    ClearClassCache,
    InvalidateDynamicLookup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suspension {
    pub task: Uuid,
    pub frame: Uuid,
    pub resume_point: ResumePoint,
    pub pending: PendingOperation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumePoint {
    BytecodePc(usize),
    Host(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingOperation {
    HostInteraction,
    Provider,
    Filesystem,
    Timer,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_flow_distinguishes_durable_values_from_lists() {
        let value = Value::Num(1.0);
        assert!(RuntimeFlow::NoValue.is_no_value());
        assert!(RuntimeFlow::Single(value)
            .durable_workspace_value()
            .is_some());
        assert!(RuntimeFlow::OutputList(vec![Value::Num(1.0)])
            .durable_workspace_value()
            .is_none());
        assert!(RuntimeFlow::CommaList(vec![Value::Num(1.0)])
            .durable_workspace_value()
            .is_none());
    }

    #[test]
    fn execution_outcome_defaults_to_empty_no_value_contract() {
        let outcome = ExecutionOutcome::default();
        assert!(outcome.flow.is_no_value());
        assert!(outcome.workspace_delta.upserts.is_empty());
        assert!(outcome.workspace_delta.removals.is_empty());
        assert!(!outcome.workspace_delta.full_snapshot_required);
        assert!(outcome.display_events.is_empty());
        assert!(outcome.diagnostics.is_empty());
        assert!(outcome.effects.is_empty());
        assert!(outcome.suspension.is_none());
    }

    #[test]
    fn workspace_keys_are_stable_boundary_identities() {
        let session = Uuid::from_u128(1);
        let key = WorkspaceBindingKey::Interactive {
            session,
            name: BindingName("adjusted".to_string()),
        };
        assert_eq!(
            key,
            WorkspaceBindingKey::Interactive {
                session,
                name: BindingName("adjusted".to_string())
            }
        );
    }
}
