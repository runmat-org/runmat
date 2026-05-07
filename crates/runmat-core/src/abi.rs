use runmat_builtins::Value;
use runmat_hir::{BindingName, DefPath, EntrypointId, Span};
use uuid::Uuid;

use crate::execution::{ExecutionProfiling, ExecutionResult, ExecutionStreamEntry, StdinEvent};
use crate::fusion::FusionPlanSnapshot;

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
    pub streams: Vec<ExecutionStreamEntry>,
    pub diagnostics: Vec<RuntimeDiagnostic>,
    pub effects: Vec<ObservedEffect>,
    pub suspension: Option<Suspension>,
    pub profiling: Option<ExecutionProfiling>,
    pub execution_time_ms: u64,
    pub used_jit: bool,
    pub type_info: Option<String>,
    pub figures_touched: Vec<u32>,
    pub stdin_events: Vec<StdinEvent>,
    pub fusion_plan: Option<FusionPlanSnapshot>,
}

impl Default for ExecutionOutcome {
    fn default() -> Self {
        Self {
            flow: RuntimeFlow::NoValue,
            workspace_delta: WorkspaceDelta::default(),
            display_events: Vec::new(),
            streams: Vec::new(),
            diagnostics: Vec::new(),
            effects: Vec::new(),
            suspension: None,
            profiling: None,
            execution_time_ms: 0,
            used_jit: false,
            type_info: None,
            figures_touched: Vec::new(),
            stdin_events: Vec::new(),
            fusion_plan: None,
        }
    }
}

impl From<ExecutionResult> for ExecutionOutcome {
    fn from(result: ExecutionResult) -> Self {
        let mut diagnostics = Vec::new();
        if let Some(error) = result.error {
            diagnostics.push(RuntimeDiagnostic {
                code: error
                    .identifier()
                    .unwrap_or("RunMat:RuntimeError")
                    .to_string(),
                severity: DiagnosticSeverity::Error,
                message: error.message().to_string(),
                span: None,
            });
        }
        diagnostics.extend(
            result
                .warnings
                .into_iter()
                .map(|warning| RuntimeDiagnostic {
                    code: warning.identifier,
                    severity: DiagnosticSeverity::Warning,
                    message: warning.message,
                    span: None,
                }),
        );

        let display_events = result
            .value
            .as_ref()
            .map(|value| DisplayEvent {
                label: DisplayLabel::Anonymous,
                value: value.clone(),
                span: Span::default(),
            })
            .into_iter()
            .collect();

        Self {
            flow: result
                .value
                .map(RuntimeFlow::Single)
                .unwrap_or(RuntimeFlow::NoValue),
            workspace_delta: WorkspaceDelta {
                full_snapshot_required: result.workspace.full,
                ..WorkspaceDelta::default()
            },
            display_events,
            streams: result.streams,
            diagnostics,
            effects: Vec::new(),
            suspension: None,
            execution_time_ms: result.execution_time_ms,
            used_jit: result.used_jit,
            type_info: result.type_info,
            figures_touched: result.figures_touched,
            stdin_events: result.stdin_events,
            fusion_plan: result.fusion_plan,
            profiling: result.profiling,
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
        assert!(outcome.streams.is_empty());
        assert!(outcome.diagnostics.is_empty());
        assert!(outcome.effects.is_empty());
        assert!(outcome.suspension.is_none());
        assert_eq!(outcome.execution_time_ms, 0);
        assert!(!outcome.used_jit);
        assert!(outcome.type_info.is_none());
        assert!(outcome.figures_touched.is_empty());
        assert!(outcome.stdin_events.is_empty());
        assert!(outcome.fusion_plan.is_none());
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
