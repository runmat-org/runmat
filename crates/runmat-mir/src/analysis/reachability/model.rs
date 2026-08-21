use std::collections::BTreeMap;

use runmat_types::{BindingId, ProgramSourceId};
use serde::{Deserialize, Serialize};

pub const REACHABILITY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReachabilityCertainty {
    Definite,
    FiniteDynamic,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReachabilityNodeKind {
    Function,
    Builtin,
    Class,
    Method,
    Provider,
    Extension,
    ArtifactDependency,
    GlobalState,
    PersistentState,
    RuntimeCatalog,
    ExternalCallable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReachabilityReason {
    Entrypoint,
    PublicCallable,
    DirectCall,
    OperatorDispatch,
    FutureCall,
    FunctionHandle,
    DynamicNamedCall,
    DynamicCall,
    MethodDispatch,
    SuperDispatch,
    ClassReference,
    BuiltinLinkContract,
    AcceleratorPlacement,
    ExtensionCapability,
    WorkspaceGlobal,
    WorkspacePersistent,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReachabilityNode {
    pub id: String,
    pub kind: ReachabilityNodeKind,
    pub module: String,
    pub symbol: String,
    pub certainty: ReachabilityCertainty,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReachabilityEdge {
    pub from: Option<String>,
    pub to: String,
    pub certainty: ReachabilityCertainty,
    pub reason: ReachabilityReason,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReachabilityReport {
    pub schema_version: u16,
    pub nodes: Vec<ReachabilityNode>,
    pub edges: Vec<ReachabilityEdge>,
    pub has_unknown_edges: bool,
}

impl ReachabilityReport {
    pub fn node(&self, id: &str) -> Option<&ReachabilityNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    pub fn retained_function_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes.iter().filter_map(|node| {
            (node.kind == ReachabilityNodeKind::Function)
                .then(|| node.id.strip_prefix("function:"))
                .flatten()
                .and_then(|id| id.parse().ok())
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReachabilityNames {
    pub sources: BTreeMap<ProgramSourceId, String>,
    pub bindings: BTreeMap<BindingId, String>,
}
