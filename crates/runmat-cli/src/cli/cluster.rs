use clap::{Subcommand, ValueEnum};
use uuid::Uuid;

#[derive(Subcommand, Clone)]
pub enum ClusterCommand {
    /// List clusters in an organization
    List {
        #[arg(long)]
        org: Option<Uuid>,
        #[arg(long)]
        limit: Option<u32>,
        #[arg(long)]
        cursor: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Create a cluster
    Create {
        #[arg(long)]
        org: Option<Uuid>,
        #[arg(long)]
        name: String,
        #[arg(long)]
        project: Option<String>,
        #[arg(long = "queue", default_value = "default")]
        queues: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Change a cluster's scheduling state
    State {
        #[arg(long)]
        org: Option<Uuid>,
        cluster: String,
        state: ClusterStateArg,
        #[arg(long)]
        json: bool,
    },
    /// Create a single-use node enrollment token
    Enroll {
        #[arg(long)]
        org: Option<Uuid>,
        cluster: String,
        #[arg(long, default_value = "900")]
        ttl_seconds: i64,
        #[arg(long)]
        identity_fingerprint: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// List nodes enrolled in a cluster
    Nodes {
        #[arg(long)]
        org: Option<Uuid>,
        cluster: String,
        #[arg(long)]
        limit: Option<u32>,
        #[arg(long)]
        cursor: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Change an enrolled node's lifecycle state
    NodeState {
        #[arg(long)]
        org: Option<Uuid>,
        cluster: String,
        node: String,
        state: NodeStateArg,
        #[arg(long)]
        json: bool,
    },
}

impl ClusterCommand {
    pub(crate) fn machine_output(&self) -> bool {
        match self {
            Self::List { json, .. }
            | Self::Create { json, .. }
            | Self::State { json, .. }
            | Self::Enroll { json, .. }
            | Self::Nodes { json, .. }
            | Self::NodeState { json, .. } => *json,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ClusterStateArg {
    Active,
    Draining,
    Disabled,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum NodeStateArg {
    Active,
    Draining,
    Offline,
    Revoked,
}
