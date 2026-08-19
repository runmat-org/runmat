use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};
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
    /// Enroll this machine or run it as a cluster execution node
    Join(NodeJoinArgs),
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
            Self::Join(args) => args.command.machine_output(),
        }
    }
}

#[derive(Args, Clone)]
pub struct NodeJoinArgs {
    /// Load the canonical execution-node JSON configuration
    #[arg(long, env = "RUNMAT_NODE_AGENT_CONFIG")]
    pub node_config: Option<PathBuf>,
    /// RunMat Server API URL
    #[arg(long, env = "RUNMAT_SERVER_URL")]
    pub server: Option<String>,
    /// Private node state directory
    #[arg(long)]
    pub state_directory: Option<PathBuf>,
    /// RunMat executable used to launch allocations (defaults to this executable)
    #[arg(long, env = "RUNMAT_EXECUTABLE")]
    pub runmat: Option<PathBuf>,
    /// Execution trust tier advertised by this node
    #[arg(long, value_enum)]
    pub trust_tier: Option<NodeTrustTier>,
    #[command(subcommand)]
    pub command: NodeJoinCommand,
}

#[derive(Subcommand, Clone)]
pub enum NodeJoinCommand {
    /// Enroll this machine with a single-use token
    Enroll {
        #[arg(long, env = "RUNMAT_NODE_ENROLLMENT_TOKEN")]
        token: String,
    },
    /// Run the execution-node service in the foreground
    Run,
    /// Rotate the enrolled node credential
    RotateCredential,
    /// Print this machine's execution inventory as JSON
    Inventory,
    /// Install, inspect, or remove the operating-system service
    Service {
        #[command(subcommand)]
        command: NodeServiceCommand,
    },
    /// Internal Windows Service Control Manager entry point
    #[command(hide = true)]
    WindowsServiceRun,
}

impl NodeJoinCommand {
    fn machine_output(&self) -> bool {
        matches!(
            self,
            Self::Inventory
                | Self::Service {
                    command: NodeServiceCommand::Print
                        | NodeServiceCommand::Install { dry_run: true }
                        | NodeServiceCommand::Uninstall { dry_run: true }
                }
        )
    }
}

#[derive(Subcommand, Clone)]
pub enum NodeServiceCommand {
    /// Persist the validated configuration and start the service at boot
    Install {
        /// Print the exact installation plan without changing the host
        #[arg(long)]
        dry_run: bool,
    },
    /// Stop and remove the operating-system service
    Uninstall {
        /// Print the exact removal plan without changing the host
        #[arg(long)]
        dry_run: bool,
    },
    /// Print the exact service files and commands for this host
    Print,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum NodeTrustTier {
    CustomerTrusted,
    HostedOrdinary,
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
