use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};
use uuid::Uuid;

#[derive(Subcommand, Clone)]
pub enum ClusterCommand {
    /// List clusters in an organization
    List {
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
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
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
        org: Option<Uuid>,
        #[arg(long)]
        name: String,
        #[arg(long, value_parser = runmat_server_client::auth::parse_project_id_input)]
        project: Option<Uuid>,
        #[arg(long = "queue", default_value = "default")]
        queues: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Change a cluster's scheduling state
    State {
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
        org: Option<Uuid>,
        cluster: String,
        state: ClusterStateArg,
        #[arg(long)]
        json: bool,
    },
    /// Create a single-use node enrollment token
    Enroll {
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
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
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
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
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
        org: Option<Uuid>,
        cluster: String,
        node: String,
        state: NodeStateArg,
        #[arg(long)]
        json: bool,
    },
    /// Register this machine using its signed-in account and start a cluster node
    Join(ClusterJoinArgs),
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
            Self::Join(_) => false,
        }
    }
}

#[derive(Args, Clone)]
pub struct NodeArgs {
    #[command(flatten)]
    pub options: NodeOptions,
    #[command(subcommand)]
    pub command: NodeCommand,
}

#[derive(Args, Clone)]
pub struct ClusterJoinArgs {
    /// Cluster name in the selected organization
    pub cluster: String,
    #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
    pub org: Option<Uuid>,
    /// Install and start an operating-system service
    #[arg(long, conflicts_with = "foreground")]
    pub service: bool,
    /// Run in this terminal (the default)
    #[arg(long)]
    pub foreground: bool,
    #[command(flatten)]
    pub options: NodeOptions,
}

#[derive(Args, Clone)]
pub struct NodeOptions {
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
}

#[derive(Subcommand, Clone)]
pub enum NodeCommand {
    /// Show the registered machine identity without revealing its credentials
    Status,
    /// Enroll this machine with a single-use token
    Enroll {
        #[arg(
            long,
            env = "RUNMAT_NODE_ENROLLMENT_TOKEN",
            required_unless_present = "token_stdin",
            conflicts_with = "token_stdin"
        )]
        token: Option<String>,
        /// Read the enrollment token from stdin (never from process arguments)
        #[arg(long)]
        token_stdin: bool,
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

impl NodeCommand {
    pub(crate) fn machine_output(&self) -> bool {
        matches!(
            self,
            Self::Status
                | Self::Inventory
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
