use std::path::PathBuf;

use clap::Subcommand;
use uuid::Uuid;

#[derive(Subcommand, Clone)]
pub enum JobCommand {
    /// Submit an exact encrypted program as a durable remote job
    Submit {
        file: PathBuf,
        #[arg(long)]
        project: Option<Uuid>,
        #[arg(long)]
        cluster: String,
        #[arg(long, default_value = "default")]
        queue: String,
        /// Pinned endpoint identity fingerprint trusted for this run
        #[arg(long)]
        trust_identity: String,
        #[arg(long)]
        function: Option<String>,
        #[arg(long)]
        idempotency_key: Option<String>,
        /// Number of remote worker allocations for driver-owned scheduling
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=1024))]
        workers: u32,
        #[arg(long)]
        detach: bool,
        #[arg(long)]
        json: bool,
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// List durable remote jobs
    List {
        #[arg(long)]
        project: Option<Uuid>,
        #[arg(long)]
        limit: Option<u32>,
        #[arg(long)]
        cursor: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Show one durable remote job
    Show {
        run_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Attach to a durable remote job and decrypt its terminal result
    Attach {
        run_id: String,
        #[arg(long)]
        no_follow: bool,
        #[arg(long)]
        json: bool,
    },
    /// Request cancellation of a durable remote job
    Cancel {
        run_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Manage organization-held recovery keys and recover encrypted job output
    Recovery {
        #[command(subcommand)]
        command: JobRecoveryCommand,
    },
}

impl JobCommand {
    pub(crate) fn machine_output(&self) -> bool {
        match self {
            Self::Submit { json, .. }
            | Self::List { json, .. }
            | Self::Show { json, .. }
            | Self::Attach { json, .. }
            | Self::Cancel { json, .. } => *json,
            Self::Recovery { command } => command.machine_output(),
        }
    }
}

#[derive(Subcommand, Clone)]
pub enum JobRecoveryCommand {
    /// Generate a new local recovery key file without sending its secret to RunMat
    Keygen {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 365, value_parser = clap::value_parser!(u32).range(1..=3650))]
        valid_days: u32,
        #[arg(long)]
        custodian_uri: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Configure an organization to require an envelope for a recovery key
    Configure {
        #[arg(long)]
        org: String,
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        max_active_runs: Option<u32>,
        #[arg(long)]
        max_active_runs_per_project: Option<u32>,
        #[arg(long)]
        max_active_runs_per_principal: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Disable the recovery-recipient requirement without deleting local key material
    Disable {
        #[arg(long)]
        org: String,
        #[arg(long)]
        json: bool,
    },
    /// Show the current organization execution and recovery policy
    Show {
        #[arg(long)]
        org: String,
        #[arg(long)]
        json: bool,
    },
    /// Recover and decrypt a terminal result or diagnostic with a local key
    Recover {
        run_id: String,
        #[arg(long)]
        project: Option<Uuid>,
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

impl JobRecoveryCommand {
    fn machine_output(&self) -> bool {
        match self {
            Self::Keygen { json, .. }
            | Self::Configure { json, .. }
            | Self::Disable { json, .. }
            | Self::Show { json, .. }
            | Self::Recover { json, .. } => *json,
        }
    }
}
