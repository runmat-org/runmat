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
}
