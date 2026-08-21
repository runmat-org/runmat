use std::path::PathBuf;

use clap::Subcommand;
use runmat_execution::JobId;

#[derive(Subcommand, Clone)]
pub enum BatchCommand {
    /// Submit a script as a durable local job
    Submit {
        /// MATLAB script to freeze and submit
        file: PathBuf,
        /// Stable key for safely retrying the same submission
        #[arg(long)]
        idempotency_key: Option<String>,
        /// Retain terminal job metadata and logs for this many hours
        #[arg(long, default_value = "168")]
        retention_hours: u64,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
        /// Arguments passed to the script
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// List durable local jobs
    List {
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
    /// Show one durable local job
    Show {
        job_id: JobId,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
    /// Attach to a durable local job's output
    Attach {
        job_id: JobId,
        /// Print currently available output and return without following
        #[arg(long)]
        no_follow: bool,
    },
    /// Cancel a durable local job
    Cancel {
        job_id: JobId,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
}
