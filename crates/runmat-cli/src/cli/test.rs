use std::path::PathBuf;

use clap::{Args, ValueEnum};

#[derive(Args, Clone, Debug)]
pub struct TestArgs {
    /// Test files or directories; defaults to manifest test roots
    #[arg(value_name = "PATH")]
    pub targets: Vec<PathBuf>,

    /// Retain tests whose display name contains this value
    #[arg(long = "name", value_name = "PATTERN")]
    pub names: Vec<String>,

    /// Require a test tag (repeatable)
    #[arg(long = "tag")]
    pub tags: Vec<String>,

    /// Exclude a test tag (repeatable)
    #[arg(long = "exclude-tag")]
    pub excluded_tags: Vec<String>,

    /// Maximum local fixture groups executing concurrently
    #[arg(short = 'j', long)]
    pub jobs: Option<usize>,

    /// Worker isolation policy
    #[arg(long, value_enum)]
    pub isolation: Option<TestIsolationArg>,

    /// Per-test timeout in milliseconds
    #[arg(long = "timeout-ms")]
    pub timeout_ms: Option<u64>,

    /// Add a test support source path
    #[arg(long = "path", value_name = "DIRECTORY")]
    pub paths: Vec<PathBuf>,

    /// Allow one parent environment variable in workers
    #[arg(long = "allow-env", value_name = "NAME")]
    pub environment_allowlist: Vec<String>,

    /// Emit a report format (repeatable)
    #[arg(long = "report", value_enum)]
    pub reports: Vec<TestReportArg>,

    /// Directory for machine reports and captured artifacts
    #[arg(long = "report-dir", default_value = "test-results")]
    pub report_dir: PathBuf,

    /// Select one deterministic zero-based shard
    #[arg(long = "shard-index", requires = "shard_count")]
    pub shard_index: Option<u32>,

    /// Number of deterministic shards
    #[arg(long = "shard-count", requires = "shard_index")]
    pub shard_count: Option<u32>,

    /// Discover and print selected tests without executing them
    #[arg(long)]
    pub list: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestIsolationArg {
    Auto,
    Process,
    Session,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestReportArg {
    Human,
    Json,
    Junit,
    Tap,
}
