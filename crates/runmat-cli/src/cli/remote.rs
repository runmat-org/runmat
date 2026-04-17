use clap::Subcommand;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Subcommand, Clone)]
pub enum OrgCommand {
    /// List organizations
    List {
        /// Page size
        #[arg(long)]
        limit: Option<u32>,
        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,
    },
}

#[derive(Subcommand, Clone)]
pub enum ProjectCommand {
    /// List projects for an organization
    List {
        /// Organization id
        #[arg(long)]
        org: Option<Uuid>,
        /// Page size
        #[arg(long)]
        limit: Option<u32>,
        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Create a project
    Create {
        /// Organization id
        #[arg(long)]
        org: Option<Uuid>,
        /// Project name
        name: String,
    },
    /// Project membership commands
    Members {
        #[command(subcommand)]
        members_command: ProjectMembersCommand,
    },
    /// Project filesystem commands
    Fs {
        #[command(subcommand)]
        fs_command: FsCommand,
    },
    /// Project retention policy
    Retention {
        #[command(subcommand)]
        retention_command: ProjectRetentionCommand,
    },
    /// Set default project
    Select {
        /// Project id
        project: Uuid,
    },
}

#[derive(Subcommand, Clone)]
pub enum ProjectMembersCommand {
    /// List project members
    List {
        /// Project id
        #[arg(long)]
        project: Option<Uuid>,
        /// Page size
        #[arg(long)]
        limit: Option<u32>,
        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,
    },
}

#[derive(Subcommand, Clone)]
pub enum ProjectRetentionCommand {
    /// Show retention settings
    Get {
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Set retention max versions
    Set {
        /// Max versions to keep (0 = unlimited)
        max_versions: usize,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
}

#[derive(Subcommand, Clone)]
pub enum FsCommand {
    /// Read a remote file
    Read {
        /// Remote path
        path: String,
        /// Output file (stdout when omitted)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Write a remote file from local input
    Write {
        /// Remote path
        path: String,
        /// Input file
        input: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List directory contents
    Ls {
        /// Remote path
        #[arg(default_value = "/")]
        path: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Create directory
    Mkdir {
        /// Remote path
        path: String,
        /// Create parent directories
        #[arg(short, long)]
        recursive: bool,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Remove file or directory
    Rm {
        /// Remote path
        path: String,
        /// Remove directory
        #[arg(long)]
        dir: bool,
        /// Remove directory recursively
        #[arg(short, long)]
        recursive: bool,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List file history
    History {
        /// Remote path
        path: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Restore file version
    Restore {
        /// Version id to restore
        version: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Delete file version
    HistoryDelete {
        /// Version id to delete
        version: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List filesystem snapshots
    SnapshotList {
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Create filesystem snapshot
    SnapshotCreate {
        /// Optional message
        #[arg(long)]
        message: Option<String>,
        /// Parent snapshot override
        #[arg(long)]
        parent: Option<Uuid>,
        /// Optional tag
        #[arg(long)]
        tag: Option<String>,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Restore filesystem snapshot
    SnapshotRestore {
        /// Snapshot id to restore
        snapshot: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Delete filesystem snapshot
    SnapshotDelete {
        /// Snapshot id to delete
        snapshot: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List snapshot tags
    SnapshotTagList {
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Set snapshot tag
    SnapshotTagSet {
        /// Snapshot id to tag
        snapshot: Uuid,
        /// Tag name
        tag: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Delete snapshot tag
    SnapshotTagDelete {
        /// Tag name
        tag: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Clone project snapshots into a git repo
    GitClone {
        /// Destination directory
        directory: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server override
        #[arg(long)]
        server: Option<String>,
    },
    /// Pull latest snapshots into git repo
    GitPull {
        /// Repo directory
        #[arg(default_value = ".")]
        directory: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server override
        #[arg(long)]
        server: Option<String>,
    },
    /// Push git repo history into project snapshots
    GitPush {
        /// Repo directory
        #[arg(default_value = ".")]
        directory: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server override
        #[arg(long)]
        server: Option<String>,
    },
    /// List shard manifest history
    ManifestHistory {
        /// Remote path
        path: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Restore shard manifest version
    ManifestRestore {
        /// Version id to restore
        version: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Update shard manifest with partial edits
    ManifestUpdate {
        /// Remote path
        path: String,
        /// Base manifest version id
        #[arg(long)]
        base_version: Uuid,
        /// Manifest JSON file
        #[arg(long)]
        manifest: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
}

#[derive(Subcommand, Clone)]
pub enum RemoteCommand {
    /// Load and run a script from the remote filesystem
    Run {
        /// Remote script path
        script: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server URL override
        #[arg(long)]
        server: Option<String>,
    },
}
