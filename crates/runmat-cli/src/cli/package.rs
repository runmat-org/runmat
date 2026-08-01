use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand, Clone)]
pub enum PackageCommand {
    /// Resolve the project and update runmat.lock when needed
    Resolve(PackageProjectArgs),
    /// Fetch every selected immutable dependency into the shared cache
    Fetch(PackageProjectArgs),
    /// Explicitly update mutable dependency selectors and runmat.lock
    Update(PackageProjectArgs),
    /// Print the resolved dependency tree
    Tree(PackageProjectArgs),
    /// Explain which dependency paths select an alias or package
    Why {
        /// Dependency alias or canonical package ID
        query: String,
        #[command(flatten)]
        project: PackageProjectArgs,
    },
    /// Copy the complete immutable dependency closure into a project-local directory
    Vendor {
        #[command(flatten)]
        project: PackageProjectArgs,
        /// Vendor output directory
        #[arg(long, default_value = "vendor")]
        output: PathBuf,
    },
    /// Inspect or collect the shared package cache
    Cache {
        #[command(subcommand)]
        command: PackageCacheCommand,
    },
    /// Manage recipient keys for private registry packages
    Keys {
        #[command(subcommand)]
        command: PackageKeyCommand,
    },
    /// Build, upload, verify, approve, and finalize a package release
    Publish(PackagePublishArgs),
    /// Build and inspect the exact deterministic release payload without uploading it
    Inspect(PackageInspectArgs),
}

#[derive(clap::Args, Clone, Debug)]
pub struct PackageInspectArgs {
    /// Project manifest path
    #[arg(long, default_value = "runmat.toml")]
    pub manifest_path: PathBuf,
    /// Permit native library entries selected by the publication policy
    #[arg(long)]
    pub allow_native: bool,
    /// Emit the release manifest and inventory as structured JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Clone, Debug)]
pub struct PackagePublishArgs {
    #[command(flatten)]
    pub artifact: PackageInspectArgs,
    /// Owning organization ID
    #[arg(long)]
    pub org_id: String,
    /// Registry package ID returned when the package was created
    #[arg(long)]
    pub package_id: String,
    /// Registry origin; defaults to the configured RunMat Server
    #[arg(long)]
    pub registry: Option<String>,
    /// Encrypt the artifact for every active registered recipient key
    #[arg(long = "private")]
    pub private_package: bool,
    /// Monotonic private-package content-key version
    #[arg(long, default_value = "1")]
    pub key_version: u64,
}

#[derive(Subcommand, Clone)]
pub enum PackageKeyCommand {
    /// Register a P-256 recipient key and secure its private half in the OS credential store
    Register(PackageKeyTarget),
    /// List this principal's recipient keys
    List(PackageKeyTarget),
    /// Revoke a recipient key and remove its private half from the OS credential store
    Revoke {
        #[command(flatten)]
        target: PackageKeyTarget,
        /// Recipient key ID returned by registration
        key_id: String,
    },
}

#[derive(clap::Args, Clone, Debug)]
pub struct PackageKeyTarget {
    /// Package namespace
    pub namespace: String,
    /// Package name
    pub name: String,
    /// Registry origin; defaults to the configured RunMat Server
    #[arg(long)]
    pub registry: Option<String>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct PackageProjectArgs {
    /// Project manifest path
    #[arg(long, default_value = "runmat.toml")]
    pub manifest_path: PathBuf,
}

#[derive(Subcommand, Clone)]
pub enum PackageCacheCommand {
    /// Show deterministic cache statistics
    Status {
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
    /// Reclaim least-recently-used unprotected payloads up to a byte target
    Gc {
        /// Desired number of bytes to reclaim
        #[arg(long, default_value = "0")]
        target_bytes: u64,
    },
    /// Remove every object not protected by a pin or active lease
    Prune,
}
