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
