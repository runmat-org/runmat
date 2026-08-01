mod cache;
mod private_keys;
mod publication;
mod publication_artifact;
mod publication_manifest;
mod publication_retry;
#[cfg(test)]
mod publication_tests;
mod registry_transport;
mod resolve;
mod server_transport;
mod tree;
mod vendor;

pub(crate) use resolve::{install_project_for_source, resolve_for_source};

use crate::cli::{Cli, PackageCommand};
use anyhow::Result;
use runmat_package::GitAcquisitionIntent;

pub async fn execute(command: PackageCommand, cli: &Cli) -> Result<()> {
    match command {
        PackageCommand::Resolve(args) => {
            resolve::resolve(&args, cli, GitAcquisitionIntent::Execute).await?;
        }
        PackageCommand::Fetch(args) => {
            resolve::resolve(&args, cli, GitAcquisitionIntent::Fetch).await?;
        }
        PackageCommand::Update(args) => {
            resolve::resolve(&args, cli, GitAcquisitionIntent::Update).await?;
        }
        PackageCommand::Tree(args) => {
            let project = resolve::resolve(&args, cli, GitAcquisitionIntent::Execute).await?;
            tree::print_tree(&project.resolved.frozen.graph);
        }
        PackageCommand::Why { query, project } => {
            let project = resolve::resolve(&project, cli, GitAcquisitionIntent::Execute).await?;
            tree::print_why(&project.resolved.frozen.graph, &query)?;
        }
        PackageCommand::Vendor { project, output } => {
            let project = resolve::resolve(&project, cli, GitAcquisitionIntent::Execute).await?;
            vendor::vendor(&project, &output).await?;
        }
        PackageCommand::Cache { command } => cache::execute(command).await?,
        PackageCommand::Keys { command } => private_keys::execute(command).await?,
        PackageCommand::Publish(args) => publication::publish(&args).await?,
        PackageCommand::Inspect(args) => publication::inspect(&args)?,
    }
    Ok(())
}
