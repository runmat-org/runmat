use anyhow::Result;

use crate::cli::PkgCommand;

pub async fn execute_pkg_command(pkg_command: PkgCommand) -> Result<()> {
    let msg = "RunMat package manager is coming soon. Track progress in the repo.";
    match pkg_command {
        PkgCommand::Add { name } => println!("pkg add {name}: {msg}"),
        PkgCommand::Remove { name } => println!("pkg remove {name}: {msg}"),
        PkgCommand::Install => println!("pkg install: {msg}"),
        PkgCommand::Update => println!("pkg update: {msg}"),
        PkgCommand::Publish => println!("pkg publish: {msg}"),
    }
    Ok(())
}
