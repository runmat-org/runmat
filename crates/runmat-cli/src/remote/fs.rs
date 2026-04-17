use anyhow::{Context, Result};
use runmat_config::RunMatConfig;
use runmat_filesystem::{FsFileType, FsProvider, RemoteFsConfig, RemoteFsProvider};
use runmat_server_client::auth::{
    resolve_auth_token, resolve_project_id, resolve_server_url, RemoteConfig,
};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use uuid::Uuid;

use crate::cli::Cli;
use crate::cli::FsCommand;
use crate::commands::script::execute_script_contents;
use crate::remote::shared::read_u64_env;
use crate::remote::{git, history, snapshots};

pub async fn execute_fs_command(command: FsCommand) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let project_id = match &command {
        FsCommand::Read { project, .. }
        | FsCommand::Write { project, .. }
        | FsCommand::Ls { project, .. }
        | FsCommand::Mkdir { project, .. }
        | FsCommand::Rm { project, .. }
        | FsCommand::History { project, .. }
        | FsCommand::Restore { project, .. }
        | FsCommand::HistoryDelete { project, .. }
        | FsCommand::SnapshotList { project }
        | FsCommand::SnapshotCreate { project, .. }
        | FsCommand::SnapshotRestore { project, .. }
        | FsCommand::SnapshotDelete { project, .. }
        | FsCommand::SnapshotTagList { project }
        | FsCommand::SnapshotTagSet { project, .. }
        | FsCommand::SnapshotTagDelete { project, .. }
        | FsCommand::GitClone { project, .. }
        | FsCommand::GitPull { project, .. }
        | FsCommand::GitPush { project, .. }
        | FsCommand::ManifestHistory { project, .. }
        | FsCommand::ManifestRestore { project, .. }
        | FsCommand::ManifestUpdate { project, .. } => resolve_project_id(&config, *project)?,
    };
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let base_url = format!(
        "{}/v1/projects/{}",
        server_url.trim_end_matches('/'),
        project_id
    );

    if matches!(
        command,
        FsCommand::ManifestHistory { .. }
            | FsCommand::ManifestRestore { .. }
            | FsCommand::ManifestUpdate { .. }
            | FsCommand::History { .. }
            | FsCommand::Restore { .. }
            | FsCommand::HistoryDelete { .. }
            | FsCommand::SnapshotList { .. }
            | FsCommand::SnapshotCreate { .. }
            | FsCommand::SnapshotRestore { .. }
            | FsCommand::SnapshotDelete { .. }
            | FsCommand::SnapshotTagList { .. }
            | FsCommand::SnapshotTagSet { .. }
            | FsCommand::SnapshotTagDelete { .. }
            | FsCommand::GitClone { .. }
            | FsCommand::GitPull { .. }
            | FsCommand::GitPush { .. }
    ) {
        match command {
            FsCommand::History { path, .. } => {
                history::list_history(&server_url, &token, project_id, &path).await?
            }
            FsCommand::Restore { version, .. } => {
                history::restore_version(&server_url, &token, project_id, version).await?
            }
            FsCommand::HistoryDelete { version, .. } => {
                history::delete_version(&server_url, &token, project_id, version).await?
            }
            FsCommand::ManifestHistory { path, .. } => {
                history::list_manifest_history(&server_url, &token, project_id, &path).await?
            }
            FsCommand::ManifestRestore { version, .. } => {
                history::restore_manifest_version(&server_url, &token, project_id, version).await?
            }
            FsCommand::ManifestUpdate {
                path,
                base_version,
                manifest,
                ..
            } => {
                history::update_manifest(
                    &server_url,
                    &token,
                    project_id,
                    &path,
                    base_version,
                    &manifest,
                )
                .await?
            }
            FsCommand::SnapshotList { .. } => {
                snapshots::list_snapshots(&server_url, &token, project_id).await?
            }
            FsCommand::SnapshotCreate {
                message,
                parent,
                tag,
                ..
            } => {
                snapshots::create_snapshot(&server_url, &token, project_id, message, parent, tag)
                    .await?
            }
            FsCommand::SnapshotRestore { snapshot, .. } => {
                snapshots::restore_snapshot(&server_url, &token, project_id, snapshot).await?
            }
            FsCommand::SnapshotDelete { snapshot, .. } => {
                snapshots::delete_snapshot(&server_url, &token, project_id, snapshot).await?
            }
            FsCommand::SnapshotTagList { .. } => {
                snapshots::list_snapshot_tags(&server_url, &token, project_id).await?
            }
            FsCommand::SnapshotTagSet { snapshot, tag, .. } => {
                snapshots::set_snapshot_tag(&server_url, &token, project_id, snapshot, &tag).await?
            }
            FsCommand::SnapshotTagDelete { tag, .. } => {
                snapshots::delete_snapshot_tag(&server_url, &token, project_id, &tag).await?
            }
            FsCommand::GitClone {
                directory, server, ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git::git_clone(&url, &token, project_id, &directory).await?
            }
            FsCommand::GitPull {
                directory, server, ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git::git_pull(&url, &token, project_id, &directory).await?
            }
            FsCommand::GitPush {
                directory, server, ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git::git_push(&url, &token, project_id, &directory).await?
            }
            _ => {}
        }
        return Ok(());
    }

    let runtime = tokio::runtime::Handle::current();
    let task = tokio::task::spawn_blocking(move || -> Result<()> {
        let shard_threshold_bytes = read_u64_env("RUNMAT_FS_SHARD_THRESHOLD_BYTES");
        let shard_size_bytes = read_u64_env("RUNMAT_FS_SHARD_SIZE_BYTES");
        let provider = RemoteFsProvider::new(RemoteFsConfig {
            base_url,
            auth_token: Some(token),
            shard_threshold_bytes: shard_threshold_bytes
                .unwrap_or(RemoteFsConfig::default().shard_threshold_bytes),
            shard_size_bytes: shard_size_bytes
                .unwrap_or(RemoteFsConfig::default().shard_size_bytes),
            ..RemoteFsConfig::default()
        })
        .context("Failed to initialize remote filesystem")?;

        match command {
            FsCommand::Read { path, output, .. } => {
                let bytes = runtime
                    .block_on(provider.read(std::path::Path::new(&path)))
                    .context("Failed to read remote file")?;
                if let Some(output) = output {
                    fs::write(&output, bytes)
                        .with_context(|| format!("Failed to write {}", output.display()))?;
                } else {
                    let mut stdout = io::stdout();
                    stdout.write_all(&bytes).context("Failed to write output")?;
                }
                Ok(())
            }
            FsCommand::Write { path, input, .. } => {
                let bytes = fs::read(&input)
                    .with_context(|| format!("Failed to read {}", input.display()))?;
                runtime
                    .block_on(provider.write(std::path::Path::new(&path), &bytes))
                    .context("Failed to write remote file")?;
                Ok(())
            }
            FsCommand::Ls { path, .. } => {
                let entries = runtime
                    .block_on(provider.read_dir(std::path::Path::new(&path)))
                    .context("Failed to list directory")?;
                for entry in entries {
                    let kind = match entry.file_type() {
                        FsFileType::Directory => "dir",
                        FsFileType::File => "file",
                        FsFileType::Symlink => "symlink",
                        FsFileType::Other => "other",
                        FsFileType::Unknown => "unknown",
                    };
                    println!("{kind}\t{}", entry.path().display());
                }
                Ok(())
            }
            FsCommand::Mkdir {
                path, recursive, ..
            } => {
                if recursive {
                    runtime
                        .block_on(provider.create_dir_all(std::path::Path::new(&path)))
                        .context("Failed to create directory")?;
                } else {
                    runtime
                        .block_on(provider.create_dir(std::path::Path::new(&path)))
                        .context("Failed to create directory")?;
                }
                Ok(())
            }
            FsCommand::Rm {
                path,
                dir,
                recursive,
                ..
            } => {
                if recursive {
                    runtime
                        .block_on(provider.remove_dir_all(std::path::Path::new(&path)))
                        .context("Failed to remove directory")?;
                } else if dir {
                    runtime
                        .block_on(provider.remove_dir(std::path::Path::new(&path)))
                        .context("Failed to remove directory")?;
                } else {
                    runtime
                        .block_on(provider.remove_file(std::path::Path::new(&path)))
                        .context("Failed to remove file")?;
                }
                Ok(())
            }
            FsCommand::History { .. }
            | FsCommand::Restore { .. }
            | FsCommand::HistoryDelete { .. }
            | FsCommand::ManifestHistory { .. }
            | FsCommand::ManifestRestore { .. }
            | FsCommand::ManifestUpdate { .. }
            | FsCommand::SnapshotList { .. }
            | FsCommand::SnapshotCreate { .. }
            | FsCommand::SnapshotRestore { .. }
            | FsCommand::SnapshotDelete { .. }
            | FsCommand::SnapshotTagList { .. }
            | FsCommand::SnapshotTagSet { .. }
            | FsCommand::SnapshotTagDelete { .. }
            | FsCommand::GitClone { .. }
            | FsCommand::GitPull { .. }
            | FsCommand::GitPush { .. } => Ok(()),
        }
    });

    task.await.context("Filesystem task join failed")??;
    Ok(())
}

pub async fn run_with_remote_fs(
    script: PathBuf,
    project: Option<Uuid>,
    server: Option<String>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    let mut remote_config = RemoteConfig::load()?;
    let project_id = resolve_project_id(&remote_config, project)?;
    let server_url = resolve_server_url(&remote_config, server)?;
    let token = resolve_auth_token(&mut remote_config, &server_url).await?;
    let shard_threshold_bytes = read_u64_env("RUNMAT_FS_SHARD_THRESHOLD_BYTES")
        .unwrap_or(RemoteFsConfig::default().shard_threshold_bytes);
    let shard_size_bytes = read_u64_env("RUNMAT_FS_SHARD_SIZE_BYTES")
        .unwrap_or(RemoteFsConfig::default().shard_size_bytes);
    let base_url = format!(
        "{}/v1/projects/{}",
        server_url.trim_end_matches('/'),
        project_id
    );
    let provider = RemoteFsProvider::new(RemoteFsConfig {
        base_url,
        auth_token: Some(token),
        shard_threshold_bytes,
        shard_size_bytes,
        ..RemoteFsConfig::default()
    })
    .context("Failed to initialize remote filesystem")?;
    let script_bytes = provider
        .read(script.as_path())
        .await
        .with_context(|| format!("Failed to read remote script file: {}", script.display()))?;
    let script_content = String::from_utf8(script_bytes)
        .with_context(|| format!("Remote script is not valid UTF-8: {}", script.display()))?;

    runmat_filesystem::set_provider(std::sync::Arc::new(provider));
    let source_name = PathBuf::from(format!("remote:{}", script.display()));
    execute_script_contents(source_name, script_content, cli.emit_bytecode.clone(), cli, config).await
}
