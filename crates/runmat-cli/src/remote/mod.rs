mod fs;
mod git;
mod history;
mod login;
mod orgs;
mod projects;
mod retention;
mod shared;
mod snapshots;

use anyhow::Result;

use crate::cli::{
    FsCommand, OrgCommand, ProjectCommand, ProjectMembersCommand, ProjectRetentionCommand,
    RemoteCommand,
};

pub async fn execute_login_command(
    server: Option<String>,
    api_key: Option<String>,
    email: Option<String>,
    credential_store: runmat_server_client::auth::CredentialStoreMode,
    org: Option<uuid::Uuid>,
    project: Option<uuid::Uuid>,
) -> Result<()> {
    login::execute_login_command(server, api_key, email, credential_store, org, project).await
}

pub async fn execute_org_command(command: OrgCommand) -> Result<()> {
    match command {
        OrgCommand::List { limit, cursor } => orgs::list_orgs(limit, cursor).await,
    }
}

pub async fn execute_project_command(command: ProjectCommand) -> Result<()> {
    match command {
        ProjectCommand::List { org, limit, cursor } => {
            projects::list_projects(org, limit, cursor).await
        }
        ProjectCommand::Create { org, name } => projects::create_project(org, name).await,
        ProjectCommand::Members { members_command } => {
            execute_project_members_command(members_command).await
        }
        ProjectCommand::Fs { fs_command } => execute_fs_command(fs_command).await,
        ProjectCommand::Retention { retention_command } => {
            execute_project_retention_command(retention_command).await
        }
        ProjectCommand::Select { project } => projects::select_project(project).await,
    }
}

pub async fn execute_fs_command(command: FsCommand) -> Result<()> {
    fs::execute_fs_command(command).await
}

pub async fn execute_remote_command(
    command: RemoteCommand,
    cli: &crate::cli::Cli,
    config: &runmat_config::RunMatConfig,
) -> Result<()> {
    match command {
        RemoteCommand::Run {
            script,
            project,
            server,
        } => fs::run_with_remote_fs(script, project, server, cli, config).await,
    }
}

async fn execute_project_members_command(command: ProjectMembersCommand) -> Result<()> {
    match command {
        ProjectMembersCommand::List {
            project,
            limit,
            cursor,
        } => projects::list_project_members(project, limit, cursor).await,
    }
}

async fn execute_project_retention_command(command: ProjectRetentionCommand) -> Result<()> {
    match command {
        ProjectRetentionCommand::Get { project } => retention::get_project_retention(project).await,
        ProjectRetentionCommand::Set {
            max_versions,
            project,
        } => retention::set_project_retention(project, max_versions).await,
    }
}

#[cfg(test)]
mod tests {
    use runmat_server_client::auth::{resolve_project_id, RemoteConfig};
    use std::env;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn remote_config_roundtrip() {
        let dir = tempdir().expect("tempdir");
        env::set_var("RUNMAT_CLI_CONFIG_DIR", dir.path());
        let config = RemoteConfig {
            server_url: Some("http://localhost".to_string()),
            org_id: Some(Uuid::new_v4()),
            project_id: Some(Uuid::new_v4()),
            ..RemoteConfig::default()
        };
        config.save().expect("save");

        let loaded = RemoteConfig::load().expect("load");
        assert_eq!(loaded.server_url, config.server_url);
        assert_eq!(loaded.org_id, config.org_id);
        assert_eq!(loaded.project_id, config.project_id);
        env::remove_var("RUNMAT_CLI_CONFIG_DIR");
    }

    #[test]
    fn resolve_project_id_from_env() {
        let project_id = Uuid::new_v4();
        env::set_var("RUNMAT_PROJECT_ID", project_id.to_string());
        let config = RemoteConfig::default();
        let resolved = resolve_project_id(&config, None).expect("resolve");
        assert_eq!(resolved, project_id);
        env::remove_var("RUNMAT_PROJECT_ID");
    }
}
