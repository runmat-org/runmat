use anyhow::Result;
use runmat_server_client::auth::{execute_login, CredentialStoreMode};
use uuid::Uuid;

pub async fn execute_login_command(
    server: Option<String>,
    api_key: Option<String>,
    email: Option<String>,
    credential_store: CredentialStoreMode,
    org: Option<Uuid>,
    project: Option<Uuid>,
) -> Result<()> {
    execute_login(server, api_key, email, org, project, Some(credential_store)).await
}
