use runmat_server_client::auth::{resolve_auth_token, resolve_server_url, RemoteConfig};
use runmat_server_client::packages::RegistryClient;

pub(super) async fn registry_client(index: &str) -> Result<RegistryClient, String> {
    let mut config = RemoteConfig::load().map_err(|error| error.to_string())?;
    let configured = resolve_server_url(&config, None).map_err(|error| error.to_string())?;
    let token = if configured.trim_end_matches('/') == index.trim_end_matches('/') {
        resolve_auth_token(&mut config, &configured).await.ok()
    } else {
        None
    };
    RegistryClient::new(index, token).map_err(|error| error.to_string())
}
