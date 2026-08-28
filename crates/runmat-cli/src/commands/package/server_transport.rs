use runmat_server_client::auth::{resolve_server_url, RemoteConfig, DEFAULT_SERVER_URL};

pub(super) fn default_server_origin() -> String {
    let config = RemoteConfig::load().unwrap_or_default();
    resolve_server_url(&config, None).unwrap_or_else(|_| DEFAULT_SERVER_URL.to_string())
}
