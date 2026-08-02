use runmat_package_cache_native::http::{AccessTokenProvider, AccessTokenSnapshot};
use runmat_server_client::auth::{resolve_auth_token, resolve_server_url, RemoteConfig};

#[derive(Debug, Default)]
pub(super) struct CliAccessTokenProvider;

impl AccessTokenProvider for CliAccessTokenProvider {
    fn snapshot<'a>(
        &'a self,
        origin: &'a str,
    ) -> futures::future::LocalBoxFuture<'a, Result<AccessTokenSnapshot, String>> {
        Box::pin(async move { credentials(origin).await })
    }

    fn refresh_after_rejection<'a>(
        &'a self,
        origin: &'a str,
        _observed_generation: u64,
    ) -> futures::future::LocalBoxFuture<'a, Result<AccessTokenSnapshot, String>> {
        Box::pin(async move { credentials(origin).await })
    }
}

async fn credentials(origin: &str) -> Result<AccessTokenSnapshot, String> {
    let mut config = RemoteConfig::load().map_err(|error| error.to_string())?;
    let configured = resolve_server_url(&config, None).map_err(|error| error.to_string())?;
    let token = if configured.trim_end_matches('/') == origin.trim_end_matches('/') {
        resolve_auth_token(&mut config, &configured).await.ok()
    } else {
        None
    };
    Ok(AccessTokenSnapshot::new(token, 0))
}
