use futures::future::LocalBoxFuture;
use zeroize::Zeroize;

pub struct AccessTokenSnapshot {
    token: Option<String>,
    pub generation: u64,
}

impl AccessTokenSnapshot {
    pub fn new(token: Option<String>, generation: u64) -> Self {
        Self { token, generation }
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }
}

impl std::fmt::Debug for AccessTokenSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AccessTokenSnapshot")
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("generation", &self.generation)
            .finish()
    }
}

impl Drop for AccessTokenSnapshot {
    fn drop(&mut self) {
        if let Some(token) = &mut self.token {
            token.zeroize();
        }
    }
}

pub trait AccessTokenProvider: Send + Sync {
    fn snapshot<'a>(
        &'a self,
        origin: &'a str,
    ) -> LocalBoxFuture<'a, Result<AccessTokenSnapshot, String>>;

    fn refresh_after_rejection<'a>(
        &'a self,
        origin: &'a str,
        observed_generation: u64,
    ) -> LocalBoxFuture<'a, Result<AccessTokenSnapshot, String>>;
}

#[derive(Default)]
pub struct StaticAccessTokenProvider {
    origin: Option<String>,
    token: Option<String>,
}

impl StaticAccessTokenProvider {
    pub fn new(origin: Option<String>, token: Option<String>) -> Self {
        Self { origin, token }
    }

    fn credentials(&self, origin: &str) -> AccessTokenSnapshot {
        let token = self
            .origin
            .as_deref()
            .filter(|configured| configured.trim_end_matches('/') == origin.trim_end_matches('/'))
            .and(self.token.clone());
        AccessTokenSnapshot::new(token, 0)
    }
}

impl AccessTokenProvider for StaticAccessTokenProvider {
    fn snapshot<'a>(
        &'a self,
        origin: &'a str,
    ) -> LocalBoxFuture<'a, Result<AccessTokenSnapshot, String>> {
        Box::pin(async move { Ok(self.credentials(origin)) })
    }

    fn refresh_after_rejection<'a>(
        &'a self,
        origin: &'a str,
        _observed_generation: u64,
    ) -> LocalBoxFuture<'a, Result<AccessTokenSnapshot, String>> {
        Box::pin(async move { Ok(self.credentials(origin)) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_redacts_token_material() {
        let credentials = AccessTokenSnapshot::new(Some("secret-token".into()), 7);
        let debug = format!("{credentials:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("secret-token"));
    }

    #[test]
    fn static_credentials_are_scoped_to_the_configured_origin() {
        let provider = StaticAccessTokenProvider::new(
            Some("https://api.runmat.test".into()),
            Some("secret".into()),
        );
        assert_eq!(
            provider.credentials("https://api.runmat.test/").token(),
            Some("secret")
        );
        assert_eq!(
            provider.credentials("https://other.runmat.test").token(),
            None
        );
    }
}
