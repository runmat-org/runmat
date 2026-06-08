use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

pub(crate) fn build_http_client(
    server_url: &str,
    token: &str,
) -> Result<(reqwest::Client, String)> {
    let mut headers = HeaderMap::new();
    let auth_value =
        HeaderValue::from_str(&format!("Bearer {token}")).context("Invalid auth token")?;
    headers.insert(AUTHORIZATION, auth_value);
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .context("Failed to build HTTP client")?;
    Ok((client, server_url.trim_end_matches('/').to_string()))
}

pub(crate) fn read_u64_env(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|value| value.parse().ok())
}
