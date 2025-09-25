use std::path::PathBuf;

use anyhow::{anyhow, Result};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CodexRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub doc_markdown: Option<String>,
    pub sources: Vec<PathBuf>,
}

#[derive(Debug, Serialize)]
pub struct CodexResponse {
    pub summary: String,
}

pub trait CodexClient {
    fn run(&self, request: &CodexRequest) -> Result<CodexResponse>;
}

#[cfg(feature = "embedded-codex")]
pub fn default_client() -> Result<Box<dyn CodexClient>> {
    // Placeholder for future integration with codex-rs crate.
    Err(anyhow!("embedded Codex integration not yet implemented"))
}

#[cfg(not(feature = "embedded-codex"))]
pub fn default_client() -> Result<Box<dyn CodexClient>> {
    Ok(Box::new(StubCodexClient))
}

#[cfg(not(feature = "embedded-codex"))]
struct StubCodexClient;

#[cfg(not(feature = "embedded-codex"))]
impl CodexClient for StubCodexClient {
    fn run(&self, _request: &CodexRequest) -> Result<CodexResponse> {
        Err(anyhow!(
            "Codex integration requires the 'embedded-codex' feature and linked codex-rs"
        ))
    }
}
