use std::path::PathBuf;

#[cfg(feature = "embedded-codex")]
use std::{ffi::OsString, sync::Arc};

#[cfg(feature = "embedded-codex")]
use codex_core::{ContentItem, ModelClient, Prompt, ResponseEvent, ResponseItem};
#[cfg(feature = "embedded-codex")]
use codex_protocol::mcp_protocol::ConversationId;
#[cfg(feature = "embedded-codex")]
use core_test_support::load_default_config_for_test;
#[cfg(feature = "embedded-codex")]
use futures::StreamExt;
#[cfg(feature = "embedded-codex")]
use tempfile::TempDir;

use anyhow::Result;

#[cfg(not(feature = "embedded-codex"))]
use anyhow::anyhow;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CodexRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub doc_markdown: Option<String>,
    pub sources: Vec<PathBuf>,
}

#[derive(Debug, Serialize, Clone)]
pub struct CodexResponse {
    pub summary: String,
}

pub trait CodexClient {
    fn run(&self, request: &CodexRequest) -> Result<CodexResponse>;
}

#[cfg(feature = "embedded-codex")]
pub fn default_client() -> Result<Box<dyn CodexClient>> {
    Ok(Box::new(EmbeddedCodexClient))
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

#[cfg(feature = "embedded-codex")]
struct EmbeddedCodexClient;

#[cfg(feature = "embedded-codex")]
impl CodexClient for EmbeddedCodexClient {
    fn run(&self, request: &CodexRequest) -> Result<CodexResponse> {
        const FIXTURE_SSE: &str = include_str!("../../tests/fixtures/codex_fixture.sse");

        let runtime = tokio::runtime::Runtime::new()?;
        let codex_home = TempDir::new()?;

        let config = Arc::new(load_default_config_for_test(&codex_home));
        let provider = config.model_provider.clone();
        let effort = config.model_reasoning_effort;
        let summary = config.model_reasoning_summary;

        // Write fixture to temp file and configure Codex to read from it.
        let fixture_path = codex_home.path().join("embedded_fixture.sse");
        std::fs::write(&fixture_path, FIXTURE_SSE)?;

        let _fixture_guard = EnvVarGuard::set("CODEX_RS_SSE_FIXTURE", &fixture_path);
        let _api_key_guard = EnvVarGuard::set("OPENAI_API_KEY", "dummy");
        let _base_url_guard = EnvVarGuard::set("OPENAI_BASE_URL", "http://unused.local");

        let client = ModelClient::new(
            Arc::clone(&config),
            None,
            provider,
            effort,
            summary,
            ConversationId::new(),
        );

        let mut prompt = Prompt::default();
        prompt.input.push(ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: request.prompt.clone(),
            }],
        });
        if let Some(doc) = &request.doc_markdown {
            prompt.input.push(ResponseItem::Message {
                id: None,
                role: "system".to_string(),
                content: vec![ContentItem::InputText { text: doc.clone() }],
            });
        }

        let mut summary_text = runtime.block_on(async {
            let mut stream = client.stream(&prompt).await?;
            let mut collected = String::new();
            while let Some(event) = stream.next().await {
                match event? {
                    ResponseEvent::OutputTextDelta(delta) => {
                        collected.push_str(&delta);
                    }
                    ResponseEvent::OutputItemDone(item) => {
                        if let ResponseItem::Message { content, .. } = item {
                            for chunk in content {
                                if let ContentItem::OutputText { text } = chunk {
                                    collected.push_str(&text);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Result::<String>::Ok(collected)
        })?;

        if summary_text.trim().is_empty() {
            summary_text = "Codex returned no response".to_string();
        }

        Ok(CodexResponse {
            summary: summary_text,
        })
    }
}

#[cfg(feature = "embedded-codex")]
struct EnvVarGuard {
    key: String,
    previous: Option<OsString>,
}

#[cfg(feature = "embedded-codex")]
impl EnvVarGuard {
    fn set(key: &str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self {
            key: key.to_string(),
            previous,
        }
    }
}

#[cfg(feature = "embedded-codex")]
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(val) => std::env::set_var(&self.key, val),
            None => std::env::remove_var(&self.key),
        }
    }
}

#[cfg(feature = "embedded-codex")]
pub fn is_available() -> bool {
    true
}

#[cfg(not(feature = "embedded-codex"))]
pub fn is_available() -> bool {
    false
}

#[cfg(all(test, not(feature = "embedded-codex")))]
mod tests {
    #[test]
    fn stub_reports_unavailable() {
        assert!(!super::is_available());
    }
}

#[cfg(all(test, feature = "embedded-codex"))]
mod embedded_tests {
    #[test]
    fn embedded_reports_available() {
        assert!(super::is_available());
    }
}
