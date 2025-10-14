use std::path::PathBuf;

use serde::Serialize;

#[cfg(feature = "embedded-codex")]
use {
    anyhow::{anyhow, bail, Context, Result},
    codex_core::{
        config::{Config, ConfigOverrides},
        protocol::{AskForApproval, SandboxPolicy},
        AuthManager,
    },
    codex_protocol::config_types::ReasoningEffort,
    core_test_support::load_default_config_for_test,
    serde_json::Value,
    std::collections::HashSet,
    std::ffi::OsString,
    std::io::Write,
    std::path::Path,
    std::process::{Command, Stdio},
    std::sync::Arc,
    std::{env, fs},
    tempfile::TempDir,
};

#[cfg(not(feature = "embedded-codex"))]
use anyhow::Result;

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
    let context = load_runtime_context()?;
    let binary = locate_codex_binary()?;
    Ok(Box::new(CliCodexClient { context, binary }))
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
        Err(anyhow::anyhow!(
            "Codex integration requires the 'embedded-codex' feature and linked codex-rs"
        ))
    }
}

#[cfg(feature = "embedded-codex")]
struct CliCodexClient {
    context: RuntimeContext,
    binary: PathBuf,
}

#[cfg(feature = "embedded-codex")]
impl CodexClient for CliCodexClient {
    fn run(&self, request: &CodexRequest) -> Result<CodexResponse> {
        run_via_cli(&self.binary, &self.context, request)
    }
}

#[cfg(feature = "embedded-codex")]
fn run_via_cli(
    binary: &Path,
    context: &RuntimeContext,
    request: &CodexRequest,
) -> Result<CodexResponse> {
    let mut prompt = request.prompt.clone();
    if let Some(doc) = &request.doc_markdown {
        prompt.push_str("\n\nDOC_MD:\n");
        prompt.push_str(doc);
    }

    let model = request
        .model
        .as_deref()
        .unwrap_or(&context.config.model)
        .to_string();

    let mut command = Command::new(binary);
    command
        .arg("exec")
        .arg("--experimental-json")
        .arg("--skip-git-repo-check")
        .arg("--sandbox")
        .arg(match context.config.sandbox_policy {
            SandboxPolicy::DangerFullAccess => "danger-full-access",
            SandboxPolicy::ReadOnly => "read-only",
            SandboxPolicy::WorkspaceWrite { .. } => "workspace-write",
        })
        .arg("--model")
        .arg(&model)
        .arg("--cd")
        .arg(context.config.cwd.as_os_str());

    command
        .current_dir(&context.config.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().context("failed to spawn codex CLI")?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .context("failed to write prompt to codex CLI")?;
    }

    let output = child
        .wait_with_output()
        .context("failed to wait for codex CLI")?;
    if !output.status.success() {
        let stderr_text = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "codex CLI exited with {}: {}",
            output.status,
            stderr_text.trim()
        ));
    }

    let stdout_text =
        String::from_utf8(output.stdout).context("codex CLI produced non-utf8 stdout")?;
    let mut aggregated = String::new();
    let mut last_agent_message: Option<String> = None;
    let mut seen_agent_items: HashSet<String> = HashSet::new();

    for raw_line in stdout_text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(err) => {
                append_with_gap(
                    &mut aggregated,
                    &format!("codex emitted non-json output ({err}): {line}"),
                );
                continue;
            }
        };

        match parsed.get("type").and_then(|v| v.as_str()) {
            Some("item.completed") => {
                if let Some(item) = parsed.get("item") {
                    if let Some(item_type) = item.get("item_type").and_then(|v| v.as_str()) {
                        match item_type {
                            "agent_message" => {
                                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                                    let key = item
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| text.to_string());
                                    if seen_agent_items.insert(key) {
                                        append_with_gap(&mut aggregated, text);
                                        last_agent_message = Some(text.to_string());
                                    }
                                }
                            }
                            "command_execution" => {
                                let command = item
                                    .get("command")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown command");
                                let exit_code = item
                                    .get("exit_code")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or_default();
                                let output = item
                                    .get("aggregated_output")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default();
                                append_with_gap(
                                    &mut aggregated,
                                    &format!(
                                        "exec `{command}` exited {exit_code}\n{}",
                                        output.trim_end()
                                    ),
                                );
                            }
                            "file_change" => {
                                if let Some(status) = item.get("status").and_then(|v| v.as_str()) {
                                    append_with_gap(
                                        &mut aggregated,
                                        &format!("apply_patch status: {status}"),
                                    );
                                }
                            }
                            "error" => {
                                if let Some(message) = item.get("message").and_then(|v| v.as_str())
                                {
                                    append_with_gap(&mut aggregated, message);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some("turn.failed") => {
                let message = parsed
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("codex turn failed");
                return Err(anyhow!(message.to_string()));
            }
            Some("error") => {
                let message = parsed
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("codex error");
                return Err(anyhow!(message.to_string()));
            }
            _ => {}
        }
    }

    let stderr_text = String::from_utf8_lossy(&output.stderr);
    if !stderr_text.trim().is_empty() {
        append_with_gap(
            &mut aggregated,
            &format!("codex stderr:\n{}", stderr_text.trim_end()),
        );
    }

    let summary = last_agent_message
        .filter(|msg| !msg.trim().is_empty())
        .unwrap_or_else(|| {
            let trimmed = aggregated.trim();
            if trimmed.is_empty() {
                "Codex returned no response".to_string()
            } else {
                trimmed.to_string()
            }
        });

    Ok(CodexResponse { summary })
}

#[cfg(feature = "embedded-codex")]
fn locate_codex_binary() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("RUNMATFUNC_CODEX_PATH") {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Ok(candidate);
        } else {
            bail!(
                "RUNMATFUNC_CODEX_PATH points to '{}', but no file exists there",
                candidate.display()
            );
        }
    }

    let path_var = env::var_os("PATH").context("PATH environment variable not set")?;
    for entry in env::split_paths(&path_var) {
        let candidate = entry.join(if cfg!(windows) { "codex.exe" } else { "codex" });
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!("failed to locate `codex` binary in PATH. Set RUNMATFUNC_CODEX_PATH if it resides elsewhere.")
}

#[cfg(feature = "embedded-codex")]
fn append_with_gap(buffer: &mut String, text: &str) {
    if text.trim().is_empty() {
        return;
    }
    if !buffer.is_empty() {
        buffer.push('\n');
    }
    buffer.push_str(text);
}

#[cfg(feature = "embedded-codex")]
pub(crate) struct EnvVarGuard {
    key: String,
    previous: Option<OsString>,
}

#[cfg(feature = "embedded-codex")]
struct FixtureGuard {
    _home: TempDir,
    _sse: EnvVarGuard,
    _api: EnvVarGuard,
    _base: EnvVarGuard,
    _home_env: EnvVarGuard,
}

#[cfg(feature = "embedded-codex")]
struct RuntimeContext {
    config: Arc<Config>,
    _fixture_guard: Option<FixtureGuard>,
}

#[cfg(feature = "embedded-codex")]
impl EnvVarGuard {
    pub(crate) fn set(key: &str, value: impl AsRef<std::ffi::OsStr>) -> Self {
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
fn load_runtime_context() -> Result<RuntimeContext> {
    match attempt_user_context()? {
        UserContextStatus::Ready(context) => Ok(context),
        UserContextStatus::MissingConfig if should_use_fixture() => load_fixture_context(),
        UserContextStatus::MissingAuth if should_use_fixture() => load_fixture_context(),
        UserContextStatus::MissingConfig => bail!(
            "Codex configuration not found. Configure Codex (see https://github.com/openai/codex/blob/main/docs/config.md) or set RUNMATFUNC_USE_CODEX_FIXTURE=1 when running tests."
        ),
        UserContextStatus::MissingAuth => bail!(
            "Codex authentication not available. Run `codex auth login` or set OPENAI_API_KEY before enabling --codex."
        ),
    }
}

#[cfg(feature = "embedded-codex")]
fn attempt_user_context() -> Result<UserContextStatus> {
    let config = match Config::load_with_cli_overrides(Vec::new(), ConfigOverrides::default()) {
        Ok(cfg) => cfg,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(UserContextStatus::MissingConfig)
        }
        Err(err) => return Err(err.into()),
    };

    let mut config = config;
    config.model = "gpt-5-codex".to_string();
    config.model_reasoning_effort = Some(ReasoningEffort::High);
    config.cwd = std::env::current_dir()?;
    config.approval_policy = AskForApproval::Never;
    config.sandbox_policy = SandboxPolicy::DangerFullAccess;

    let auth_manager = AuthManager::shared(config.codex_home.clone());
    let has_token = auth_manager.auth().is_some()
        || std::env::var("OPENAI_API_KEY")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);

    if !has_token {
        return Ok(UserContextStatus::MissingAuth);
    }

    Ok(UserContextStatus::Ready(RuntimeContext {
        config: Arc::new(config),
        _fixture_guard: None,
    }))
}

#[cfg(feature = "embedded-codex")]
fn load_fixture_context() -> Result<RuntimeContext> {
    const FIXTURE_SSE: &str = include_str!("../../tests/fixtures/codex_fixture.sse");

    let codex_home = TempDir::new()?;
    let mut config = load_default_config_for_test(&codex_home);
    config.model = "gpt-5-codex".to_string();
    config.model_reasoning_effort = Some(ReasoningEffort::High);
    config.cwd = std::env::current_dir()?;
    config.approval_policy = AskForApproval::Never;
    config.sandbox_policy = SandboxPolicy::DangerFullAccess;
    let config = Arc::new(config);

    let fixture_path = codex_home.path().join("embedded_fixture.sse");
    fs::write(&fixture_path, FIXTURE_SSE).context("writing Codex SSE fixture")?;

    let sse_guard = EnvVarGuard::set("CODEX_RS_SSE_FIXTURE", &fixture_path);
    let api_guard = EnvVarGuard::set("OPENAI_API_KEY", "dummy");
    let base_guard = EnvVarGuard::set("OPENAI_BASE_URL", "http://unused.local");
    let home_guard = EnvVarGuard::set("CODEX_HOME", codex_home.path());

    Ok(RuntimeContext {
        config,
        _fixture_guard: Some(FixtureGuard {
            _home: codex_home,
            _sse: sse_guard,
            _api: api_guard,
            _base: base_guard,
            _home_env: home_guard,
        }),
    })
}

#[cfg(feature = "embedded-codex")]
enum UserContextStatus {
    Ready(RuntimeContext),
    MissingConfig,
    MissingAuth,
}

#[cfg(feature = "embedded-codex")]
fn should_use_fixture() -> bool {
    matches!(
        std::env::var("RUNMATFUNC_USE_CODEX_FIXTURE")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

#[cfg(feature = "embedded-codex")]
pub fn is_available() -> bool {
    matches!(attempt_user_context(), Ok(UserContextStatus::Ready(_))) || should_use_fixture()
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
    use std::fs;
    use std::path::PathBuf;

    use super::{default_client, CodexRequest, EnvVarGuard};
    use tempfile::TempDir;

    fn codex_binary_available() -> bool {
        super::locate_codex_binary().is_ok()
    }

    #[test]
    fn embedded_reports_available() {
        if !codex_binary_available() {
            eprintln!("codex binary not available; skipping test");
            return;
        }

        let _fixture_flag = EnvVarGuard::set("RUNMATFUNC_USE_CODEX_FIXTURE", "1");
        let temp_home = TempDir::new().expect("temp dir");
        let _home_guard = EnvVarGuard::set("CODEX_HOME", temp_home.path());
        assert!(super::is_available());
    }

    #[test]
    fn fixture_applies_patch() -> anyhow::Result<()> {
        if !codex_binary_available() {
            eprintln!("codex binary not available; skipping fixture test");
            return Ok(());
        }

        let _fixture_flag = EnvVarGuard::set("RUNMATFUNC_USE_CODEX_FIXTURE", "1");
        let temp_home = TempDir::new()?;
        let _home_guard = EnvVarGuard::set("CODEX_HOME", temp_home.path());

        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixture.txt");
        if fixture_path.exists() {
            fs::remove_file(&fixture_path)?;
        }

        let client = default_client()?;
        let request = CodexRequest {
            model: None,
            prompt: "apply fixture patch".to_string(),
            doc_markdown: None,
            sources: vec![],
        };
        let response = match client.run(&request) {
            Ok(resp) => resp,
            Err(err) => {
                eprintln!("codex CLI unavailable for fixture test: {err:?}; skipping");
                return Ok(());
            }
        };

        // wait briefly for the CLI to finish writing the file
        let mut attempts = 0;
        while attempts < 50 && !fixture_path.exists() {
            std::thread::sleep(std::time::Duration::from_millis(20));
            attempts += 1;
        }

        assert!(fixture_path.exists());
        let contents = fs::read_to_string(&fixture_path)?;
        assert_eq!(contents, "fixture contents\n");
        fs::remove_file(fixture_path)?;
        assert!(response.summary.contains("Applied"));
        Ok(())
    }
}
