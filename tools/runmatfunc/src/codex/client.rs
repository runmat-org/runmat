use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

#[cfg(feature = "embedded-codex")]
use std::{env, ffi::OsString, fs, io::ErrorKind, sync::Arc};

use anyhow::Result;
#[cfg(feature = "embedded-codex")]
use anyhow::{anyhow, bail, Context};

#[cfg(feature = "embedded-codex")]
use codex_core::{
    config::{Config, ConfigOverrides},
    protocol::{
        AskForApproval, EventMsg, FileChange, InputItem, Op, PatchApplyBeginEvent, ReviewDecision,
        SandboxPolicy, Submission,
    },
    AuthManager, CodexAuth, ConversationManager,
};
#[cfg(feature = "embedded-codex")]
use codex_protocol::config_types::ReasoningEffort;
#[cfg(feature = "embedded-codex")]
use core_test_support::load_default_config_for_test;
#[cfg(feature = "embedded-codex")]
use tempfile::TempDir;

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
        Err(anyhow::anyhow!(
            "Codex integration requires the 'embedded-codex' feature and linked codex-rs"
        ))
    }
}

#[cfg(feature = "embedded-codex")]
struct EmbeddedCodexClient;

#[cfg(feature = "embedded-codex")]
impl CodexClient for EmbeddedCodexClient {
    fn run(&self, request: &CodexRequest) -> Result<CodexResponse> {
        let runtime = tokio::runtime::Runtime::new()?;
        let context = load_runtime_context()?;
        runtime.block_on(async { run_conversation(context, request).await })
    }
}

#[cfg(feature = "embedded-codex")]
async fn run_conversation(
    context: RuntimeContext,
    request: &CodexRequest,
) -> Result<CodexResponse> {
    let manager = ConversationManager::new(Arc::clone(&context.auth_manager));
    let config_owned = (*context.config).clone();
    let conversation = manager
        .new_conversation(config_owned)
        .await
        .map(|new_conv| (new_conv.conversation_id, new_conv.conversation))?;
    let (conversation_id, conversation) = conversation;

    let result = drive_turn(&conversation, &context, request).await;
    let _ = manager.remove_conversation(&conversation_id).await;
    result
}

#[cfg(feature = "embedded-codex")]
async fn drive_turn(
    conversation: &Arc<codex_core::CodexConversation>,
    context: &RuntimeContext,
    request: &CodexRequest,
) -> Result<CodexResponse> {
    let mut text = request.prompt.clone();
    if let Some(doc) = &request.doc_markdown {
        text.push_str("\n\nDOC_MD:\n");
        text.push_str(doc);
    }

    let items = vec![InputItem::Text { text }];
    let submission = Submission {
        id: "turn-1".to_string(),
        op: Op::UserTurn {
            items,
            cwd: context.config.cwd.clone(),
            approval_policy: context.config.approval_policy,
            sandbox_policy: context.config.sandbox_policy.clone(),
            model: context.config.model.clone(),
            effort: context.config.model_reasoning_effort,
            summary: context.config.model_reasoning_summary,
            final_output_json_schema: None,
        },
    };
    conversation.submit_with_id(submission).await?;

    let mut combined = String::new();
    let mut last_agent_message: Option<String> = None;
    let mut turn_diffs: Vec<String> = Vec::new();
    let mut manual_patch_results: HashMap<String, ManualPatchOutcome> = HashMap::new();
    let mut exec_commands: HashMap<String, ExecCommandContext> = HashMap::new();

    loop {
        let event = conversation.next_event().await?;
        match event.msg {
            EventMsg::TaskComplete(task) => {
                last_agent_message = task.last_agent_message;
                break;
            }
            EventMsg::AgentMessage(msg) => {
                append_with_gap(&mut combined, &msg.message);
            }
            EventMsg::AgentMessageDelta(delta) => {
                combined.push_str(&delta.delta);
            }
            EventMsg::ApplyPatchApprovalRequest(_) => {
                auto_patch_approval(conversation, &event.id).await?;
            }
            EventMsg::ExecApprovalRequest(_) => {
                auto_exec_approval(conversation, &event.id).await?;
            }
            EventMsg::PatchApplyBegin(begin) => {
                if !manual_patch_results.contains_key(&begin.call_id) {
                    let outcome = apply_patch_locally(&begin, &context.config.cwd)
                        .with_context(|| "failed to apply patch locally")?;
                    if let Some(ref stdout) = outcome.stdout {
                        if !stdout.trim().is_empty() {
                            append_with_gap(
                                &mut combined,
                                &format!("apply_patch stdout:\n{}", stdout.trim_end()),
                            );
                        }
                    }
                    if let Some(ref stderr) = outcome.stderr {
                        if !stderr.trim().is_empty() {
                            append_with_gap(
                                &mut combined,
                                &format!("apply_patch stderr:\n{}", stderr.trim_end()),
                            );
                        }
                    }
                    manual_patch_results.insert(begin.call_id.clone(), outcome);
                }
            }
            EventMsg::PatchApplyEnd(end) => {
                if let Some(outcome) = manual_patch_results.remove(&end.call_id) {
                    if !outcome.success {
                        let mut message = outcome.stderr.clone().unwrap_or_default();
                        if message.trim().is_empty() {
                            message = end.stderr.clone();
                        }
                        let message = message.trim();
                        if message.is_empty() {
                            return Err(anyhow!("apply_patch failed"));
                        } else {
                            return Err(anyhow!(message.to_string()));
                        }
                    }
                } else if !end.success {
                    let stderr = end.stderr.trim();
                    let err_msg = if stderr.is_empty() {
                        "apply_patch failed".to_string()
                    } else {
                        stderr.to_string()
                    };
                    return Err(anyhow!(err_msg));
                }
            }
            EventMsg::ExecCommandBegin(begin) => {
                exec_commands.insert(
                    begin.call_id.clone(),
                    ExecCommandContext {
                        command: begin.command,
                        aggregated_output: String::new(),
                    },
                );
            }
            EventMsg::ExecCommandOutputDelta(delta) => {
                if let Some(ctx) = exec_commands.get_mut(&delta.call_id) {
                    if let Ok(chunk) = String::from_utf8(delta.chunk) {
                        ctx.aggregated_output.push_str(&chunk);
                    }
                }
            }
            EventMsg::ExecCommandEnd(end) => {
                if let Some(ctx) = exec_commands.remove(&end.call_id) {
                    let output = if end.aggregated_output.is_empty() {
                        ctx.aggregated_output
                    } else {
                        end.aggregated_output.clone()
                    };
                    let command = ctx.command.join(" ");
                    let exit_line = format!("exec `{}` exited {}", command, end.exit_code);
                    append_with_gap(&mut combined, &exit_line);
                    if !output.trim().is_empty() {
                        append_with_gap(
                            &mut combined,
                            &format!("exec output:\n{}", output.trim_end()),
                        );
                    }
                    if !end.stderr.trim().is_empty() {
                        append_with_gap(
                            &mut combined,
                            &format!("exec stderr:\n{}", end.stderr.trim_end()),
                        );
                    }
                    if end.exit_code != 0 {
                        return Err(anyhow!(format!(
                            "command `{}` failed with exit code {}",
                            command, end.exit_code
                        )));
                    }
                }
            }
            EventMsg::TurnDiff(diff) => {
                if !diff.unified_diff.trim().is_empty() {
                    turn_diffs.push(diff.unified_diff);
                }
            }
            EventMsg::Error(err) => {
                return Err(anyhow!(err.message));
            }
            EventMsg::StreamError(err) => {
                return Err(anyhow!(err.message));
            }
            EventMsg::ShutdownComplete => break,
            _ => {}
        }
    }

    if !turn_diffs.is_empty() {
        append_with_gap(
            &mut combined,
            &format!("turn diff:\n{}", turn_diffs.join("\n")),
        );
    }

    let summary = last_agent_message
        .filter(|msg| !msg.trim().is_empty())
        .unwrap_or_else(|| {
            let trimmed = combined.trim();
            if trimmed.is_empty() {
                "Codex returned no response".to_string()
            } else {
                trimmed.to_string()
            }
        });

    Ok(CodexResponse { summary })
}

#[cfg(feature = "embedded-codex")]
async fn auto_patch_approval(
    conversation: &Arc<codex_core::CodexConversation>,
    request_id: &str,
) -> Result<()> {
    let submission = Submission {
        id: format!("patch-approval-{}", request_id),
        op: Op::PatchApproval {
            id: request_id.to_string(),
            decision: ReviewDecision::ApprovedForSession,
        },
    };
    conversation.submit_with_id(submission).await?;
    Ok(())
}

#[cfg(feature = "embedded-codex")]
async fn auto_exec_approval(
    conversation: &Arc<codex_core::CodexConversation>,
    request_id: &str,
) -> Result<()> {
    let submission = Submission {
        id: format!("exec-approval-{}", request_id),
        op: Op::ExecApproval {
            id: request_id.to_string(),
            decision: ReviewDecision::ApprovedForSession,
        },
    };
    conversation.submit_with_id(submission).await?;
    Ok(())
}

#[cfg(feature = "embedded-codex")]
struct ManualPatchOutcome {
    success: bool,
    stdout: Option<String>,
    stderr: Option<String>,
}

#[cfg(feature = "embedded-codex")]
struct ExecCommandContext {
    command: Vec<String>,
    aggregated_output: String,
}

#[cfg(feature = "embedded-codex")]
fn apply_patch_locally(begin: &PatchApplyBeginEvent, cwd: &Path) -> Result<ManualPatchOutcome> {
    if begin.changes.is_empty() {
        return Ok(ManualPatchOutcome {
            success: true,
            stdout: Some("apply_patch: no changes".to_string()),
            stderr: None,
        });
    }

    let patch = build_patch_from_changes(&begin.changes, cwd)?;
    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();
    let guard = WorkingDirGuard::new(cwd)?;
    let result = codex_apply_patch::apply_patch(&patch, &mut stdout_buf, &mut stderr_buf);
    drop(guard);

    let stdout = String::from_utf8_lossy(&stdout_buf).to_string();
    let stderr = String::from_utf8_lossy(&stderr_buf).to_string();

    match result {
        Ok(()) => Ok(ManualPatchOutcome {
            success: true,
            stdout: if stdout.trim().is_empty() {
                None
            } else {
                Some(stdout)
            },
            stderr: if stderr.trim().is_empty() {
                None
            } else {
                Some(stderr)
            },
        }),
        Err(err) => {
            let mut message = err.to_string();
            if !stderr.trim().is_empty() {
                if !message.is_empty() {
                    message.push_str("\n");
                }
                message.push_str(stderr.trim());
            }

            Ok(ManualPatchOutcome {
                success: false,
                stdout: if stdout.trim().is_empty() {
                    None
                } else {
                    Some(stdout)
                },
                stderr: Some(message),
            })
        }
    }
}

#[cfg(feature = "embedded-codex")]
fn build_patch_from_changes(changes: &HashMap<PathBuf, FileChange>, cwd: &Path) -> Result<String> {
    let mut ordered: BTreeMap<PathBuf, &FileChange> = BTreeMap::new();
    for (path, change) in changes {
        ordered.insert(path.clone(), change);
    }

    let mut patch = String::from("*** Begin Patch\n");
    for (index, (path, change)) in ordered.into_iter().enumerate() {
        if index > 0 {
            if !patch.ends_with('\n') {
                patch.push('\n');
            }
        }
        let rel = match path.strip_prefix(cwd) {
            Ok(rel) => rel,
            Err(_) => path.as_path(),
        };
        let rel_display = rel.to_string_lossy().replace('\\', "/");
        match change {
            FileChange::Add { content } => {
                writeln!(patch, "*** Add File: {}", rel_display)?;
                for line in content.split_inclusive('\n') {
                    if let Some(stripped) = line.strip_suffix('\n') {
                        writeln!(patch, "+{}", stripped)?;
                    } else {
                        writeln!(patch, "+{}", line)?;
                    }
                }
            }
            FileChange::Delete { .. } => {
                writeln!(patch, "*** Delete File: {}", rel_display)?;
            }
            FileChange::Update {
                unified_diff,
                move_path,
            } => {
                writeln!(patch, "*** Update File: {}", rel_display)?;
                if let Some(move_path) = move_path {
                    let rel_move = match move_path.strip_prefix(cwd) {
                        Ok(rel) => rel,
                        Err(_) => move_path.as_path(),
                    };
                    writeln!(
                        patch,
                        "*** Move to: {}",
                        rel_move.to_string_lossy().replace('\\', "/")
                    )?;
                }
                patch.push_str(unified_diff);
                if !unified_diff.ends_with('\n') {
                    patch.push('\n');
                }
            }
        }
    }
    if !patch.ends_with('\n') {
        patch.push('\n');
    }
    patch.push_str("*** End Patch\n");
    Ok(patch)
}

#[cfg(feature = "embedded-codex")]
struct WorkingDirGuard {
    original: PathBuf,
    changed: bool,
}

#[cfg(feature = "embedded-codex")]
impl WorkingDirGuard {
    fn new(target: &Path) -> Result<Self> {
        let original = env::current_dir()?;
        if original != target {
            env::set_current_dir(target)?;
            Ok(Self {
                original,
                changed: true,
            })
        } else {
            Ok(Self {
                original,
                changed: false,
            })
        }
    }
}

#[cfg(feature = "embedded-codex")]
impl Drop for WorkingDirGuard {
    fn drop(&mut self) {
        if self.changed {
            if let Err(err) = env::set_current_dir(&self.original) {
                eprintln!("warning: failed to restore current dir: {err}");
            }
        }
    }
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
    auth_manager: Arc<AuthManager>,
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
        Err(err) if err.kind() == ErrorKind::NotFound => {
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
        auth_manager,
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
    let auth_manager = AuthManager::from_auth_for_testing(CodexAuth::from_api_key("dummy"));
    let config = Arc::new(config);

    let fixture_path = codex_home.path().join("embedded_fixture.sse");
    fs::write(&fixture_path, FIXTURE_SSE).context("writing Codex SSE fixture")?;

    let sse_guard = EnvVarGuard::set("CODEX_RS_SSE_FIXTURE", &fixture_path);
    let api_guard = EnvVarGuard::set("OPENAI_API_KEY", "dummy");
    let base_guard = EnvVarGuard::set("OPENAI_BASE_URL", "http://unused.local");
    let home_guard = EnvVarGuard::set("CODEX_HOME", codex_home.path());

    Ok(RuntimeContext {
        config,
        auth_manager,
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

    #[test]
    fn embedded_reports_available() {
        let _fixture_flag = EnvVarGuard::set("RUNMATFUNC_USE_CODEX_FIXTURE", "1");
        let temp_home = TempDir::new().expect("temp dir");
        let _home_guard = EnvVarGuard::set("CODEX_HOME", temp_home.path());
        assert!(super::is_available());
    }

    #[test]
    fn fixture_applies_patch() -> anyhow::Result<()> {
        let _fixture_flag = EnvVarGuard::set("RUNMATFUNC_USE_CODEX_FIXTURE", "1");
        let temp_home = TempDir::new()?;
        let _home_guard = EnvVarGuard::set("CODEX_HOME", temp_home.path());

        let fixture_path = PathBuf::from("tools/runmatfunc/fixture.txt");
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
        let response = client.run(&request)?;
        assert!(fixture_path.exists());
        let contents = fs::read_to_string(&fixture_path)?;
        assert_eq!(contents, "fixture contents\n");
        fs::remove_file(fixture_path)?;
        assert!(response.summary.contains("Applied"));
        Ok(())
    }
}
