use once_cell::sync::OnceCell;
use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;

use crate::{build_runtime_error, RuntimeError};
#[cfg(not(target_arch = "wasm32"))]
use std::io::IsTerminal;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Read, Write};
#[cfg(all(feature = "interaction-test-hooks", not(target_arch = "wasm32")))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

pub use runmat_async::InteractionKind;

#[derive(Clone)]
pub struct InteractionPromptOwned {
    pub prompt: String,
    pub kind: InteractionKind,
}

#[derive(Clone)]
pub enum InteractionResponse {
    Line(String),
    KeyPress,
}

pub type AsyncInteractionFuture =
    Pin<Box<dyn Future<Output = Result<InteractionResponse, String>> + 'static>>;

pub type AsyncInteractionHandler =
    dyn Fn(InteractionPromptOwned) -> AsyncInteractionFuture + Send + Sync;

static ASYNC_HANDLER: OnceCell<RwLock<Option<Arc<AsyncInteractionHandler>>>> = OnceCell::new();
runmat_thread_local! {
    static QUEUED_RESPONSE: RefCell<Option<Result<InteractionResponse, String>>> =
        const { RefCell::new(None) };
}

#[cfg(all(feature = "interaction-test-hooks", not(target_arch = "wasm32")))]
static FORCE_INTERACTIVE_STDIN: AtomicBool = AtomicBool::new(false);

#[cfg(all(feature = "interaction-test-hooks", not(target_arch = "wasm32")))]
pub fn force_interactive_stdin_for_tests(enable: bool) {
    FORCE_INTERACTIVE_STDIN.store(enable, Ordering::Relaxed);
}

#[cfg(all(not(feature = "interaction-test-hooks"), not(target_arch = "wasm32")))]
#[inline]
fn force_interactive_stdin() -> bool {
    false
}

#[cfg(all(feature = "interaction-test-hooks", not(target_arch = "wasm32")))]
#[inline]
fn force_interactive_stdin() -> bool {
    FORCE_INTERACTIVE_STDIN.load(Ordering::Relaxed)
}

fn async_handler_slot() -> &'static RwLock<Option<Arc<AsyncInteractionHandler>>> {
    ASYNC_HANDLER.get_or_init(|| RwLock::new(None))
}

fn interaction_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(identifier.to_string())
        .build()
}

pub struct AsyncHandlerGuard {
    previous: Option<Arc<AsyncInteractionHandler>>,
}

impl Drop for AsyncHandlerGuard {
    fn drop(&mut self) {
        let mut slot = async_handler_slot()
            .write()
            .unwrap_or_else(|_| panic!("interaction async handler lock poisoned"));
        *slot = self.previous.take();
    }
}

pub fn replace_async_handler(handler: Option<Arc<AsyncInteractionHandler>>) -> AsyncHandlerGuard {
    let mut slot = async_handler_slot()
        .write()
        .unwrap_or_else(|_| panic!("interaction async handler lock poisoned"));
    let previous = std::mem::replace(&mut *slot, handler);
    AsyncHandlerGuard { previous }
}

pub async fn request_line_async(prompt: &str, echo: bool) -> Result<String, RuntimeError> {
    if let Some(response) = QUEUED_RESPONSE.with(|slot| slot.borrow_mut().take()) {
        return match response
            .map_err(|err| interaction_error("RunMat:interaction:QueuedResponseError", err))?
        {
            InteractionResponse::Line(value) => Ok(value),
            InteractionResponse::KeyPress => Err(interaction_error(
                "RunMat:interaction:UnexpectedQueuedKeypress",
                "queued keypress response used for line request",
            )),
        };
    }

    if let Some(handler) = async_handler_slot()
        .read()
        .ok()
        .and_then(|slot| slot.clone())
    {
        let owned = InteractionPromptOwned {
            prompt: prompt.to_string(),
            kind: InteractionKind::Line { echo },
        };
        let value = handler(owned)
            .await
            .map_err(|err| interaction_error("RunMat:interaction:AsyncHandlerError", err))?;
        return match value {
            InteractionResponse::Line(line) => Ok(line),
            InteractionResponse::KeyPress => Err(interaction_error(
                "RunMat:interaction:UnexpectedAsyncKeypress",
                "interaction async handler returned keypress for line request",
            )),
        };
    }

    default_read_line(prompt, echo)
        .map_err(|err| interaction_error("RunMat:interaction:ReadLineFailed", err))
}

pub async fn wait_for_key_async(prompt: &str) -> Result<(), RuntimeError> {
    if let Some(response) = QUEUED_RESPONSE.with(|slot| slot.borrow_mut().take()) {
        return match response
            .map_err(|err| interaction_error("RunMat:interaction:QueuedResponseError", err))?
        {
            InteractionResponse::Line(_) => Err(interaction_error(
                "RunMat:interaction:UnexpectedQueuedLine",
                "queued line response used for keypress request",
            )),
            InteractionResponse::KeyPress => Ok(()),
        };
    }

    if let Some(handler) = async_handler_slot()
        .read()
        .ok()
        .and_then(|slot| slot.clone())
    {
        let owned = InteractionPromptOwned {
            prompt: prompt.to_string(),
            kind: InteractionKind::KeyPress,
        };
        let value = handler(owned)
            .await
            .map_err(|err| interaction_error("RunMat:interaction:AsyncHandlerError", err))?;
        return match value {
            InteractionResponse::Line(_) => Err(interaction_error(
                "RunMat:interaction:UnexpectedAsyncLine",
                "interaction async handler returned line value for keypress request",
            )),
            InteractionResponse::KeyPress => Ok(()),
        };
    }

    default_wait_for_key(prompt)
        .map_err(|err| interaction_error("RunMat:interaction:WaitForKeyFailed", err))
}

pub fn default_read_line(prompt: &str, echo: bool) -> Result<String, String> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (prompt, echo);
        Err("stdin input is not available on wasm targets".to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if !prompt.is_empty() {
            let mut stdout = io::stdout();
            write!(stdout, "{prompt}")
                .map_err(|err| format!("input: failed to write prompt ({err})"))?;
            stdout
                .flush()
                .map_err(|err| format!("input: failed to flush stdout ({err})"))?;
        }
        let mut line = String::new();
        let stdin = io::stdin();
        stdin
            .read_line(&mut line)
            .map_err(|err| format!("input: failed to read from stdin ({err})"))?;
        if !echo {
            // When echo is disabled we still read the full line; no additional handling needed.
        }
        Ok(line.trim_end_matches(&['\r', '\n'][..]).to_string())
    }
}

pub fn default_wait_for_key(prompt: &str) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = prompt;
        Err("keypress input is not available on wasm targets".to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if !prompt.is_empty() {
            let mut stdout = io::stdout();
            write!(stdout, "{prompt}")
                .map_err(|err| format!("pause: failed to write prompt ({err})"))?;
            stdout
                .flush()
                .map_err(|err| format!("pause: failed to flush stdout ({err})"))?;
        }
        let stdin = io::stdin();
        if !stdin.is_terminal() && !force_interactive_stdin() {
            return Ok(());
        }
        let mut handle = stdin.lock();
        let mut buf = [0u8; 1];
        handle
            .read(&mut buf)
            .map_err(|err| format!("pause: failed to read from stdin ({err})"))?;
        Ok(())
    }
}

pub fn push_queued_response(response: Result<InteractionResponse, String>) {
    QUEUED_RESPONSE.with(|slot| {
        *slot.borrow_mut() = Some(response);
    });
}

// NOTE: The old suspend/resume control flow has been removed.

// ---------------------------------------------------------------------------
// Eval hook – lets runmat-core install a stateless expression evaluator so
// that `input()` can parse numeric responses through the full MATLAB pipeline
// instead of falling back to `str2double` (which cannot handle matrix literals,
// named constants like `pi`, arithmetic, etc.).
// ---------------------------------------------------------------------------

/// Future returned by the eval hook.
pub type EvalHookFuture = Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'static>>;

/// Function signature for the eval hook.
pub type EvalHookFn = dyn Fn(String) -> EvalHookFuture + Send + Sync;

static EVAL_HOOK: OnceCell<RwLock<Option<Arc<EvalHookFn>>>> = OnceCell::new();

fn eval_hook_slot() -> &'static RwLock<Option<Arc<EvalHookFn>>> {
    EVAL_HOOK.get_or_init(|| RwLock::new(None))
}

/// RAII guard that restores the previous eval hook on drop.
pub struct EvalHookGuard {
    previous: Option<Arc<EvalHookFn>>,
}

impl Drop for EvalHookGuard {
    fn drop(&mut self) {
        let mut slot = eval_hook_slot()
            .write()
            .unwrap_or_else(|_| panic!("interaction eval hook lock poisoned"));
        *slot = self.previous.take();
    }
}

/// Replace the global eval hook for the duration of the returned guard's
/// lifetime. Mirrors the pattern used by `replace_async_handler`.
pub fn replace_eval_hook(hook: Option<Arc<EvalHookFn>>) -> EvalHookGuard {
    let mut slot = eval_hook_slot()
        .write()
        .unwrap_or_else(|_| panic!("interaction eval hook lock poisoned"));
    let previous = std::mem::replace(&mut *slot, hook);
    EvalHookGuard { previous }
}

/// Return the currently installed eval hook, if any.
pub fn current_eval_hook() -> Option<Arc<EvalHookFn>> {
    eval_hook_slot().read().ok().and_then(|slot| slot.clone())
}
