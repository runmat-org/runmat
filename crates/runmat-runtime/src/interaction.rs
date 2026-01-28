use once_cell::sync::OnceCell;
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
        return match response.map_err(|err| build_runtime_error(err).build())? {
            InteractionResponse::Line(value) => Ok(value),
            InteractionResponse::KeyPress => Err(build_runtime_error(
                "queued keypress response used for line request",
            )
            .build()),
        };
    }

    if let Some(handler) = async_handler_slot().read().ok().and_then(|slot| slot.clone()) {
        let owned = InteractionPromptOwned {
            prompt: prompt.to_string(),
            kind: InteractionKind::Line { echo },
        };
        let value = handler(owned)
            .await
            .map_err(|err| build_runtime_error(err).build())?;
        return match value {
            InteractionResponse::Line(line) => Ok(line),
            InteractionResponse::KeyPress => Err(build_runtime_error(
                "interaction async handler returned keypress for line request",
            )
            .build()),
        };
    }

    default_read_line(prompt, echo).map_err(|err| build_runtime_error(err).build())
}

pub async fn wait_for_key_async(prompt: &str) -> Result<(), RuntimeError> {
    if let Some(response) = QUEUED_RESPONSE.with(|slot| slot.borrow_mut().take()) {
        return match response.map_err(|err| build_runtime_error(err).build())? {
            InteractionResponse::Line(_) => Err(build_runtime_error(
                "queued line response used for keypress request",
            )
            .build()),
            InteractionResponse::KeyPress => Ok(()),
        };
    }

    if let Some(handler) = async_handler_slot().read().ok().and_then(|slot| slot.clone()) {
        let owned = InteractionPromptOwned {
            prompt: prompt.to_string(),
            kind: InteractionKind::KeyPress,
        };
        let value = handler(owned)
            .await
            .map_err(|err| build_runtime_error(err).build())?;
        return match value {
            InteractionResponse::Line(_) => Err(build_runtime_error(
                "interaction async handler returned line value for keypress request",
            )
            .build()),
            InteractionResponse::KeyPress => Ok(()),
        };
    }

    default_wait_for_key(prompt).map_err(|err| build_runtime_error(err).build())
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
