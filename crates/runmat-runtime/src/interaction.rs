use crate::console::{self, ConsoleStream};
use once_cell::sync::OnceCell;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::io::IsTerminal;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Read, Write};
#[cfg(all(feature = "interaction-test-hooks", not(target_arch = "wasm32")))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy)]
pub enum InteractionKind {
    Line { echo: bool },
    KeyPress,
    /// Internal suspension used by the wasm runtime while waiting for a WebGPU map/readback
    /// completion. The `PendingInteraction.prompt` carries a human-readable label.
    GpuMapRead,
}

pub struct InteractionPrompt<'a> {
    pub prompt: &'a str,
    pub kind: InteractionKind,
}

#[derive(Clone)]
pub enum InteractionResponse {
    Line(String),
    KeyPress,
    /// Resume token for `InteractionKind::GpuMapRead`.
    GpuMapReady,
}

#[derive(Clone)]
pub struct PendingInteraction {
    pub prompt: String,
    pub kind: InteractionKind,
}

#[derive(Clone)]
pub enum InteractionDecision {
    Respond(Result<InteractionResponse, String>),
    Pending,
}

type InteractionHandler =
    dyn for<'a> Fn(InteractionPrompt<'a>) -> InteractionDecision + Send + Sync;

static HANDLER: OnceCell<RwLock<Option<Arc<InteractionHandler>>>> = OnceCell::new();
runmat_thread_local! {
    static LAST_PENDING: RefCell<Option<PendingInteraction>> = const { RefCell::new(None) };
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

pub const PENDING_INTERACTION_ERR: &str = "__RUNMAT_PENDING_INTERACTION__";

fn handler_slot() -> &'static RwLock<Option<Arc<InteractionHandler>>> {
    HANDLER.get_or_init(|| RwLock::new(None))
}

pub struct HandlerGuard {
    previous: Option<Arc<InteractionHandler>>,
}

impl HandlerGuard {
    pub fn install(handler: Option<Arc<InteractionHandler>>) -> Self {
        let mut slot = handler_slot()
            .write()
            .unwrap_or_else(|_| panic!("interaction handler lock poisoned"));
        let previous = std::mem::replace(&mut *slot, handler);
        Self { previous }
    }
}

impl Drop for HandlerGuard {
    fn drop(&mut self) {
        let mut slot = handler_slot()
            .write()
            .unwrap_or_else(|_| panic!("interaction handler lock poisoned"));
        *slot = self.previous.take();
    }
}

pub fn replace_handler(handler: Option<Arc<InteractionHandler>>) -> HandlerGuard {
    HandlerGuard::install(handler)
}

pub fn request_line(prompt: &str, echo: bool) -> Result<String, String> {
    if let Some(response) = QUEUED_RESPONSE.with(|slot| slot.borrow_mut().take()) {
        return match response? {
            InteractionResponse::Line(value) => Ok(value),
            InteractionResponse::KeyPress => {
                Err("queued keypress response used for line request".to_string())
            }
            InteractionResponse::GpuMapReady => {
                Err("queued gpu map response used for line request".to_string())
            }
        };
    }
    if let Some(handler) = handler_slot().read().ok().and_then(|slot| slot.clone()) {
        match handler(InteractionPrompt {
            prompt,
            kind: InteractionKind::Line { echo },
        }) {
            InteractionDecision::Respond(result) => match result? {
                InteractionResponse::Line(value) => Ok(value),
                InteractionResponse::KeyPress => {
                    Err("interaction handler returned keypress for line request".to_string())
                }
                InteractionResponse::GpuMapReady => Err(
                    "interaction handler returned gpu map response for line request".to_string(),
                ),
            },
            InteractionDecision::Pending => {
                suspend(InteractionKind::Line { echo }, prompt)?;
                unreachable!("suspend always returns Err")
            }
        }
    } else {
        default_read_line(prompt, echo)
    }
}

pub fn wait_for_key(prompt: &str) -> Result<(), String> {
    if let Some(response) = QUEUED_RESPONSE.with(|slot| slot.borrow_mut().take()) {
        return match response? {
            InteractionResponse::Line(_) => {
                Err("queued line response used for keypress request".to_string())
            }
            InteractionResponse::KeyPress => Ok(()),
            InteractionResponse::GpuMapReady => {
                Err("queued gpu map response used for keypress request".to_string())
            }
        };
    }
    if let Some(handler) = handler_slot().read().ok().and_then(|slot| slot.clone()) {
        match handler(InteractionPrompt {
            prompt,
            kind: InteractionKind::KeyPress,
        }) {
            InteractionDecision::Respond(result) => match result? {
                InteractionResponse::Line(_) => {
                    Err("interaction handler returned line value for keypress request".to_string())
                }
                InteractionResponse::KeyPress => Ok(()),
                InteractionResponse::GpuMapReady => Err(
                    "interaction handler returned gpu map response for keypress request"
                        .to_string(),
                ),
            },
            InteractionDecision::Pending => {
                suspend(InteractionKind::KeyPress, prompt)?;
                unreachable!("suspend always returns Err")
            }
        }
    } else {
        default_wait_for_key(prompt)
    }
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

pub fn take_pending_interaction() -> Option<PendingInteraction> {
    LAST_PENDING.with(|slot| slot.borrow_mut().take())
}

pub fn push_queued_response(response: Result<InteractionResponse, String>) {
    QUEUED_RESPONSE.with(|slot| {
        *slot.borrow_mut() = Some(response);
    });
}

/// Suspend the current execution, recording the pending interaction for the host to resume.
///
/// This is the central suspension primitive used by the interpreter/host bridge. Builtins like
/// `input()` use it, and internal subsystems (e.g. wasm WebGPU readbacks) can also use it to
/// avoid blocking the browser event loop.
pub fn suspend(kind: InteractionKind, prompt: &str) -> Result<(), String> {
    console::record_console_output(ConsoleStream::Stdout, prompt);
    LAST_PENDING.with(|slot| {
        *slot.borrow_mut() = Some(PendingInteraction {
            prompt: prompt.to_string(),
            kind,
        });
    });
    Err(PENDING_INTERACTION_ERR.to_string())
}
