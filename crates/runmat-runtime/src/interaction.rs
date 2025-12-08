use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use std::io::IsTerminal;
use std::io::{self, Read, Write};
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy)]
pub enum InteractionKind {
    Line { echo: bool },
    KeyPress,
}

pub struct InteractionPrompt<'a> {
    pub prompt: &'a str,
    pub kind: InteractionKind,
}

pub enum InteractionResponse {
    Line(String),
    KeyPress,
}

type InteractionHandler =
    dyn for<'a> Fn(InteractionPrompt<'a>) -> Result<InteractionResponse, String> + Send + Sync;

static HANDLER: OnceCell<RwLock<Option<Arc<InteractionHandler>>>> = OnceCell::new();

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
    if let Some(handler) = handler_slot().read().ok().and_then(|slot| slot.clone()) {
        match handler(InteractionPrompt {
            prompt,
            kind: InteractionKind::Line { echo },
        })? {
            InteractionResponse::Line(value) => Ok(value),
            InteractionResponse::KeyPress => {
                Err("interaction handler returned keypress for line request".to_string())
            }
        }
    } else {
        default_read_line(prompt, echo)
    }
}

pub fn wait_for_key(prompt: &str) -> Result<(), String> {
    if let Some(handler) = handler_slot().read().ok().and_then(|slot| slot.clone()) {
        match handler(InteractionPrompt {
            prompt,
            kind: InteractionKind::KeyPress,
        })? {
            InteractionResponse::Line(_) => {
                Err("interaction handler returned line value for keypress request".to_string())
            }
            InteractionResponse::KeyPress => Ok(()),
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
        if !stdin.is_terminal() {
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
