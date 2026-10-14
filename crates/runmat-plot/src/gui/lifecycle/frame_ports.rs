//! Exact live-window frame-port registration.
//!
//! This is a GUI routing table, not source/session authorization. A graphics
//! service must capture and authenticate its selected scene separately before
//! using a port returned for a numeric figure handle.

use crate::gui::frame_command::FramePort;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct RegisteredPort {
    generation: Arc<()>,
    port: Option<FramePort>,
}

static PORTS: Lazy<Mutex<HashMap<u32, RegisteredPort>>> = Lazy::new(|| Mutex::new(HashMap::new()));

pub struct FramePortRegistration {
    handle: u32,
    generation: Arc<()>,
}

/// Reserve a numeric routing slot before the host creates an OS window.
/// The reservation is invisible to frame producers until activation and is
/// retired on every failed/abandoned window-open path.
pub fn reserve(handle: u32) -> Result<FramePortRegistration, String> {
    if handle == 0 {
        return Err("a zero figure handle cannot own a window frame port".into());
    }
    let generation = Arc::new(());
    let mut ports = PORTS
        .lock()
        .map_err(|_| "plot window frame registry is poisoned".to_string())?;
    if ports.contains_key(&handle) {
        return Err(format!("figure {handle} already owns a live frame port"));
    }
    ports.insert(
        handle,
        RegisteredPort {
            generation: Arc::clone(&generation),
            port: None,
        },
    );
    Ok(FramePortRegistration { handle, generation })
}

impl FramePortRegistration {
    pub fn activate(&mut self, port: FramePort) -> Result<(), String> {
        let mut ports = PORTS
            .lock()
            .map_err(|_| "plot window frame registry is poisoned".to_string())?;
        let registered = ports
            .get_mut(&self.handle)
            .ok_or_else(|| "plot window frame reservation retired".to_string())?;
        if !Arc::ptr_eq(&registered.generation, &self.generation) || registered.port.is_some() {
            return Err("plot window frame reservation changed".into());
        }
        registered.port = Some(port);
        Ok(())
    }
}

/// Existing window managers can still register their already-created port in
/// one call. The host's two-phase path reserves before creating a window.
pub fn register(handle: u32, port: FramePort) -> Result<FramePortRegistration, String> {
    let mut registration = reserve(handle)?;
    registration.activate(port)?;
    Ok(registration)
}

/// GUI routing only. This lookup does not authenticate a runtime session or
/// capture a revision-pinned scene.
pub fn for_handle(handle: u32) -> Option<FramePort> {
    PORTS
        .lock()
        .ok()?
        .get(&handle)
        .and_then(|registered| registered.port.clone())
}

impl Drop for FramePortRegistration {
    fn drop(&mut self) {
        if let Ok(mut ports) = PORTS.lock() {
            if ports
                .get(&self.handle)
                .is_some_and(|registered| Arc::ptr_eq(&registered.generation, &self.generation))
            {
                ports.remove(&self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_handle_is_invisible_and_retired_without_window_creation() {
        let handle = 0x7fff_fffe;
        let first = reserve(handle).expect("reserve one routing identity");
        assert!(for_handle(handle).is_none(), "unopened window has no port");
        assert!(
            reserve(handle).is_err(),
            "duplicate open is denied pre-effect"
        );
        drop(first);
        let second = reserve(handle).expect("failed open released exact generation");
        assert!(for_handle(handle).is_none());
        drop(second);
    }
}
