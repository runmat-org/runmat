//! Placeholder for high-level operations that mutate application state.

use anyhow::Result;

use super::state::AppContext;

impl AppContext {
    pub fn not_implemented(&mut self) -> Result<()> {
        println!("[runmatfunc] action not implemented yet");
        Ok(())
    }
}
