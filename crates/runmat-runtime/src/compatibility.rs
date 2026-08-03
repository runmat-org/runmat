//! Active language compatibility policy for runtime builtin dispatch.

use runmat_thread_local::runmat_thread_local;
use std::cell::Cell;

runmat_thread_local! {
    static RUNMAT_EXTENSIONS_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Returns whether the current execution may use deliberately classified
/// RunMat-only builtin forms.
pub fn runmat_extensions_enabled() -> bool {
    RUNMAT_EXTENSIONS_ENABLED.with(Cell::get)
}

/// Set the extension policy for subsequent builtin dispatch on this thread.
pub fn set_runmat_extensions_enabled(enabled: bool) {
    RUNMAT_EXTENSIONS_ENABLED.with(|slot| slot.set(enabled));
}

/// Temporarily replace the extension policy and restore it on drop.
pub fn push_runmat_extensions_enabled(enabled: bool) -> RunMatExtensionsGuard {
    let previous = runmat_extensions_enabled();
    set_runmat_extensions_enabled(enabled);
    RunMatExtensionsGuard { previous }
}

#[must_use]
pub struct RunMatExtensionsGuard {
    previous: bool,
}

impl Drop for RunMatExtensionsGuard {
    fn drop(&mut self) {
        set_runmat_extensions_enabled(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_extension_policy_restores_previous_state() {
        let original = runmat_extensions_enabled();
        {
            let _guard = push_runmat_extensions_enabled(!original);
            assert_eq!(runmat_extensions_enabled(), !original);
        }
        assert_eq!(runmat_extensions_enabled(), original);
    }
}
