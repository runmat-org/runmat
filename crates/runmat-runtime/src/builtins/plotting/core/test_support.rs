use super::state;

#[doc(hidden)]
pub use state::PlotTestLockGuard;

#[doc(hidden)]
pub fn lock_plot_test_context() -> PlotTestLockGuard {
    state::lock_plot_test_registry()
}

#[cfg(test)]
pub(crate) fn ensure_plot_test_env() {
    state::disable_rendering_for_tests();
}

#[cfg(test)]
pub(crate) fn lock_plot_registry() -> state::PlotTestLockGuard {
    state::lock_plot_test_registry()
}
