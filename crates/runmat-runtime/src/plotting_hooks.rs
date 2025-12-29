//! Lightweight wrappers that expose plotting-specific helpers without requiring
//! downstream crates to depend directly on the plotting feature flag.

/// Reset the per-thread "figures touched" set. No-op when plotting is disabled.
pub fn reset_recent_figures() {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::reset_recent_figures();
    }
}

/// Drain the per-thread "figures touched" set, returning the raw handles.
pub fn take_recent_figures() -> Vec<u32> {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::take_recent_figures()
            .into_iter()
            .map(|handle| handle.as_u32())
            .collect()
    }
    #[cfg(not(feature = "plot-core"))]
    {
        Vec::new()
    }
}
