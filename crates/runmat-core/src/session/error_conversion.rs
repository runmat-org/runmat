use runmat_runtime::RuntimeError;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn into_anyhow(error: RuntimeError) -> anyhow::Error {
    anyhow::Error::new(error)
}

#[cfg(target_arch = "wasm32")]
pub(super) fn into_anyhow(error: RuntimeError) -> anyhow::Error {
    // Browser-local sources need not be Send + Sync. The session's public
    // anyhow boundary retains the diagnostic without imposing thread safety
    // on the runtime's local error source graph.
    anyhow::anyhow!(error.to_string())
}
