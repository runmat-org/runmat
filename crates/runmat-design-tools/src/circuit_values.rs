//! A circuit's values computed by `.m` functions (PCB-38), for the design tools.
//!
//! The circuit compiler lists the calls it can't make (`pending_values`). With a value host set
//! for the thread (the CLI and MCP server set one with `m_values::call_with_m_values`), each
//! listed call runs in the host's runner and the compile sees the results. Without a host the
//! options are empty and the compile reports its values as waiting for the runtime, as before.

use crate::Files;
use runmat_circuit::runtime::{values, CompileOptions};
use runmat_cad_wasm::runtime::values_host;

/// How many rounds of calls a compile may need: one, unless a call's result changes which
/// calls the design makes (a variant chosen by a computed value).
const ROUNDS: usize = 4;

/// The compile options for the circuit (or layout) at `path`: the results of its `.m` calls,
/// run through the thread's value host.
pub fn circuit_options(path: &str, text: &str, files: &Files) -> CompileOptions {
    let mut results = values::Values::new();
    for _ in 0..ROUNDS {
        let pending = values::pending_values(path, text, files, &results);
        let fresh: Vec<_> = pending.into_iter().filter(|call| !results.contains_key(&call.key)).collect();
        if fresh.is_empty() {
            break;
        }
        match values_host::run_calls(&fresh) {
            Some(ran) => results.extend(ran),
            None => break,
        }
    }
    CompileOptions {
        values: results,
        ..CompileOptions::default()
    }
}

/// Runs `f` with `.m` values run natively when the tools are built with `m-values`
/// ([`crate::inspect::m_values::with_m_values`]); otherwise runs `f` as it is, and computed
/// values wait for a runtime.
pub fn with_native_values<R>(files: &Files, f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "m-values")]
    {
        crate::inspect::m_values::with_m_values(files, f)
    }
    #[cfg(not(feature = "m-values"))]
    {
        let _ = files;
        f()
    }
}
