use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use runmat_builtins::TEMPNAME_ERROR_UNABLE_TO_GENERATE;

use super::super::error;

const IDENTITY: &str = "tempname";
const MAX_ATTEMPTS: usize = 64;
static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) async fn unused_path(base: &Path) -> crate::BuiltinResult<PathBuf> {
    for _ in 0..MAX_ATTEMPTS {
        let candidate = base.join(token());
        if runmat_filesystem::metadata_async(&candidate).await.is_err() {
            return Ok(candidate);
        }
    }
    Err(error::builtin(IDENTITY, &TEMPNAME_ERROR_UNABLE_TO_GENERATE))
}

fn token() -> String {
    let now = runmat_time::system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "tp{:016x}{:08x}{:08x}{:016x}",
        now.as_secs(),
        now.subsec_nanos(),
        process_id(),
        counter
    )
}

fn process_id() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        u64::from(std::process::id())
    }
}
