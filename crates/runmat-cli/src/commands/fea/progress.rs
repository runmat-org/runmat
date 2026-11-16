//! Live run progress for local CLI runs.
//!
//! With `RUNMAT_FEA_PROGRESS=json`, a local `runmat run` of a `.fea` study writes each progress event to
//! standard error as one JSON object per line, prefixed with the milliseconds since the run
//! started (`{"t_ms":1234,"event":{...}}`). Standard output keeps only the run result.
use std::io::Write as _;
use std::sync::Arc;

pub(super) const PROGRESS_ENV: &str = "RUNMAT_FEA_PROGRESS";

pub(super) fn observer_from_env() -> Option<runmat_execution_runner_native::LocalFeaProgressObserver>
{
    if std::env::var(PROGRESS_ENV).ok()? != "json" {
        return None;
    }
    let started = std::time::Instant::now();
    Some(Arc::new(move |event| {
        let line = serde_json::json!({
            "t_ms": started.elapsed().as_millis() as u64,
            "event": event,
        });
        let _ = writeln!(std::io::stderr().lock(), "{line}");
    }))
}
