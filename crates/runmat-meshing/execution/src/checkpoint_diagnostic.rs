//! Internal first-miss telemetry. Diagnostic collection must not perform stack unwinding.
#[track_caller]
pub(crate) fn report_first_miss(prior_maximum: u64, latency: u64, maximum: u64) {
    if latency <= maximum || prior_maximum > maximum {
        return;
    }
    let location = std::panic::Location::caller();
    tracing::warn!(target: "runmat.meshing.checkpoint",
        checkpoint_latency_ms = latency,
        maximum_checkpoint_latency_ms = maximum,
        checkpoint_observer_file = location.file(),
        checkpoint_observer_line = location.line(),
        "meshing checkpoint responsiveness exceeded");
}
