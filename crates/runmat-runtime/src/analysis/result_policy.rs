use super::contracts::{QualityGate, RunStatus};

pub(super) fn classify_run_status(
    result_quality: QualityGate,
    numerically_acceptable: bool,
) -> RunStatus {
    if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else if numerically_acceptable {
        RunStatus::Completed
    } else {
        RunStatus::Degraded
    }
}
