//! Actual retirement is independent from the original operation outcome.
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, DriverAction, DriverCommand};

use super::BrowserExecutionService;

impl BrowserExecutionService {
    pub(super) fn apply_browser_report(
        &self,
        mut report: BackendReport,
        retirement_certified: bool,
    ) -> Vec<DriverAction> {
        if let Err(error) = self.advance_driver_now() {
            self.deadline_timer_failed(error);
        }
        let mut state = self.state.borrow_mut();
        let Some(attempt) = state.driver.snapshot_ref().attempts.get(&report.attempt_id) else {
            return Vec::new();
        };
        if !state.driver.accepts_backend_report(&report) {
            return state
                .driver
                .handle(DriverCommand::BackendReport(report))
                .unwrap_or_default();
        }
        let proof = BackendReport::for_request(&attempt.request, AttemptReport::Retired);
        if retirement_certified
            && attempt.cancellation_requested_at.is_some()
            && matches!(report.report, AttemptReport::Succeeded { .. })
        {
            if let AttemptReport::Succeeded { result } = &report.report {
                report.unsuccessful_resources = result.resources().cloned().map(Box::new);
            }
            report.report = AttemptReport::Retired;
        }
        let mut actions = state
            .driver
            .handle(DriverCommand::BackendReport(report))
            .unwrap_or_default();
        if retirement_certified
            && state
                .driver
                .snapshot_ref()
                .attempts
                .get(&proof.attempt_id)
                .is_some_and(|attempt| {
                    attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none())
                })
        {
            actions.extend(
                state
                    .driver
                    .handle(DriverCommand::BackendReport(proof))
                    .unwrap_or_default(),
            );
        }
        actions
    }
}
