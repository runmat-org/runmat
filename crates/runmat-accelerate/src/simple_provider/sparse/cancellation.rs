use runmat_accelerate_api::ProviderCancellationToken;

pub(super) const MAXIMUM_WORK_UNITS_BETWEEN_CHECKS: u64 = 1_024;

pub(super) enum Controlled<T> {
    Completed(T),
    Cancelled,
}

pub(super) struct CancellationMonitor<'a> {
    token: &'a ProviderCancellationToken,
    work_since_check: u64,
    checks: u64,
}

impl<'a> CancellationMonitor<'a> {
    pub(super) fn new(token: &'a ProviderCancellationToken) -> Self {
        Self {
            token,
            work_since_check: 0,
            checks: 0,
        }
    }

    pub(super) fn check_now(&mut self) -> bool {
        self.work_since_check = 0;
        self.checks = self.checks.saturating_add(1);
        self.token.is_cancelled()
    }

    pub(super) fn complete_work_unit(&mut self) -> bool {
        self.work_since_check = self.work_since_check.saturating_add(1);
        self.work_since_check >= MAXIMUM_WORK_UNITS_BETWEEN_CHECKS && self.check_now()
    }

    pub(super) const fn checks(&self) -> u64 {
        self.checks
    }
}
