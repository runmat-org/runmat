//! Shared compensated scalar accumulation; callers own scaling and finite admission.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    pub(crate) fn total(self) -> f64 {
        self.sum + self.correction
    }
}
