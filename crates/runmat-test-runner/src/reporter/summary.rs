use runmat_test::result::{RunResult, TerminalDisposition};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReportSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub incomplete: usize,
}

impl ReportSummary {
    pub fn from_result(result: &RunResult) -> Self {
        let mut summary = Self {
            total: result.tests.len(),
            ..Self::default()
        };
        for test in &result.tests {
            if test.state.disposition == TerminalDisposition::Passed {
                summary.passed += 1;
            } else {
                summary.failed += 1;
            }
            summary.incomplete += usize::from(test.state.incomplete);
        }
        summary
    }
}
