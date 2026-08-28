use runmat_test::coverage::{CoverageAggregate, CoverageFilter};
use runmat_test::result::RunResult;

use crate::coverage::{render_coverage_reports, CoverageReportFormat};

use super::{PluginError, PluginOutput, TestPlugin};

/// Event-compatible coverage adapter used by CLI/browser hosts and by the
/// MATLAB `CodeCoveragePlugin` projection.
pub struct CoveragePlugin {
    filter: CoverageFilter,
    formats: Vec<CoverageReportFormat>,
}

impl CoveragePlugin {
    pub fn new(filter: CoverageFilter, formats: Vec<CoverageReportFormat>) -> Self {
        Self { filter, formats }
    }
}

impl TestPlugin for CoveragePlugin {
    fn name(&self) -> &str {
        "matlab.unittest.plugins.CodeCoveragePlugin"
    }

    fn finish(
        &mut self,
        _result: &RunResult,
        coverage: &CoverageAggregate,
    ) -> Result<Option<PluginOutput>, PluginError> {
        let reports = render_coverage_reports(coverage, &self.filter, &self.formats)
            .map_err(|error| PluginError::new(error.to_string()))?;
        Ok(Some(PluginOutput {
            message: Some(format!("rendered {} coverage report(s)", reports.len())),
            reports,
        }))
    }
}
