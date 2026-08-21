use runmat_test::coverage::CoverageAggregate;

use crate::reporter::RenderedReport;
use crate::RunnerResult;

pub(super) fn render(coverage: &CoverageAggregate) -> RunnerResult<RenderedReport> {
    let text = serde_json::to_string_pretty(coverage).map_err(super::serialization_error)?;
    Ok(super::report(
        "coverage.json",
        "application/json",
        format!("{text}\n"),
    ))
}
