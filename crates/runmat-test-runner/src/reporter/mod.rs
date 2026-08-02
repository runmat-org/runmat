mod fanout;
mod human;
mod json;
mod junit;
mod location;
mod summary;
mod tap;

pub use fanout::ReporterFanout;
pub use human::HumanReporter;
pub use json::JsonReporter;
pub use junit::JunitReporter;
pub use summary::ReportSummary;
pub use tap::TapReporter;

use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use crate::RunnerResult;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedReport {
    pub name: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

pub trait Reporter {
    fn event(&mut self, event: &TestEvent) -> RunnerResult<()>;
    fn finish(&mut self, result: &RunResult) -> RunnerResult<RenderedReport>;
}
