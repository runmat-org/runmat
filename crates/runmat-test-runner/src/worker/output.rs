use runmat_test::event::TestEvent;
use runmat_test::result::AttemptResult;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerExecution {
    pub result: AttemptResult,
    pub events: Vec<TestEvent>,
    #[serde(default)]
    pub coverage: Vec<runmat_test::coverage::CoverageFragment>,
}
