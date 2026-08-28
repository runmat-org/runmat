use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::plan::TestPlan;
use runmat_test::result::RunResult;
use runmat_test_runner::host::IsolationMode;
use runmat_test_runner::reporter::RenderedReport;
use runmat_test_runner::schedule::RetryPolicy;
use runmat_test_runner::worker::RunSubmission;
use runmat_test_runner::CoordinatorConfig;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct BrowserRunInput {
    pub plan: TestPlan,
    pub snapshot: FrozenTestRunSnapshot,
    #[serde(default)]
    pub options: BrowserRunOptions,
}

impl BrowserRunInput {
    pub fn into_parts(
        self,
    ) -> runmat_test_runner::RunnerResult<(RunSubmission, CoordinatorConfig)> {
        let submission = RunSubmission::new(self.plan, self.snapshot)?;
        let config = CoordinatorConfig {
            isolation: self.options.isolation,
            jobs: self.options.jobs,
            timeout_ms: self.options.timeout_ms,
            cancellation_grace_ms: self.options.cancellation_grace_ms,
            retry: RetryPolicy {
                max_attempts: self.options.max_attempts,
            },
            shard_index: self.options.shard_index,
            shard_count: self.options.shard_count,
        };
        Ok((submission, config))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub(super) struct BrowserRunOptions {
    pub isolation: IsolationMode,
    pub jobs: usize,
    pub timeout_ms: Option<u64>,
    pub cancellation_grace_ms: u64,
    pub max_attempts: u32,
    pub shard_index: Option<u32>,
    pub shard_count: Option<u32>,
    pub reports: Vec<BrowserReport>,
    pub coverage: BrowserCoverageOptions,
}

impl Default for BrowserRunOptions {
    fn default() -> Self {
        Self {
            isolation: IsolationMode::Auto,
            jobs: 1,
            timeout_ms: None,
            cancellation_grace_ms: 1_000,
            max_attempts: 1,
            shard_index: None,
            shard_count: None,
            reports: vec![BrowserReport::Human],
            coverage: BrowserCoverageOptions::default(),
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum BrowserReport {
    Human,
    Json,
    Junit,
    Tap,
}

#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub(super) struct BrowserCoverageOptions {
    pub enabled: bool,
    pub formats: Vec<BrowserCoverageFormat>,
    pub roots: Vec<String>,
    pub exclude: Vec<String>,
    pub include_generated: bool,
    pub include_vendor: bool,
}

impl BrowserCoverageOptions {
    pub fn is_requested(&self) -> bool {
        self.enabled
            || !self.formats.is_empty()
            || !self.roots.is_empty()
            || !self.exclude.is_empty()
            || self.include_generated
            || self.include_vendor
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum BrowserCoverageFormat {
    Json,
    Lcov,
    Cobertura,
    Html,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct BrowserRunOutput {
    pub result: RunResult,
    pub events: Vec<runmat_test::event::TestEvent>,
    pub reports: Vec<RenderedReportWire>,
    pub infrastructure_failures: usize,
    pub plugin_failures: usize,
    pub isolation: IsolationMode,
    pub coverage: runmat_test::coverage::CoverageAggregate,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RenderedReportWire {
    pub name: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

impl From<RenderedReport> for RenderedReportWire {
    fn from(value: RenderedReport) -> Self {
        Self {
            name: value.name,
            media_type: value.media_type,
            bytes: value.bytes,
        }
    }
}
