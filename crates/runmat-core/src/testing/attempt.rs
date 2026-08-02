use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use runmat_test::context::TestExecutionContext;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::event::{RedactionPolicy, SequencedEventSink, TestEvent};
use runmat_test::identity::TestId;
use runmat_test::lifecycle::{
    CancellationProbe, FixtureScopeKey, LifecycleCase, LifecycleEngine, LifecycleStep,
};
use runmat_test::plan::TestPlan;
use runmat_test::result::AttemptResult;
use runmat_test::TestDomainError;

use crate::{InvocationControl, RunMatSession};

use super::executor::CoreTestExecutor;
use super::source_catalog::TestSourceCatalog;

#[derive(Clone, Debug)]
pub struct CoreTestAttempt {
    pub result: AttemptResult,
    pub events: Vec<TestEvent>,
    pub coverage: Vec<runmat_test::coverage::CoverageFragment>,
}

struct AtomicCancellation(Arc<AtomicBool>);

impl CancellationProbe for AtomicCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl RunMatSession {
    /// Execute one exact case from a prevalidated immutable plan. Coordination,
    /// retries, timeouts, and worker ownership remain outside Core.
    pub async fn execute_planned_test(
        &mut self,
        snapshot: &FrozenTestRunSnapshot,
        plan: &TestPlan,
        test_id: &TestId,
        attempt: u32,
        cancellation: Arc<AtomicBool>,
    ) -> Result<CoreTestAttempt, TestDomainError> {
        snapshot.validate()?;
        if snapshot.program_revision != plan.program_revision {
            return Err(TestDomainError::InvalidField {
                field: "plan.program_revision",
                reason: "test plan and frozen source snapshot revisions differ".into(),
            });
        }
        if attempt == 0 {
            return Err(TestDomainError::InvalidField {
                field: "attempt",
                reason: "attempt numbers start at one".into(),
            });
        }
        let (group, test) = plan
            .suites
            .iter()
            .flat_map(|suite| suite.fixture_groups.iter())
            .find_map(|group| {
                group
                    .tests
                    .iter()
                    .find(|test| test.id == *test_id)
                    .map(|test| (group, test))
            })
            .ok_or_else(|| TestDomainError::InvalidField {
                field: "test_id",
                reason: format!("test '{}' is not present in the plan", test_id.as_str()),
            })?;
        let parameters = plan
            .tests()
            .map(|test| (test.id.clone(), test.parameters.clone()))
            .collect::<BTreeMap<_, _>>();
        let catalog = TestSourceCatalog::from_snapshot(snapshot);
        let control = InvocationControl::default().with_cancellation(cancellation.clone());
        let mut executor = CoreTestExecutor::new(self, catalog, parameters, control);
        let lifecycle = LifecycleEngine::new(RedactionPolicy::new(
            Vec::<String>::new(),
            runmat_test::protocol::ProtocolLimits::default().max_output_bytes_per_attempt as usize,
        ));
        let case = LifecycleCase {
            context: TestExecutionContext {
                run_id: plan.run_id.clone(),
                test_id: test.id.clone(),
                attempt,
                random_seed: 0,
            },
            setups: group
                .fixtures
                .iter()
                .filter_map(|fixture| {
                    fixture.setup.clone().map(|procedure| LifecycleStep {
                        scope: FixtureScopeKey {
                            scope: fixture.scope,
                            identity: fixture.id.as_str().to_owned(),
                        },
                        procedure,
                    })
                })
                .collect(),
            body: test.procedure.clone(),
            declared_teardowns: group
                .fixtures
                .iter()
                .filter_map(|fixture| {
                    fixture.teardown.clone().map(|procedure| LifecycleStep {
                        scope: FixtureScopeKey {
                            scope: fixture.scope,
                            identity: fixture.id.as_str().to_owned(),
                        },
                        procedure,
                    })
                })
                .collect(),
        };
        let mut events = Vec::new();
        let mut sink = SequencedEventSink::new(plan.run_id.clone(), &mut events);
        let coverage = runmat_vm::coverage::CoverageSession::start();
        let outcome = lifecycle
            .execute(
                &case,
                &mut executor,
                &AtomicCancellation(cancellation),
                &mut sink,
            )
            .await;
        drop(sink);
        let coverage = executor.coverage_fragments(&coverage.counts());
        Ok(CoreTestAttempt {
            result: outcome.attempt,
            events,
            coverage,
        })
    }
}
