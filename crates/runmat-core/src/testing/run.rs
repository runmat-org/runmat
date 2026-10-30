use std::collections::BTreeMap;

use runmat_test::context::TestExecutionContext;
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::event::{RedactionPolicy, SequencedEventSink, TestEvent, TestEventPayload};
use runmat_test::lifecycle::{
    FixtureScopeKey, LifecycleCase, LifecycleEngine, LifecycleStep, NeverCancelled,
};
use runmat_test::result::{aggregate_run_state, merge_attempts, RunResult, TestResult};
use runmat_test::TestDomainError;

use crate::{InvocationControl, RunMatSession};

use super::executor::CoreTestExecutor;
use super::source_catalog::TestSourceCatalog;

#[derive(Clone, Debug)]
pub struct CoreTestRun {
    pub plan: runmat_test::plan::TestPlan,
    pub results: Vec<TestResult>,
    pub events: Vec<TestEvent>,
    pub coverage: Vec<runmat_test::coverage::CoverageFragment>,
}

impl RunMatSession {
    /// Discover, plan, and execute one immutable test snapshot through the
    /// shared portable lifecycle engine.
    pub async fn run_test_snapshot(
        &mut self,
        snapshot: &FrozenTestRunSnapshot,
        selector: &TestSelector,
    ) -> Result<CoreTestRun, TestDomainError> {
        let plan = self
            .discover_tests(snapshot)?
            .select(selector)
            .into_plan("runmat-core")?;
        self.configure_test_snapshot(snapshot)?;
        let _admission = self.admit_execution(None);
        let parameters = plan
            .tests()
            .map(|test| (test.id.clone(), test.parameters.clone()))
            .collect::<BTreeMap<_, _>>();
        let catalog = TestSourceCatalog::from_snapshot(snapshot);
        let control =
            InvocationControl::default().with_cancellation(self.interrupt_handle().token());
        let coverage = runmat_runtime::coverage::CoverageSession::start(self.runtime_context());
        let mut executor = CoreTestExecutor::new(self, catalog, parameters, control);
        let lifecycle = LifecycleEngine::new(RedactionPolicy::new(
            Vec::<String>::new(),
            runmat_test::protocol::ProtocolLimits::default().max_output_bytes_per_attempt as usize,
        ));
        let cancellation = NeverCancelled;
        let mut events = Vec::new();
        let mut sink = SequencedEventSink::new(plan.run_id.clone(), &mut events);
        let mut results = Vec::new();
        let mut abort_run = false;
        sink.emit(TestEventPayload::RunStarted);
        'suites: for suite in &plan.suites {
            for group in &suite.fixture_groups {
                for test in &group.tests {
                    let case = LifecycleCase {
                        context: TestExecutionContext {
                            run_id: plan.run_id.clone(),
                            test_id: test.id.clone(),
                            attempt: 1,
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
                    let outcome = lifecycle
                        .execute(&case, &mut executor, &cancellation, &mut sink)
                        .await;
                    abort_run |= outcome.attempt.abort_run;
                    if let Some(mut result) = merge_attempts(test.id.clone(), vec![outcome.attempt])
                    {
                        result.report_identity =
                            Some(runmat_test::result::TestReportIdentity::from_plan(
                                &suite.display_name,
                                test,
                            ));
                        results.push(result);
                    }
                    if abort_run {
                        break 'suites;
                    }
                }
            }
        }
        let run_result = RunResult {
            run_id: plan.run_id.clone(),
            state: aggregate_run_state(results.iter().map(|result| &result.state)),
            tests: results.clone(),
        };
        sink.emit(TestEventPayload::RunFinished { result: run_result });
        drop(sink);
        let coverage = executor.coverage_fragments(&coverage.counts());
        Ok(CoreTestRun {
            plan,
            results,
            events,
            coverage,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_test::descriptor::TestSelector;
    use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
    use runmat_test::result::TerminalDisposition;

    use crate::RunMatSession;

    fn digest(value: &str) -> String {
        runmat_execution::Digest::sha256(value).to_string()
    }

    fn snapshot(path: &str, content: &str) -> FrozenTestRunSnapshot {
        FrozenTestRunSnapshot::freeze(
            digest("graph"),
            "sha256:base-sources",
            crate::program_environment(crate::CompatMode::Matlab),
            digest("config"),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: path.into(),
                content: content.into(),
            }],
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn function_test_executes_exact_discovered_procedure_without_workspace_publication() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let before = session.workspace_snapshot().unwrap();
        let run = futures::executor::block_on(session.run_test_snapshot(
            &snapshot(
                "tests/arithmeticTest.m",
                "function tests = arithmeticTest()\n tests = functiontests(localfunctions);\nend\nfunction testAddition(testCase)\n testCase.verifyEqual(1 + 1, 2);\nend\n",
            ),
            &TestSelector::default(),
        ))
        .unwrap();

        assert_eq!(run.results.len(), 1, "{run:#?}");
        assert_eq!(
            run.results[0].state.disposition,
            TerminalDisposition::Passed,
            "{run:#?}"
        );
        let after = session.workspace_snapshot().unwrap();
        assert_eq!(after.version, before.version);
        assert_eq!(
            after
                .values
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            before
                .values
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(run.events.iter().any(|event| matches!(
            event.payload,
            runmat_test::event::TestEventPayload::RunFinished { .. }
        )));
    }

    #[test]
    fn function_test_qualifications_produce_distinct_pass_and_fail_results() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let run = futures::executor::block_on(session.run_test_snapshot(
            &snapshot(
                "sampleTest.m",
                "function tests = sampleTest()\n    tests = functiontests(localfunctions);\nend\n\nfunction testPass(testCase)\n    testCase.verifyEqual(2 + 2, 4);\nend\n\nfunction testFail(testCase)\n    testCase.verifyEqual(2 + 2, 5);\nend\n",
            ),
            &TestSelector::default(),
        ))
        .unwrap();

        assert_eq!(run.results.len(), 2, "{run:#?}");
        // Plan order is by stable opaque test identity, not source order.
        let result_for = |case_name| {
            run.results
                .iter()
                .find(|result| {
                    result
                        .report_identity
                        .as_ref()
                        .is_some_and(|identity| identity.case_name == case_name)
                })
                .unwrap_or_else(|| panic!("missing result for {case_name}: {run:#?}"))
        };
        let passed = result_for("testPass");
        let failed = result_for("testFail");
        assert_eq!(
            passed.state.disposition,
            TerminalDisposition::Passed,
            "{run:#?}"
        );
        assert_eq!(
            failed.state.disposition,
            TerminalDisposition::Failed,
            "{run:#?}"
        );
        assert!(failed.state.failed);
        assert!(!failed.attempts[0].diagnostics.is_empty());
    }

    #[test]
    fn script_section_uses_the_frozen_span_and_reports_failure() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let run = futures::executor::block_on(session.run_test_snapshot(
            &snapshot(
                "tests/sectionsTest.m",
                "%% passing section\nassert(sectionValue(2) == 6)\n%% failing section\nassert(sectionValue(2) == 7)\nfunction value = sectionValue(x)\nvalue = x * 3;\nend\n",
            ),
            &TestSelector {
                names: vec!["failing".into()],
                ..TestSelector::default()
            },
        ))
        .unwrap();

        assert_eq!(run.results.len(), 1, "{run:#?}");
        assert!(run.results[0].state.failed);
        assert_eq!(
            run.results[0].state.disposition,
            TerminalDisposition::Failed
        );
        assert_eq!(run.results[0].attempts[0].diagnostics.len(), 1);
    }

    #[test]
    fn captured_section_and_helper_use_worker_project_origin() {
        let worker = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(worker.path().join("src")).unwrap();
        std::fs::write(
            worker.path().join("runmat.toml"),
            "[package]\nname = \"captured-test\"\n[sources]\nroots = [\"src\"]\n",
        )
        .unwrap();
        std::fs::write(
            worker.path().join("src/ordinary.m"),
            "function y = ordinary(x)\ny = x;\nend\n",
        )
        .unwrap();
        let handoff = runmat_package::FrozenProjectHandoff::new(
            runmat_package::build_frozen_project(
                &worker.path().join("runmat.toml"),
                BTreeSet::new(),
            )
            .unwrap(),
        );
        let revision = handoff.revision();
        let snapshot = FrozenTestRunSnapshot::freeze(
            revision.graph_digest.to_string(),
            revision.source_revision.to_string(),
            crate::program_environment(crate::CompatMode::Matlab),
            digest("captured-tests"),
            vec![
                SavedRunSource {
                    owner_identity: "path:workspace".into(),
                    relative_path: "tests/helper.m".into(),
                    content: "function y = helper(x)\ny = x + 1;\nend\n".into(),
                },
                SavedRunSource {
                    owner_identity: "path:workspace".into(),
                    relative_path: "tests/sectionTest.m".into(),
                    content: "%% selected\nassert(localHelper(2) == 4)\nfunction y = localHelper(x)\ny = helper(x) + 1;\nend\n".into(),
                },
            ],
            Vec::new(),
        )
        .unwrap();
        // Neither test source exists in the worker filesystem. The installed
        // project supplies only the worker-local path origin and dependencies.
        let mut session = RunMatSession::with_options(false, false).unwrap();
        futures::executor::block_on(session.install_project_handoff(handoff)).unwrap();
        let run = futures::executor::block_on(
            session.run_test_snapshot(&snapshot, &TestSelector::default()),
        )
        .unwrap();
        assert_eq!(run.results.len(), 1, "{run:#?}");
        assert_eq!(
            run.results[0].state.disposition,
            TerminalDisposition::Passed,
            "{run:#?}"
        );
    }
}
