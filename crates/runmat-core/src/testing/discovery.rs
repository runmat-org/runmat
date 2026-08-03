use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, PreparedTestRun, TestDiscovery};
use runmat_test::TestDomainError;

use crate::RunMatSession;

impl RunMatSession {
    /// Discover tests against the exact frozen run snapshot. If a project
    /// handoff is installed, its graph and saved-source revision must match.
    pub fn discover_tests(
        &self,
        snapshot: &FrozenTestRunSnapshot,
    ) -> Result<TestDiscovery, TestDomainError> {
        // Native link-time registration and the generated WASM registry must
        // expose the same builtin semantics before HIR lowers bare builtin
        // forms such as `functiontests(localfunctions)`.
        runmat_runtime::builtins::wasm_registry::register_all();
        snapshot.validate()?;
        if let Some(installed) = self.project_revision() {
            if *snapshot.program_revision.graph_digest()
                != runmat_execution::Digest::from_bytes(*installed.graph_digest.bytes())
            {
                return Err(TestDomainError::InvalidField {
                    field: "program_revision.graph_digest",
                    reason: "test snapshot does not match the installed project graph".into(),
                });
            }
            if snapshot.base_source_digest != installed.source_revision.to_string() {
                return Err(TestDomainError::InvalidField {
                    field: "base_source_digest",
                    reason: "saved sources changed after the test snapshot was frozen".into(),
                });
            }
        }
        Ok(runmat_static_analysis::testing::discover_frozen_tests(
            snapshot,
            self.compat_mode(),
        ))
    }

    /// Discover and materialize the canonical plan for one immutable snapshot.
    ///
    /// Hosts retain the unfiltered discovery for their test explorer while the
    /// selected plan is derived by the shared Rust domain model.
    pub fn prepare_tests(
        &self,
        snapshot: &FrozenTestRunSnapshot,
        selector: &TestSelector,
    ) -> Result<PreparedTestRun, TestDomainError> {
        let discovery = self.discover_tests(snapshot)?;
        let invocation_identity =
            serde_json::to_string(selector).map_err(|error| TestDomainError::InvalidField {
                field: "selector",
                reason: format!("failed to encode deterministic test selection: {error}"),
            })?;
        let plan = discovery
            .clone()
            .select(selector)
            .into_plan(invocation_identity)?;
        Ok(PreparedTestRun {
            snapshot: snapshot.clone(),
            discovery,
            plan,
        })
    }
}

#[cfg(test)]
mod tests {
    use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};

    use crate::RunMatSession;

    fn digest(value: &str) -> String {
        runmat_execution::Digest::sha256(value).to_string()
    }

    #[test]
    fn core_discovers_from_the_caller_frozen_snapshot() {
        let snapshot = FrozenTestRunSnapshot::freeze(
            digest("graph"),
            "sha256:base-sources",
            crate::program_environment(crate::CompatMode::Matlab),
            digest("config"),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: "tests/core_test.m".into(),
                content: "%% core section\nassert(1 == 1)\n".into(),
            }],
            Vec::new(),
        )
        .unwrap();
        let session = RunMatSession::with_options(false, false).unwrap();
        let discovery = session.discover_tests(&snapshot).unwrap();

        assert!(discovery.diagnostics.is_empty());
        assert_eq!(discovery.suites.len(), 1);
        assert_eq!(discovery.suites[0].tests.len(), 1);
        assert_eq!(discovery.program_revision, snapshot.program_revision);
    }

    #[test]
    fn core_prepares_the_selected_plan_and_preserves_full_discovery() {
        let snapshot = FrozenTestRunSnapshot::freeze(
            digest("graph"),
            "sha256:base-sources",
            crate::program_environment(crate::CompatMode::Matlab),
            digest("config"),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: "tests/core_test.m".into(),
                content: "%% first\nassert(1 == 1)\n%% second\nassert(2 == 2)\n".into(),
            }],
            Vec::new(),
        )
        .unwrap();
        let session = RunMatSession::with_options(false, false).unwrap();
        let prepared = session
            .prepare_tests(
                &snapshot,
                &runmat_test::descriptor::TestSelector {
                    names: vec!["first".into()],
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(prepared.discovery.suites[0].tests.len(), 2);
        assert_eq!(prepared.plan.tests().count(), 1);
        assert_eq!(prepared.snapshot, snapshot);
    }
}
