use runmat_test::discovery::{FrozenTestRunSnapshot, TestDiscovery};
use runmat_test::TestDomainError;

use crate::RunMatSession;

impl RunMatSession {
    /// Discover tests against the exact frozen run snapshot. If a project
    /// handoff is installed, its graph and saved-source revision must match;
    /// intentionally selected unsaved buffers are already bound into the
    /// snapshot's derived `ProgramRevision`.
    pub fn discover_tests(
        &self,
        snapshot: &FrozenTestRunSnapshot,
    ) -> Result<TestDiscovery, TestDomainError> {
        snapshot.validate()?;
        if let Some(installed) = self.project_revision() {
            if snapshot.program_revision.graph_digest != installed.graph_digest.to_string() {
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
}

#[cfg(test)]
mod tests {
    use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};

    use crate::RunMatSession;

    #[test]
    fn core_discovers_from_the_caller_frozen_snapshot() {
        let snapshot = FrozenTestRunSnapshot::freeze(
            "sha256:graph",
            "sha256:base-sources",
            1,
            1,
            "sha256:config",
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
}
