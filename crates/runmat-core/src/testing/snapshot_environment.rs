use runmat_execution::LanguageCompatibilityMode;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::TestDomainError;

use crate::CompatMode;

impl crate::RunMatSession {
    /// Configure a test session from its validated frozen execution identity,
    /// before installing project sources or executing fixture code.
    pub fn configure_test_snapshot(
        &mut self,
        snapshot: &FrozenTestRunSnapshot,
    ) -> Result<(), TestDomainError> {
        self.set_compat_mode(compatibility(snapshot)?);
        Ok(())
    }
}

/// A frozen test revision owns its language policy. Workers may have been
/// constructed with a different embedding default, but cannot reinterpret the
/// captured source or accept a different compiler/runtime/catalog identity.
pub(super) fn compatibility(
    snapshot: &FrozenTestRunSnapshot,
) -> Result<CompatMode, TestDomainError> {
    snapshot.validate()?;
    let environment = snapshot.program_revision.environment();
    let mode = match environment.compatibility_mode {
        LanguageCompatibilityMode::RunMat => CompatMode::RunMat,
        LanguageCompatibilityMode::Matlab => CompatMode::Matlab,
        LanguageCompatibilityMode::Strict => CompatMode::Strict,
    };
    if environment != crate::program_environment(mode) {
        return Err(TestDomainError::InvalidField {
            field: "program_revision.environment",
            reason: "test snapshot compiler, runtime or builtin catalog differs from this executor"
                .into(),
        });
    }
    Ok(mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_test::descriptor::TestSelector;
    use runmat_test::discovery::SavedRunSource;

    fn snapshot(mode: CompatMode) -> FrozenTestRunSnapshot {
        FrozenTestRunSnapshot::freeze(
            runmat_execution::Digest::sha256(b"test graph").to_string(),
            "test sources",
            crate::program_environment(mode),
            runmat_execution::Digest::sha256(b"test config").to_string(),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: "policy_test.m".into(),
                content: "%% policy\nx=uint16([1;2;3;4]); y=int32([2;1;4;3]); assert(abs(corr(x,y)-0.6)<1e-12);\n".into(),
            }],
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn frozen_language_policy_reaches_discovery_and_execution_in_a_default_worker() {
        for mode in [CompatMode::RunMat, CompatMode::Matlab, CompatMode::Strict] {
            let snapshot = snapshot(mode);
            assert_eq!(compatibility(&snapshot).unwrap(), mode);
            let mut session = crate::RunMatSession::with_options(false, false).unwrap();
            let run = futures::executor::block_on(
                session.run_test_snapshot(&snapshot, &TestSelector::default()),
            )
            .unwrap();
            assert_eq!(session.compat_mode(), mode);
            assert_eq!(run.results.len(), 1);
            assert_eq!(run.results[0].state.failed, mode != CompatMode::RunMat);
        }
    }
}
