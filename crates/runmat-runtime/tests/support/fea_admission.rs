//! Explicit finite admission for numerical contract tests, with thread-local restoration.
//!
//! This test host admits each run like a real host does: it grants the run's operation budget
//! and installs the run's admitted memory as the study publication authority, so publications
//! stage within a fixed, machine-independent budget.
use runmat_runtime::analysis::fea_run_admission::*;
use runmat_simulation_usage::OperationCount;

/// The test host's admitted run memory: a fixed fixture with nothing resident, so every machine
/// sees the same publication budget.
struct TestHostMemory;
impl runmat_runtime::analysis::PublicationMemoryAuthority for TestHostMemory {
    fn publication_memory(&self) -> Result<runmat_runtime::analysis::PublicationMemory, String> {
        Ok(runmat_runtime::analysis::PublicationMemory {
            admitted_bytes: 128 * 1024 * 1024,
            resident_bytes: 0,
        })
    }
}

/// Install the test host's run memory once per test process. Every admitted run in the process
/// shares the same fixture, so concurrent tests never replace each other's authority.
fn admit_publication_memory() {
    static ADMITTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ADMITTED.get_or_init(|| {
        runmat_runtime::analysis::install_publication_memory_authority(std::sync::Arc::new(
            TestHostMemory,
        ))
        .expect("install the test host's publication memory");
    });
}

pub fn guard() -> FeaRunAdmissionHostGuard {
    admit_publication_memory();
    struct Host;
    struct Permit {
        budget: FeaRunBudget,
    }
    impl FeaRunAdmission for Host {
        fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            if request.site != FeaRunSite::Local {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            let budget = FeaRunBudget {
                budget_ops: OperationCount::from_operations(u64::MAX),
                granted_ops: OperationCount::from_operations(u64::MAX),
                expires_at_unix_millis: u64::try_from(runmat_time::unix_timestamp_ms() + 3_600_000)
                    .map_err(|_| FeaRunDenied::LicenseInvalid)?,
                heartbeat_millis: 30_000,
            };
            budget.validate()?;
            Ok(Box::new(Permit { budget }))
        }
    }
    impl FeaRunPermit for Permit {
        fn budget(&self) -> FeaRunBudget {
            self.budget
        }
        fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
            self.budget.validate()?;
            Ok(self.budget)
        }
        fn settle(self: Box<Self>, _: FeaRunOutcome, _: &FeaRunUsage) -> Result<(), FeaRunDenied> {
            Ok(())
        }
    }
    FeaRunAdmissionContext::new(std::sync::Arc::new(Host)).enter()
}
