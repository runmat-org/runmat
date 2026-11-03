//! Explicit finite admission for numerical contract tests, with thread-local restoration.
use runmat_execution::resource::ResourceUsage;
use runmat_runtime::analysis::fea_run_admission::*;
pub fn guard() -> FeaRunAdmissionHostGuard {
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
                held_core_millis: request
                    .allocated_cpu_millicores
                    .checked_mul(3_600_000)
                    .ok_or(FeaRunDenied::LicenseInvalid)?
                    .div_ceil(1000),
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
        fn heartbeat(&self, _: &ResourceUsage) -> Result<FeaRunBudget, FeaRunDenied> {
            self.budget.validate()?;
            Ok(self.budget)
        }
        fn settle(
            self: Box<Self>,
            _: FeaRunOutcome,
            _: &ResourceUsage,
        ) -> Result<(), FeaRunDenied> {
            Ok(())
        }
    }
    FeaRunAdmissionContext::new(std::sync::Arc::new(Host)).enter()
}
