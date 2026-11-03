//! Explicit numerical-harness host, never linked into the production runtime.
use runmat_execution::resource::ResourceUsage;
use runmat_runtime::analysis::fea_run_admission::*;

pub fn context() -> FeaRunAdmissionContext {
    struct NumericalHost;
    struct Permit {
        budget: FeaRunBudget,
    }
    impl FeaRunAdmission for NumericalHost {
        fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            if request.site != FeaRunSite::Local {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| FeaRunDenied::LicenseInvalid)?
                .as_millis();
            let held = (u128::from(request.allocated_cpu_millicores) * 3_600_000).div_ceil(1000);
            let budget = FeaRunBudget {
                held_core_millis: u64::try_from(held).map_err(|_| FeaRunDenied::LicenseInvalid)?,
                expires_at_unix_millis: u64::try_from(now + 3_600_000)
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
    FeaRunAdmissionContext::new(std::sync::Arc::new(NumericalHost))
}
