//! Explicit numerical-harness host, never linked into the production runtime.
use runmat_runtime::analysis::fea_run_admission::*;
use runmat_simulation_usage::OperationCount;

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
            let budget = FeaRunBudget {
                budget_ops: OperationCount::from_operations(u64::MAX),
                granted_ops: OperationCount::from_operations(u64::MAX),
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
        fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
            self.budget.validate()?;
            Ok(self.budget)
        }
        fn settle(self: Box<Self>, _: FeaRunOutcome, _: &FeaRunUsage) -> Result<(), FeaRunDenied> {
            Ok(())
        }
    }
    FeaRunAdmissionContext::new(std::sync::Arc::new(NumericalHost))
}
