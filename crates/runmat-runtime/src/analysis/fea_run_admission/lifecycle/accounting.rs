//! Numerical grant acquisition and disjoint child delegation for one root.
use super::*;

impl RunState {
    pub(super) fn request_grant(&self, required: OperationCount) -> Result<(), FeaRunDenied> {
        self.check()?;
        let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner());
        // Recheck after waiting for an in-flight heartbeat or settlement.
        self.check()?;
        let current = self.accounting.ledger().snapshot()?;
        if required > current.budget {
            return Err(runmat_simulation_usage::LedgerError::BudgetExceeded {
                required,
                budget: current.budget,
            }
            .into());
        }
        if required <= current.granted {
            return Ok(());
        }
        let permit = permit.as_ref().ok_or_else(|| {
            FeaRunDenied::from(runmat_simulation_usage::LedgerError::MoreGrantRequired {
                required,
                granted: current.granted,
            })
        })?;
        let renewed = permit.request_grant(&self.usage()?, required);
        let budget = match renewed {
            Ok(budget) => budget,
            Err(denied) => {
                self.deny(denied.clone());
                return Err(denied);
            }
        };
        self.renew(budget)?;
        if budget.granted_ops < required {
            return Err(runmat_simulation_usage::LedgerError::MoreGrantRequired {
                required,
                granted: budget.granted_ops,
            }
            .into());
        }
        Ok(())
    }
}

impl crate::analysis::fea_run_admission::FeaRunDelegationSource for RunState {
    fn current_budget(&self) -> Result<FeaRunBudget, FeaRunDenied> {
        self.check()?;
        let lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        match lease.authority {
            LeaseAuthority::Metered(budget) => Ok(budget),
            LeaseAuthority::ParentSupervised(_) => Err(FeaRunDenied::LicenseInvalid),
        }
    }
    fn delegate(
        &self,
        maximum: OperationCount,
    ) -> Result<crate::analysis::fea_run_admission::FeaRunDelegatedGrant, FeaRunDenied> {
        self.check()?;
        let lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        let LeaseAuthority::Metered(budget) = lease.authority else {
            return Err(FeaRunDenied::LicenseInvalid);
        };
        let mut budget = budget;
        budget.granted_ops = maximum;
        let grant = crate::analysis::fea_run_admission::FeaRunDelegation {
            origin: lease.origin.clone().ok_or(FeaRunDenied::LicenseInvalid)?,
            request: self.request.clone(),
            budget,
            elapsed_wall_millis: self.resource_usage().wall_millis,
            observed_at_unix_millis: runmat_time::unix_timestamp_ms().min(u128::from(u64::MAX))
                as u64,
            allocation_ceiling_millicores: lease.allocation_ceiling_millicores,
        };
        grant.validate()?;
        drop(lease); // Acquiring capacity may call the root's renewal port.
        let reservation = self.accounting.delegate(maximum)?;
        Ok(crate::analysis::fea_run_admission::FeaRunDelegatedGrant {
            delegation: grant,
            reservation,
        })
    }
    fn cancellation_token(&self) -> CancellationToken {
        self.token()
    }
}
impl runmat_simulation_usage::GrantRenewal for RunState {
    fn request_grant(
        &self,
        required: OperationCount,
    ) -> Result<(), runmat_simulation_usage::LedgerError> {
        RunState::request_grant(self, required).map_err(|denied| match denied {
            FeaRunDenied::Accounting(cause) => cause,
            denied => {
                // Keep the original authenticated host cause on the run. The
                // numerical layer has no dependency on account/transport types.
                self.deny(denied);
                runmat_simulation_usage::LedgerError::GrantUnavailable
            }
        })
    }
}
