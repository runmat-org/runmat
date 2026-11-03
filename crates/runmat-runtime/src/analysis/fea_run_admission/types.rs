use runmat_execution::resource::ResourceUsage;
use serde::{Deserialize, Serialize};
pub const FEA_DEADLINE_MONITOR_STACK_BYTES: usize = 64 * 1024;
pub const FEA_HEARTBEAT_STACK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaRunKind {
    Meshing,
    Solve,
    MeshAndSolve,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaRunSite {
    Local,
    Dispatched,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaRunOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaRunRequest {
    pub run_id: String,
    pub kind: FeaRunKind,
    pub site: FeaRunSite,
    pub allocated_cpu_millicores: u64,
    pub estimated_wall_seconds: Option<u64>,
    pub study_identity: String,
}

impl FeaRunRequest {
    pub fn local(
        kind: FeaRunKind,
        allocated_cpu_millicores: u64,
        study_identity: runmat_execution::Digest,
    ) -> Result<Self, FeaRunDenied> {
        use std::fmt::Write as _;
        let mut random = [0_u8; 16];
        getrandom::getrandom(&mut random).map_err(|_| FeaRunDenied::LicenseInvalid)?;
        let mut run_id = String::with_capacity(36);
        run_id.push_str("run_");
        for byte in random {
            write!(run_id, "{byte:02x}").expect("formatting into a String");
        }
        let request = Self {
            run_id,
            kind,
            site: FeaRunSite::Local,
            allocated_cpu_millicores,
            estimated_wall_seconds: None,
            study_identity: study_identity.to_string(),
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), FeaRunDenied> {
        let valid_id = !self.run_id.is_empty()
            && self.run_id.len() <= 128
            && self
                .run_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b));
        let valid_estimate = self
            .estimated_wall_seconds
            .is_none_or(|v| (1..=604_800).contains(&v));
        if !valid_id
            || !(1..=4_096_000).contains(&self.allocated_cpu_millicores)
            || !valid_estimate
            || self.study_identity.is_empty()
            || self.study_identity.len() > 256
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FeaRunDenied {
    SignInRequired,
    AllowanceUsed {
        refills_at_unix_seconds: Option<i64>,
    },
    LeaseExpired,
    LicenseInvalid,
    ServerUnreachable,
}

impl FeaRunDenied {
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::SignInRequired => "RM.FEA.SignInRequired",
            Self::AllowanceUsed { .. } => "RM.FEA.AllowanceUsed",
            Self::LeaseExpired | Self::LicenseInvalid => "RM.FEA.LeaseInvalid",
            Self::ServerUnreachable => "RM.FEA.ServerUnreachable",
        }
    }

    pub fn into_operation_error(
        self,
        operation: &str,
        version: &str,
        context: &crate::operations::OperationContext,
    ) -> crate::operations::OperationErrorEnvelope {
        use crate::operations::*;
        operation_error(
            operation,
            version,
            context,
            OperationErrorSpec {
                error_code: self.error_code(),
                error_type: OperationErrorType::Capacity,
                retryable: matches!(self, Self::ServerUnreachable),
                severity: OperationErrorSeverity::Error,
            },
            self.to_string(),
            Default::default(),
        )
    }
}

impl std::fmt::Display for FeaRunDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SignInRequired => f.write_str("Sign in with `runmat login` to mesh or solve."),
            Self::AllowanceUsed {
                refills_at_unix_seconds,
            } => {
                let date =
                    refills_at_unix_seconds.and_then(|v| chrono::DateTime::from_timestamp(v, 0));
                f.write_str("You've used this week's core-hours. ")?;
                if let Some(date) = date {
                    write!(f, "They refill on {}. ", date.format("%Y-%m-%d %H:%M UTC"))?;
                }
                f.write_str("Add a core-hour pack or move to Scale: https://runmat.com/pricing")
            }
            Self::LeaseExpired | Self::LicenseInvalid => f.write_str(
                "Your FEA authorization needs refreshing. Run `runmat login`, then try again.",
            ),
            Self::ServerUnreachable => {
                f.write_str("Meshing and solving need a connection to RunMat.")
            }
        }
    }
}

impl std::error::Error for FeaRunDenied {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FeaRunExecutionError {
    Cancelled,
    Denied(FeaRunDenied),
}
impl FeaRunExecutionError {
    pub fn outcome(&self) -> FeaRunOutcome {
        match self {
            Self::Cancelled => FeaRunOutcome::Cancelled,
            Self::Denied(_) => FeaRunOutcome::Failed,
        }
    }
    pub fn into_operation_error(
        self,
        operation: &str,
        version: &str,
        context: &crate::operations::OperationContext,
    ) -> crate::operations::OperationErrorEnvelope {
        match self {
            Self::Denied(denied) => denied.into_operation_error(operation, version, context),
            Self::Cancelled => crate::operations::operation_error(
                operation,
                version,
                context,
                crate::operations::OperationErrorSpec {
                    error_code: "RM.FEA.Cancelled",
                    error_type: crate::operations::OperationErrorType::Cancelled,
                    retryable: false,
                    severity: crate::operations::OperationErrorSeverity::Error,
                },
                "FEA run cancelled.",
                Default::default(),
            ),
        }
    }
}
impl std::fmt::Display for FeaRunExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => f.write_str("FEA run cancelled."),
            Self::Denied(denied) => std::fmt::Display::fmt(denied, f),
        }
    }
}
impl std::error::Error for FeaRunExecutionError {}

/// Cumulative allocated CPU allowance and expiry from the original lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaRunBudget {
    pub held_core_millis: u64,
    pub expires_at_unix_millis: u64,
    pub heartbeat_millis: u64,
}

impl FeaRunBudget {
    pub fn validate(&self) -> Result<(), FeaRunDenied> {
        if self.held_core_millis == 0 {
            return Err(FeaRunDenied::AllowanceUsed {
                refills_at_unix_seconds: None,
            });
        }
        if self.expires_at_unix_millis
            <= runmat_time::unix_timestamp_ms().min(u128::from(u64::MAX)) as u64
        {
            return Err(FeaRunDenied::LeaseExpired);
        }
        if !(1..=120_000).contains(&self.heartbeat_millis) {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(())
    }
}

/// Host calls must have bounded transport timeouts. A separate runtime monitor
/// enforces the current hold while a heartbeat is in flight.
pub trait FeaRunPermit: Send {
    fn budget(&self) -> FeaRunBudget;
    fn heartbeat(&self, usage: &ResourceUsage) -> Result<FeaRunBudget, FeaRunDenied>;
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &ResourceUsage,
    ) -> Result<(), FeaRunDenied>;
}

/// Captured local account selection. Invalid selection denies FEA, never plain work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaLocalOrganization {
    #[default]
    Default,
    Selected(uuid::Uuid),
    Invalid,
}

pub trait FeaRunAdmission: Send + Sync {
    /// Local account selector only; Server still authorizes the signed-in person.
    fn local_organization_hint(&self) -> FeaLocalOrganization {
        FeaLocalOrganization::Default
    }
    fn allocated_cpu_capacity_millicores(&self) -> Option<u64> {
        None
    }
    fn site(&self) -> FeaRunSite {
        FeaRunSite::Local
    }
    fn origin(&self) -> Option<FeaRunOrigin> {
        None
    }
    fn delegation_source(&self) -> Option<std::sync::Arc<dyn FeaRunDelegationSource>> {
        None
    }
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied>;
}

/// Lineage retained by trusted hosts; none of these identifiers is a credential.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "camelCase", deny_unknown_fields)]
pub enum FeaRunOrigin {
    LocalLease {
        run_id: String,
    },
    ClusterDriver {
        run_id: String,
        allocation_lease_id: String,
        driver_lease_id: String,
        fencing_token: u64,
    },
    ClusterAttempt {
        run_id: String,
        scope_id: runmat_execution::ExecutionScopeId,
        pool_id: runmat_execution::PoolId,
        task_id: runmat_execution::TaskId,
        attempt_id: runmat_execution::identity::AttemptId,
        driver_fence: u64,
    },
}
impl FeaRunOrigin {
    pub fn validate(&self, site: FeaRunSite) -> Result<(), FeaRunDenied> {
        let text = |value: &str| !value.is_empty() && value.len() <= 256;
        let valid = match self {
            Self::LocalLease { run_id } => site == FeaRunSite::Local && text(run_id),
            Self::ClusterDriver {
                run_id,
                allocation_lease_id,
                driver_lease_id,
                fencing_token,
            } => {
                site == FeaRunSite::Dispatched
                    && text(run_id)
                    && text(allocation_lease_id)
                    && text(driver_lease_id)
                    && *fencing_token > 0
            }
            Self::ClusterAttempt {
                run_id,
                driver_fence,
                ..
            } => site == FeaRunSite::Dispatched && text(run_id) && *driver_fence > 0,
        };
        if valid {
            Ok(())
        } else {
            Err(FeaRunDenied::LicenseInvalid)
        }
    }
}

/// This is transport data only. A trusted private parent must authenticate and
/// bind it to the original child assignment before installing a delegated scope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaRunDelegation {
    pub origin: FeaRunOrigin,
    pub request: FeaRunRequest,
    pub budget: FeaRunBudget,
    pub elapsed_wall_millis: u64,
    pub observed_at_unix_millis: u64,
    pub allocation_ceiling_millicores: u64,
}
impl FeaRunDelegation {
    pub fn validate(&self) -> Result<(), FeaRunDenied> {
        self.request.validate()?;
        self.budget.validate()?;
        self.origin.validate(self.request.site)?;
        if matches!(&self.origin, FeaRunOrigin::LocalLease { run_id } if run_id != &self.request.run_id)
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        if self.observed_at_unix_millis == 0
            || u128::from(self.observed_at_unix_millis) > runmat_time::unix_timestamp_ms()
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let used = (u128::from(self.elapsed_wall_millis)
            * u128::from(self.request.allocated_cpu_millicores))
        .div_ceil(1000);
        if used >= u128::from(self.budget.held_core_millis) {
            return Err(FeaRunDenied::AllowanceUsed {
                refills_at_unix_seconds: None,
            });
        }
        if self.allocation_ceiling_millicores == 0
            || self.allocation_ceiling_millicores > self.request.allocated_cpu_millicores
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(())
    }
}
pub trait FeaRunDelegationSource: Send + Sync {
    fn snapshot(&self) -> Result<FeaRunDelegation, FeaRunDenied>;
    fn cancellation_token(&self) -> runmat_execution::cancellation::CancellationToken;
}
