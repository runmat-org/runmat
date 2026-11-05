//! Host authorization and measured-operation accounting for a complete FEA run.
//! Numerical libraries do not depend on account policy or lease transport.

pub(crate) mod entrypoints;
pub(crate) mod async_host;
pub use async_host::{ensure_available_capacity, reserve_work, AsyncFeaRunAdmission, AsyncFeaRunLease};
pub use entrypoints::merge_lease_error;
mod lifecycle;
mod host;
mod provider;
mod scope;
mod types;

pub use host::{FeaRunAdmissionContext, FeaRunAdmissionFuture, FeaRunAdmissionHostGuard};
pub use lifecycle::{FeaRunControl, FeaRunScope};
pub use provider::capture_fea_run_admission;
pub use provider::{
    current_fea_cpu_capacity_millicores, current_local_fea_organization_hint,
    replace_fea_run_admission, FeaRunAdmissionGuard,
};
pub use scope::{CheckedFeaRunFuture, ScopedFeaRunFuture};
pub use types::{
    FeaLocalOrganization, FeaRunAdmission, FeaRunBudget, FeaRunDelegatedGrant, FeaRunDelegation,
    FeaRunDelegationSource, FeaRunDenied, FeaRunExecutionError, FeaRunKind, FeaRunOrigin,
    FeaRunOutcome, FeaRunPermit, FeaRunRequest, FeaRunSite, FeaRunUsage,
    FEA_DEADLINE_MONITOR_STACK_BYTES, FEA_HEARTBEAT_STACK_BYTES,
};
