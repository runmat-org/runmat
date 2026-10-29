//! Host authorization and allocated-time accounting for a complete FEA run.
//! Numerical libraries do not depend on account policy or lease transport.

pub(crate) mod entrypoints;
mod lifecycle;
mod host;
mod provider;
mod scope;
mod types;

pub use host::{FeaRunAdmissionContext, FeaRunAdmissionFuture};
pub use lifecycle::FeaRunScope;
pub use provider::{replace_fea_run_admission, FeaRunAdmissionGuard};
pub use scope::ScopedFeaRunFuture;
pub use types::{
    FeaRunAdmission, FeaRunDenied, FeaRunKind, FeaRunOutcome, FeaRunPermit, FeaRunRequest,
    FeaRunSite,
};
