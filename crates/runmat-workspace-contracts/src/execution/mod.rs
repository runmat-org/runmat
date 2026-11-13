//! Provider-independent work identity and execution-authority policy.
//! Providers supply time and atomic storage under their existing project scope.
mod policy;
pub use policy::*;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

macro_rules! work_identity {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
        #[cfg_attr(feature = "ts", derive(ts_rs::TS))]
        #[cfg_attr(feature = "ts", ts(type = "string"))]
        #[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
        pub struct $name(pub String);
    };
}
work_identity!(SessionId);
work_identity!(TurnId);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, Serialize, Deserialize, JsonSchema,
)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ExecutionAuthorityError {
    #[error("invalid execution authority identity or time")]
    InvalidIdentity,
    #[error("invalid persisted execution ownership")]
    InvalidState,
    #[error("execution lease duration must be between one second and five minutes")]
    InvalidDuration,
    #[error("execution fence overflow")]
    FenceOverflow,
    #[error("execution lease expiry overflow")]
    ExpiryOverflow,
}
