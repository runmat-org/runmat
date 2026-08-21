mod evidence;
mod transfer;

use anyhow::Result;

use crate::public_api;

pub use evidence::endpoint_evidence;
pub use transfer::{ExecutionArtifactUpload, ExecutionClient};

pub type RunResponse = public_api::types::RunResponse;

pub fn public_error<T: std::fmt::Debug>(error: public_api::Error<T>) -> anyhow::Error {
    crate::auth::map_public_error(error)
}

pub(crate) fn to_i64(value: u64, field: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| anyhow::anyhow!("{field} exceeds the public API range"))
}
