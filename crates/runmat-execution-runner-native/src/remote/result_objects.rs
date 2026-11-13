use std::sync::Arc;

use runmat_execution::value::ValueRef;

use super::{object_transfer::RemoteObjectStore, RemotePoolDriver};
use crate::{NativeExecutionError, NativeExecutionResult};

/// Keeps result bytes with the execution owner until the publisher requests one logical object.
pub(super) enum RemoteResultObjects {
    Local(RemoteObjectStore),
    Pool(Arc<RemotePoolDriver>),
    /// The response carries every value inline (Graph and runtime failures).
    None,
}

impl RemoteResultObjects {
    pub(super) fn read(
        &self,
        reference: &ValueRef,
        authorization_scope: &str,
    ) -> NativeExecutionResult<Vec<u8>> {
        let bytes = match self {
            Self::Local(store) => store
                .object(reference, authorization_scope)
                .map_err(protocol)?,
            Self::Pool(pool) => pool
                .execution_object(reference)?
                .map(|bytes| bytes.as_ref().to_vec()),
            Self::None => None,
        };
        bytes.ok_or_else(|| protocol("result object is absent from its execution owner"))
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
