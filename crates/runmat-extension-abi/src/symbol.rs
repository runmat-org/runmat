use crate::{RunMatAbiVersion, RunMatExtensionVTable, RunMatStatusCode};

pub const RUNMAT_EXTENSION_QUERY_SYMBOL: &str = "runmat_extension_query_v1";
pub const RUNMAT_EXTENSION_QUERY_SYMBOL_NUL: &[u8] = b"runmat_extension_query_v1\0";

pub type RunMatExtensionQueryFn = unsafe extern "C" fn(
    host_version: RunMatAbiVersion,
    extension: *mut RunMatExtensionVTable,
) -> RunMatStatusCode;
