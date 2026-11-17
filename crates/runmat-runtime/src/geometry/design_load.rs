//! `geometry.load` of a RunMat MCAD part or assembly from async callers.
//!
//! The asynchronous form reads the design's files through the filesystem provider's async API
//! and regenerates on the installed study-geometry provider (the browser) or the in-process
//! kernel (native), so a browser caller never blocks on a provider read.

use std::collections::BTreeMap;
use std::path::Path;

use runmat_geometry_core::GeometryAsset;
use runmat_geometry_io::GeometryImportOptions;

use super::{
    design_source, geometry_load_resolved_op, geometry_load_with_options_op,
    GEOMETRY_LOAD_OPERATION, GEOMETRY_LOAD_OP_VERSION,
};
use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

/// [`geometry_load_with_options_op`] for async callers; `bytes` are the file at `path`.
pub async fn geometry_load_with_options_op_async(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryAsset>, OperationErrorEnvelope> {
    if !design_source::is_design_geometry(path) {
        return geometry_load_with_options_op(path, bytes, options, context);
    }
    let design = design_source::design_geometry_async(Path::new(path), Some(bytes))
        .await
        .map_err(|message| design_load_error(path, &context, message))?;
    geometry_load_resolved_op(path, &design, options, context)
}

/// A part or assembly that does not regenerate.
pub(super) fn design_load_error(
    path: &str,
    context: &OperationContext,
    message: String,
) -> OperationErrorEnvelope {
    operation_error(
        GEOMETRY_LOAD_OPERATION,
        GEOMETRY_LOAD_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.GEOMETRY.LOAD.DESIGN_EVALUATION_FAILED",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        BTreeMap::from([("path".to_string(), path.to_string())]),
    )
}
