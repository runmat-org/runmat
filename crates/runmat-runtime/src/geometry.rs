use std::collections::BTreeMap;

use runmat_geometry_core::GeometryAsset;
use runmat_geometry_io::{
    import::GeometryImportError, import_geometry, GeometryFormat, GeometryImportOptions,
};
use runmat_geometry_ops::{compute_stats, GeometryStats};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};
use crate::{build_runtime_error, BuiltinResult};

const GEOMETRY_INSPECT_OPERATION: &str = "geometry.inspect";
const GEOMETRY_INSPECT_OP_VERSION: &str = "geometry.inspect/v1";
const GEOMETRY_LOAD_OPERATION: &str = "geometry.load";
const GEOMETRY_LOAD_OP_VERSION: &str = "geometry.load/v1";
const GEOMETRY_COMPUTE_STATS_OPERATION: &str = "geometry.compute_stats";
const GEOMETRY_COMPUTE_STATS_OP_VERSION: &str = "geometry.compute_stats/v1";

#[derive(Debug, Clone)]
pub struct GeometryInspectResult {
    pub format: String,
    pub byte_count: usize,
}

pub fn geometry_inspect_op(
    path: &str,
    bytes: &[u8],
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryInspectResult>, OperationErrorEnvelope> {
    let format = runmat_geometry_io::detect_geometry_format(path, bytes);
    let data = GeometryInspectResult {
        format: format_name(format).to_string(),
        byte_count: bytes.len(),
    };
    Ok(OperationEnvelope::new(
        GEOMETRY_INSPECT_OPERATION,
        GEOMETRY_INSPECT_OP_VERSION,
        &context,
        data,
    ))
}

pub fn geometry_inspect(path: &str, bytes: &[u8]) -> BuiltinResult<GeometryInspectResult> {
    let envelope =
        geometry_inspect_op(path, bytes, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_INSPECT_OPERATION)
                .with_identifier("RunMat:GeometryInspectFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_load_op(
    path: &str,
    bytes: &[u8],
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryAsset>, OperationErrorEnvelope> {
    let imported = import_geometry(path, bytes, GeometryImportOptions::default())
        .map_err(|error| map_geometry_load_error(path, error, &context))?;
    Ok(OperationEnvelope::new(
        GEOMETRY_LOAD_OPERATION,
        GEOMETRY_LOAD_OP_VERSION,
        &context,
        imported.asset,
    ))
}

pub fn geometry_load(path: &str, bytes: &[u8]) -> BuiltinResult<GeometryAsset> {
    let envelope =
        geometry_load_op(path, bytes, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_LOAD_OPERATION)
                .with_identifier("RunMat:GeometryLoadFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_compute_stats_op(
    asset: &GeometryAsset,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryStats>, OperationErrorEnvelope> {
    Ok(OperationEnvelope::new(
        GEOMETRY_COMPUTE_STATS_OPERATION,
        GEOMETRY_COMPUTE_STATS_OP_VERSION,
        &context,
        compute_stats(asset),
    ))
}

pub fn geometry_compute_stats(asset: &GeometryAsset) -> BuiltinResult<GeometryStats> {
    let envelope =
        geometry_compute_stats_op(asset, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_COMPUTE_STATS_OPERATION)
                .with_identifier("RunMat:GeometryStatsFailed")
                .build()
        })?;
    Ok(envelope.data)
}

fn format_name(format: GeometryFormat) -> &'static str {
    match format {
        runmat_geometry_io::GeometryFormat::Stl => "stl",
        runmat_geometry_io::GeometryFormat::Step => "step",
        runmat_geometry_io::GeometryFormat::Obj => "obj",
        runmat_geometry_io::GeometryFormat::Ply => "ply",
        runmat_geometry_io::GeometryFormat::Gltf => "gltf",
        runmat_geometry_io::GeometryFormat::Unknown => "unknown",
    }
}

fn map_geometry_load_error(
    path: &str,
    error: GeometryImportError,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    let (error_code, error_type, retryable) = match &error {
        GeometryImportError::UnsupportedFormat => (
            "GEOMETRY_FORMAT_UNSUPPORTED",
            OperationErrorType::Input,
            false,
        ),
        GeometryImportError::ParseFailed(_) => (
            "GEOMETRY_PARSE_FAILED",
            OperationErrorType::Validation,
            false,
        ),
        GeometryImportError::CapacityExceeded { .. } => (
            "CAPACITY_LIMIT_EXCEEDED",
            OperationErrorType::Capacity,
            false,
        ),
    };
    operation_error(
        GEOMETRY_LOAD_OPERATION,
        GEOMETRY_LOAD_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable,
            severity: OperationErrorSeverity::Error,
        },
        error.to_string(),
        BTreeMap::from([("path".to_string(), path.to_string())]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

    #[test]
    fn inspect_detects_stl() {
        let result =
            geometry_inspect("/part.stl", TRIANGLE_STL.as_bytes()).expect("inspect should work");
        assert_eq!(result.format, "stl");
    }

    #[test]
    fn inspect_op_returns_typed_metadata() {
        let envelope = geometry_inspect_op(
            "/part.stl",
            TRIANGLE_STL.as_bytes(),
            OperationContext::new(Some("trace-g1".to_string()), Some("request-g1".to_string())),
        )
        .expect("inspect envelope should work");

        assert_eq!(envelope.operation, "geometry.inspect");
        assert_eq!(envelope.op_version, "geometry.inspect/v1");
        assert_eq!(envelope.trace_id.as_deref(), Some("trace-g1"));
        assert_eq!(envelope.data.format, "stl");
    }

    #[test]
    fn load_and_stats_work_for_stl() {
        let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
        let stats = geometry_compute_stats(&asset).expect("stats should work");
        assert_eq!(stats.mesh_count, 1);
        assert_eq!(stats.total_elements, 1);
    }

    #[test]
    fn inspect_and_load_step_work() {
        let inspect = geometry_inspect("/assembly.step", SIMPLE_STEP.as_bytes())
            .expect("inspect should work");
        assert_eq!(inspect.format, "step");

        let asset =
            geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
        assert_eq!(asset.source.importer_version, "step/v1");
        assert_eq!(asset.regions.len(), 1);
        assert!(asset.source_geometry.assembly.is_some());
    }

    #[test]
    fn load_op_maps_unsupported_format_error_code() {
        let error = geometry_load_op(
            "/bad.bin",
            b"not geometry",
            OperationContext::new(None, None),
        )
        .expect_err("load should fail");

        assert_eq!(error.error_code, "GEOMETRY_FORMAT_UNSUPPORTED");
        assert_eq!(error.operation, "geometry.load");
        assert_eq!(error.op_version, "geometry.load/v1");
    }
}
