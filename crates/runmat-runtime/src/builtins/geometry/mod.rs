use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use serde::Serialize;

use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const GEOMETRY_LOAD_NAME: &str = "geometry_load";
const GEOMETRY_INSPECT_NAME: &str = "geometry_inspect";

const STRUCT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operation result as a struct.",
}];
const PATH_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to the geometry file.",
}];

const GEOMETRY_LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "asset = geometry_load(path)",
    inputs: &PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];
const GEOMETRY_INSPECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = geometry_inspect(path)",
    inputs: &PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];

const GEOMETRY_LOAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY_LOAD.IO",
    identifier: Some("RunMat:geometry_load:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry_load: failed to read geometry file",
};
const GEOMETRY_LOAD_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY_LOAD.OPERATION_FAILED",
    identifier: Some("RunMat:geometry_load:OperationFailed"),
    when: "The geometry load operation rejects or cannot import the file.",
    message: "geometry_load: operation failed",
};
const GEOMETRY_LOAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY_LOAD.INTERNAL",
    identifier: Some("RunMat:geometry_load:Internal"),
    when: "The loaded geometry asset cannot be converted to a RunMat value.",
    message: "geometry_load: internal error",
};
const GEOMETRY_LOAD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GEOMETRY_LOAD_ERROR_IO,
    GEOMETRY_LOAD_ERROR_OPERATION,
    GEOMETRY_LOAD_ERROR_INTERNAL,
];
pub const GEOMETRY_LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_LOAD_ERRORS,
};

const GEOMETRY_INSPECT_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY_INSPECT.IO",
    identifier: Some("RunMat:geometry_inspect:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry_inspect: failed to read geometry file",
};
const GEOMETRY_INSPECT_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY_INSPECT.OPERATION_FAILED",
    identifier: Some("RunMat:geometry_inspect:OperationFailed"),
    when: "The geometry inspection operation fails.",
    message: "geometry_inspect: operation failed",
};
const GEOMETRY_INSPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY_INSPECT.INTERNAL",
    identifier: Some("RunMat:geometry_inspect:Internal"),
    when: "The inspection result cannot be converted to a RunMat value.",
    message: "geometry_inspect: internal error",
};
const GEOMETRY_INSPECT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GEOMETRY_INSPECT_ERROR_IO,
    GEOMETRY_INSPECT_ERROR_OPERATION,
    GEOMETRY_INSPECT_ERROR_INTERNAL,
];
pub const GEOMETRY_INSPECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_INSPECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_INSPECT_ERRORS,
};

#[runtime_builtin(
    name = "geometry_load",
    category = "geometry",
    summary = "Load a geometry file into a structured geometry asset.",
    keywords = "geometry,load,cad,mesh,stl,step,obj",
    descriptor(crate::builtins::geometry::GEOMETRY_LOAD_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_load_builtin(path: String) -> BuiltinResult<Value> {
    let bytes = read_file(GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_IO, &path).await?;
    operation_result_to_value(
        GEOMETRY_LOAD_NAME,
        &GEOMETRY_LOAD_ERROR_OPERATION,
        &GEOMETRY_LOAD_ERROR_INTERNAL,
        crate::geometry::geometry_load_op(&path, &bytes, OperationContext::new(None, None)),
    )
}

#[runtime_builtin(
    name = "geometry_inspect",
    category = "geometry",
    summary = "Inspect a geometry file without importing the full asset.",
    keywords = "geometry,inspect,cad,mesh,stl,step,obj",
    descriptor(crate::builtins::geometry::GEOMETRY_INSPECT_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_inspect_builtin(path: String) -> BuiltinResult<Value> {
    let bytes = read_file(GEOMETRY_INSPECT_NAME, &GEOMETRY_INSPECT_ERROR_IO, &path).await?;
    operation_result_to_value(
        GEOMETRY_INSPECT_NAME,
        &GEOMETRY_INSPECT_ERROR_OPERATION,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
        crate::geometry::geometry_inspect_op(&path, &bytes, OperationContext::new(None, None)),
    )
}

async fn read_file(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    path: &str,
) -> BuiltinResult<Vec<u8>> {
    runmat_filesystem::read_async(path)
        .await
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

fn operation_result_to_value<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_value(builtin, internal_error_descriptor, &envelope.data)
}

fn serializable_to_value<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &T,
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    value_from_json(&json)
}

fn operation_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    source: OperationErrorEnvelope,
) -> RuntimeError {
    let message = format!(
        "{}: {}: {}",
        error.message, source.error_code, source.message
    );
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or("RunMat:geometry:OperationFailed"),
        )
        .build()
}

fn builtin_error_with_source<E>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(error.identifier.unwrap_or("RunMat:geometry:Internal"))
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Value;

    #[test]
    fn geometry_inspect_builtin_returns_struct_value() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

        let value = block_on(geometry_inspect_builtin(path.to_string_lossy().to_string()))
            .expect("inspect builtin should return a struct");

        let Value::Struct(result) = value else {
            panic!("expected struct value");
        };
        assert!(result.fields.contains_key("format"));
        assert!(result.fields.contains_key("byte_count"));
    }
}
