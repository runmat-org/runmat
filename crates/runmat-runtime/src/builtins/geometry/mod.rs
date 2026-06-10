use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, ObjectInstance, Value,
};
use runmat_geometry_core::GeometryAsset;
use runmat_macros::runtime_builtin;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub const GEOMETRY_ASSET_CLASS: &str = "geometry.Asset";
const GEOMETRY_INSPECT_RESULT_CLASS: &str = "geometry.InspectResult";
pub const GEOMETRY_ASSET_JSON_PROPERTY: &str = "__runmat_geometry_asset_json";
const GEOMETRY_LOAD_NAME: &str = "geometry.load";
const GEOMETRY_INSPECT_NAME: &str = "geometry.inspect";
const GEOMETRY_LIST_REGIONS_NAME: &str = "geometry.listRegions";

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
    label: "asset = geometry.load(path)",
    inputs: &PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];
const GEOMETRY_INSPECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = geometry.inspect(path)",
    inputs: &PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];
const GEOMETRY_LIST_REGIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "regions = geometry.listRegions(asset)",
        inputs: &STRUCT_OUTPUT,
        outputs: &STRUCT_OUTPUT,
    }];

const GEOMETRY_LOAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.IO",
    identifier: Some("RunMat:geometry:load:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry.load: failed to read geometry file",
};
const GEOMETRY_LOAD_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.OPERATION_FAILED",
    identifier: Some("RunMat:geometry:load:OperationFailed"),
    when: "The geometry load operation rejects or cannot import the file.",
    message: "geometry.load: operation failed",
};
const GEOMETRY_LOAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LOAD.INTERNAL",
    identifier: Some("RunMat:geometry:load:Internal"),
    when: "The loaded geometry asset cannot be converted to a RunMat value.",
    message: "geometry.load: internal error",
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
    code: "RM.GEOMETRY.INSPECT.IO",
    identifier: Some("RunMat:geometry:inspect:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry.inspect: failed to read geometry file",
};
const GEOMETRY_INSPECT_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.OPERATION_FAILED",
    identifier: Some("RunMat:geometry:inspect:OperationFailed"),
    when: "The geometry inspection operation fails.",
    message: "geometry.inspect: operation failed",
};
const GEOMETRY_INSPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.INTERNAL",
    identifier: Some("RunMat:geometry:inspect:Internal"),
    when: "The inspection result cannot be converted to a RunMat value.",
    message: "geometry.inspect: internal error",
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
pub const GEOMETRY_LIST_REGIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_LIST_REGIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_INSPECT_ERRORS,
};

#[runtime_builtin(
    name = "geometry.load",
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
        Some(GEOMETRY_ASSET_CLASS),
        Some(GEOMETRY_ASSET_JSON_PROPERTY),
    )
}

#[runtime_builtin(
    name = "geometry.inspect",
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
        Some(GEOMETRY_INSPECT_RESULT_CLASS),
        None,
    )
}

#[runtime_builtin(
    name = "geometry.listRegions",
    category = "geometry",
    summary = "List regions imported into a geometry asset.",
    keywords = "geometry,regions,cad,selectors,fea",
    descriptor(crate::builtins::geometry::GEOMETRY_LIST_REGIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_list_regions_builtin(asset: Value) -> BuiltinResult<Value> {
    let asset = geometry_asset_from_value(&asset)?;
    operation_result_to_value(
        GEOMETRY_LIST_REGIONS_NAME,
        &GEOMETRY_INSPECT_ERROR_OPERATION,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
        crate::geometry::geometry_list_regions_op(&asset, OperationContext::new(None, None)),
        None,
        None,
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

fn geometry_asset_from_value(value: &Value) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            GEOMETRY_LIST_REGIONS_NAME,
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            "geometry.listRegions: expected geometry.Asset",
        ));
    };
    if object.class_name != GEOMETRY_ASSET_CLASS {
        return Err(builtin_error(
            GEOMETRY_LIST_REGIONS_NAME,
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            format!(
                "geometry.listRegions: expected {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    object_json_property(
        GEOMETRY_LIST_REGIONS_NAME,
        object,
        GEOMETRY_ASSET_JSON_PROPERTY,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
    )
}

fn object_json_property<T: DeserializeOwned>(
    builtin: &'static str,
    object: &ObjectInstance,
    property: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    let Some(Value::String(json)) = object.properties.get(property) else {
        return Err(build_runtime_error(format!(
            "{} is missing required runtime payload property `{property}`",
            object.class_name
        ))
        .with_builtin(builtin)
        .with_identifier(error.identifier.unwrap_or("RunMat:geometry:Internal"))
        .build());
    };
    serde_json::from_str(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

fn operation_result_to_value<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
    class_name: Option<&'static str>,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    match class_name {
        Some(class_name) => serializable_to_object(
            builtin,
            internal_error_descriptor,
            class_name,
            &envelope.data,
            hidden_json_property,
        ),
        None => serializable_to_value(builtin, internal_error_descriptor, &envelope.data),
    }
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

fn serializable_to_object<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    ensure_geometry_classes_registered();
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let converted = value_from_json(&json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.message().to_string(), err))?;
    let mut object = ObjectInstance::new(class_name.to_string());
    if let Value::Struct(fields) = converted {
        object.properties = fields.fields.into_iter().collect();
    } else {
        object.properties.insert("value".to_string(), converted);
    }
    if let Some(property) = hidden_json_property {
        object
            .properties
            .insert(property.to_string(), Value::String(json.to_string()));
    }
    Ok(Value::Object(object))
}

fn ensure_geometry_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        runmat_builtins::register_class(ClassDef {
            name: GEOMETRY_ASSET_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: geometry_asset_methods(),
        });
        runmat_builtins::register_class(ClassDef {
            name: GEOMETRY_INSPECT_RESULT_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::<String, MethodDef>::new(),
        });
    });
}

fn geometry_asset_methods() -> HashMap<String, MethodDef> {
    [("listRegions", GEOMETRY_LIST_REGIONS_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
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

fn builtin_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(error.identifier.unwrap_or("RunMat:geometry:Internal"))
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
    fn geometry_inspect_builtin_returns_object_value() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

        let value = block_on(geometry_inspect_builtin(path.to_string_lossy().to_string()))
            .expect("inspect builtin should return an object");

        let Value::Object(result) = value else {
            panic!("expected object value");
        };
        assert_eq!(result.class_name, GEOMETRY_INSPECT_RESULT_CLASS);
        assert!(result.properties.contains_key("format"));
        assert!(result.properties.contains_key("byte_count"));
    }

    #[test]
    fn geometry_list_regions_builtin_returns_imported_regions() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.step");
        std::fs::write(
            &path,
            "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n",
        )
        .unwrap();

        let asset = block_on(geometry_load_builtin(path.to_string_lossy().to_string()))
            .expect("geometry should load");
        let regions =
            block_on(geometry_list_regions_builtin(asset)).expect("regions should be listed");

        let Value::Struct(result) = regions else {
            panic!("expected struct value");
        };
        assert!(result.fields.contains_key("regions"));
    }
}
