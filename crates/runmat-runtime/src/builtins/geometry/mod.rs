use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_geometry_core::{GeometryAsset, UnitSystem};
use runmat_geometry_io::GeometryImportOptions;
use runmat_macros::runtime_builtin;
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, StructValue, Tensor, Value};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;

use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub mod generated;
pub mod region;
pub mod triangulation;

pub const GEOMETRY_ASSET_CLASS: &str = "geometry.Asset";
const GEOMETRY_INSPECT_RESULT_CLASS: &str = "geometry.InspectResult";
pub const GEOMETRY_ASSET_JSON_PROPERTY: &str = "__runmat_geometry_asset_json";
const GEOMETRY_LOAD_NAME: &str = "geometry.load";
const GEOMETRY_INSPECT_NAME: &str = "geometry.inspect";
const GEOMETRY_LIST_REGIONS_NAME: &str = "geometry.listRegions";
const GEOMETRY_MESHES_NAME: &str = "geometry.meshes";
const GEOMETRY_MESH_METHOD_NAME: &str = "__runmat_geometry_mesh";

static GEOMETRY_CLASSES_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(GEOMETRY_ASSET_CLASS);

pub const GEOMETRY_LIST_REGIONS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "geometry.listRegions accepts only a geometry.Asset object and returns imported-region metadata; it has no direct integer data or control argument.",
    };
pub const GEOMETRY_MESHES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "geometry.meshes accepts only a geometry.Asset object and returns host mesh structs; vertices, 1-based faces/triangles, and region ranges are deliberate double-valued API outputs rather than a typed-integer input surface.",
    };
pub const GEOMETRY_MESH_METHOD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "mesh(asset, ...) accepts a geometry.Asset and logical display controls; it has no integer input role.",
    };

const STRUCT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operation result as a struct.",
}];
const FIGURE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "figure",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure handle.",
}];
const PATH_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to the geometry file.",
}];
const LOAD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to the geometry file.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional Units Name, Value pair.",
    },
];

const GEOMETRY_LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "asset = geometry.load(path, Name, Value, ...)",
    inputs: &LOAD_INPUTS,
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
const GEOMETRY_MESHES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "meshes = geometry.meshes(asset)",
    inputs: &STRUCT_OUTPUT,
    outputs: &STRUCT_OUTPUT,
}];
const GEOMETRY_MESH_DISPLAY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "asset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Loaded geometry.Asset object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Geometry display options: Edges and XRay.",
    },
];
const GEOMETRY_MESH_DISPLAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "figure = mesh(asset, Name, Value, ...)",
        inputs: &GEOMETRY_MESH_DISPLAY_INPUTS,
        outputs: &FIGURE_OUTPUT,
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
pub const GEOMETRY_MESHES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_MESHES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GEOMETRY_INSPECT_ERRORS,
};
pub const GEOMETRY_MESH_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRY_MESH_DISPLAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
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
pub async fn geometry_load_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (path, options) = args.split_first().ok_or_else(|| {
        builtin_error(
            GEOMETRY_LOAD_NAME,
            &GEOMETRY_LOAD_ERROR_INTERNAL,
            "geometry.load requires a path",
        )
    })?;
    let path = scalar_text(path, GEOMETRY_LOAD_NAME)?;
    let import_options = parse_geometry_load_options(options)?;
    let bytes = read_file(GEOMETRY_LOAD_NAME, &GEOMETRY_LOAD_ERROR_IO, &path).await?;
    operation_result_to_value(
        GEOMETRY_LOAD_NAME,
        &GEOMETRY_LOAD_ERROR_OPERATION,
        &GEOMETRY_LOAD_ERROR_INTERNAL,
        crate::geometry::geometry_load_with_options_op(
            &path,
            &bytes,
            import_options,
            OperationContext::new(None, None),
        ),
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
    integer_audit(crate::builtins::geometry::GEOMETRY_LIST_REGIONS_INTEGER_AUDIT),
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

#[runtime_builtin(
    name = "geometry.meshes",
    category = "geometry",
    summary = "Return renderable surface mesh topology for a geometry asset.",
    keywords = "geometry,mesh,vertices,triangles,faces,patch,fea",
    descriptor(crate::builtins::geometry::GEOMETRY_MESHES_DESCRIPTOR),
    integer_audit(crate::builtins::geometry::GEOMETRY_MESHES_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_meshes_builtin(asset: Value) -> BuiltinResult<Value> {
    let asset = geometry_asset_from_value_with_builtin(&asset, GEOMETRY_MESHES_NAME)?;
    geometry_meshes_value(&asset)
}

#[runtime_builtin(
    name = "__runmat_geometry_mesh",
    category = "geometry",
    summary = "Implement mesh display for geometry.Asset objects.",
    keywords = "geometry,mesh,display,cad",
    descriptor(crate::builtins::geometry::GEOMETRY_MESH_METHOD_DESCRIPTOR),
    integer_audit(crate::builtins::geometry::GEOMETRY_MESH_METHOD_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry"
)]
pub async fn geometry_mesh_method_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let (asset, options) = args.split_first().ok_or_else(|| {
            builtin_error(
                "mesh",
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                "mesh requires a geometry.Asset",
            )
        })?;
        let asset = geometry_asset_from_value_with_builtin(asset, "mesh")?;
        let (show_edges, xray) = parse_geometry_mesh_options(options)?;
        let mut preview_options = crate::geometry::GeometryPreviewFigureOptions::cad_preview();
        preview_options.edge_overlay_triangle_limit = if show_edges { usize::MAX } else { 0 };
        preview_options.xray = xray;
        let figure = crate::geometry::geometry_preview_figure(&asset, "Geometry", preview_options)
            .map_err(|message| builtin_error("mesh", &GEOMETRY_INSPECT_ERROR_OPERATION, message))?;
        let handle = crate::builtins::plotting::import_runtime_figure(figure);
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            "mesh",
            &GEOMETRY_INSPECT_ERROR_OPERATION,
            "mesh display requires the plot-core runtime feature",
        ))
    }
}

fn parse_geometry_load_options(args: &[Value]) -> BuiltinResult<GeometryImportOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            GEOMETRY_LOAD_NAME,
            &GEOMETRY_LOAD_ERROR_INTERNAL,
            "geometry.load options must be Name, Value pairs",
        ));
    }
    let mut options = GeometryImportOptions::default();
    let mut units_seen = false;
    for pair in args.chunks(2) {
        let key = scalar_text(&pair[0], GEOMETRY_LOAD_NAME)?
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        if key != "units" || units_seen {
            let detail = if units_seen && key == "units" {
                "duplicate Units option"
            } else {
                "unsupported geometry.load option"
            };
            return Err(builtin_error(
                GEOMETRY_LOAD_NAME,
                &GEOMETRY_LOAD_ERROR_INTERNAL,
                detail,
            ));
        }
        units_seen = true;
        options.units = match scalar_text(&pair[1], GEOMETRY_LOAD_NAME)?
            .to_ascii_lowercase()
            .as_str()
        {
            "m" | "meter" | "meters" | "metre" | "metres" => UnitSystem::Meter,
            "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => {
                UnitSystem::Millimeter
            }
            "in" | "inch" | "inches" => UnitSystem::Inch,
            other => {
                return Err(builtin_error(
                    GEOMETRY_LOAD_NAME,
                    &GEOMETRY_LOAD_ERROR_INTERNAL,
                    format!("unsupported geometry unit `{other}`"),
                ))
            }
        };
    }
    Ok(options)
}

#[cfg(feature = "plot-core")]
fn parse_geometry_mesh_options(args: &[Value]) -> BuiltinResult<(bool, bool)> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            "mesh",
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            "mesh options must be Name, Value pairs",
        ));
    }
    let mut show_edges = true;
    let mut xray = false;
    for pair in args.chunks(2) {
        let key = scalar_text(&pair[0], "mesh")?.to_ascii_lowercase();
        let Value::Bool(value) = pair[1] else {
            return Err(builtin_error(
                "mesh",
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                format!("{key} must be logical"),
            ));
        };
        match key.as_str() {
            "edges" => show_edges = value,
            "xray" => xray = value,
            other => {
                return Err(builtin_error(
                    "mesh",
                    &GEOMETRY_INSPECT_ERROR_INTERNAL,
                    format!("unsupported geometry mesh option `{other}`"),
                ))
            }
        }
    }
    Ok((show_edges, xray))
}

fn scalar_text(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::CharArray(value) if value.rows == 1 => Ok(value.data.iter().collect()),
        _ => Err(builtin_error(
            builtin,
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            "expected a text scalar",
        )),
    }
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
    geometry_asset_from_value_with_builtin(value, GEOMETRY_LIST_REGIONS_NAME)
}

pub(crate) fn geometry_asset_from_value_with_builtin(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            format!("{builtin}: expected geometry.Asset"),
        ));
    };
    if object.class_name != GEOMETRY_ASSET_CLASS {
        return Err(builtin_error(
            builtin,
            &GEOMETRY_INSPECT_ERROR_INTERNAL,
            format!(
                "{builtin}: expected {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    object_json_property(
        builtin,
        object,
        GEOMETRY_ASSET_JSON_PROPERTY,
        &GEOMETRY_INSPECT_ERROR_INTERNAL,
    )
}

fn geometry_meshes_value(asset: &GeometryAsset) -> BuiltinResult<Value> {
    let values = asset
        .surface_meshes
        .iter()
        .map(|surface| {
            let mut mesh = StructValue::new();
            mesh.insert("mesh_id", Value::String(surface.mesh_id.clone()));
            mesh.insert("vertices", vertices_tensor(&surface.vertices)?);
            mesh.insert("triangles", triangles_tensor(&surface.triangles)?);
            mesh.insert("faces", triangles_tensor(&surface.triangles)?);
            mesh.insert(
                "region_mappings",
                region_mappings_value(asset, &surface.mesh_id)?,
            );
            Ok(Value::Struct(mesh))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    crate::make_cell_with_shape(values, vec![1, asset.surface_meshes.len()])
        .map_err(|err| builtin_error(GEOMETRY_MESHES_NAME, &GEOMETRY_INSPECT_ERROR_INTERNAL, err))
}

fn vertices_tensor(vertices: &[[f64; 3]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(vertices.len() * 3);
    for col in 0..3 {
        for vertex in vertices {
            data.push(vertex[col]);
        }
    }
    Tensor::new_2d(data, vertices.len(), 3)
        .map(Value::Tensor)
        .map_err(|err| {
            builtin_error(
                GEOMETRY_MESHES_NAME,
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                format!("geometry.meshes: failed to build vertices tensor: {err}"),
            )
        })
}

fn triangles_tensor(triangles: &[[u32; 3]]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(triangles.len() * 3);
    for col in 0..3 {
        for triangle in triangles {
            data.push(f64::from(triangle[col]) + 1.0);
        }
    }
    Tensor::new_2d(data, triangles.len(), 3)
        .map(Value::Tensor)
        .map_err(|err| {
            builtin_error(
                GEOMETRY_MESHES_NAME,
                &GEOMETRY_INSPECT_ERROR_INTERNAL,
                format!("geometry.meshes: failed to build triangle tensor: {err}"),
            )
        })
}

fn region_mappings_value(asset: &GeometryAsset, mesh_id: &str) -> BuiltinResult<Value> {
    let values = asset
        .region_entity_mappings
        .iter()
        .filter(|mapping| mapping.mesh_id == mesh_id)
        .map(|mapping| {
            let mut value = StructValue::new();
            value.insert("region_id", Value::String(mapping.region_id.clone()));
            value.insert(
                "entity_kind",
                Value::String(format!("{:?}", mapping.entity_kind).to_ascii_lowercase()),
            );
            let mut ranges = Vec::with_capacity(mapping.ranges.len() * 2);
            for col in 0..2 {
                for range in &mapping.ranges {
                    ranges.push(if col == 0 {
                        range.start as f64 + 1.0
                    } else {
                        range.count as f64
                    });
                }
            }
            let range_tensor = Tensor::new_2d(ranges, mapping.ranges.len(), 2)
                .map(Value::Tensor)
                .map_err(|err| {
                    builtin_error(
                        GEOMETRY_MESHES_NAME,
                        &GEOMETRY_INSPECT_ERROR_INTERNAL,
                        format!("geometry.meshes: failed to build range tensor: {err}"),
                    )
                })?;
            value.insert("ranges", range_tensor);
            Ok(Value::Struct(value))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let cols = values.len();
    crate::make_cell_with_shape(values, vec![1, cols])
        .map_err(|err| builtin_error(GEOMETRY_MESHES_NAME, &GEOMETRY_INSPECT_ERROR_INTERNAL, err))
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
    GEOMETRY_CLASSES_REGISTERED.ensure(|| {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: GEOMETRY_ASSET_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: geometry_asset_methods(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: GEOMETRY_INSPECT_RESULT_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::<String, crate::class_registry::RuntimeMethod>::new(),
        });
        triangulation::register_delaunaytri_class();
    });
}

fn geometry_asset_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    [
        ("mesh", GEOMETRY_MESH_METHOD_NAME),
        ("listRegions", GEOMETRY_LIST_REGIONS_NAME),
        ("meshes", GEOMETRY_MESHES_NAME),
        ("region", region::GEOMETRY_REGION_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            crate::class_registry::RuntimeMethod {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
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
    use runmat_value::Value;

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

        let asset = block_on(geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .expect("geometry should load");
        let regions =
            block_on(geometry_list_regions_builtin(asset)).expect("regions should be listed");

        let Value::Struct(result) = regions else {
            panic!("expected struct value");
        };
        assert!(result.fields.contains_key("regions"));
    }

    #[test]
    fn geometry_meshes_builtin_returns_patch_ready_surface_topology() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

        let asset = block_on(geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .expect("geometry should load");
        let meshes = block_on(geometry_meshes_builtin(asset)).expect("meshes should project");

        let Value::Cell(cell) = meshes else {
            panic!("expected cell array of mesh structs");
        };
        assert_eq!(cell.data.len(), 1);
        let Value::Struct(mesh) = &cell.data[0] else {
            panic!("expected mesh struct");
        };
        let Some(Value::Tensor(vertices)) = mesh.fields.get("vertices") else {
            panic!("expected vertices tensor");
        };
        assert_eq!(vertices.shape, vec![3, 3]);
        let Some(Value::Tensor(faces)) = mesh.fields.get("faces") else {
            panic!("expected faces tensor");
        };
        assert_eq!(faces.shape, vec![1, 3]);
        assert_eq!(faces.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn geometry_load_builtin_preserves_explicit_source_units_and_coordinates() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

        let value = block_on(geometry_load_builtin(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::String("Units".to_string()),
            Value::String("millimeter".to_string()),
        ]))
        .expect("geometry should load with explicit units");
        let asset = geometry_asset_from_value_with_builtin(&value, GEOMETRY_LOAD_NAME)
            .expect("asset should decode");

        assert_eq!(asset.units, UnitSystem::Millimeter);
        // The preview asset retains source coordinates with its unit tag;
        // canonical faceted/exact topology converts them to meters later.
        assert_eq!(asset.surface_meshes[0].vertices[1][0], 1.0);
        assert_eq!(asset.source.sha256.len(), 64);
        assert!(asset
            .source
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn geometry_asset_registers_standard_mesh_method() {
        ensure_geometry_classes_registered();
        let (method, _) = crate::class_registry::lookup_method(GEOMETRY_ASSET_CLASS, "mesh")
            .expect("geometry.Asset should overload mesh");
        assert_eq!(method.function_name, GEOMETRY_MESH_METHOD_NAME);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn standard_mesh_dispatches_geometry_assets_through_object_protocol() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();
        let asset = block_on(geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .expect("geometry should load");

        let handle = block_on(crate::call_builtin_async("mesh", &[asset]))
            .expect("standard mesh should dispatch geometry.Asset");
        assert!(matches!(handle, Value::Num(value) if value > 0.0));
    }
}
