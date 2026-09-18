use runmat_geometry_core::{
    GeometryFeatureGraph, GeometryFeatureOperation, GeometryScalarExpression,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::codec::feature_value;
use super::combine::{append_operation, unary_operation};
use super::{feature_type, invalid};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.box",
    category = "geometry",
    summary = "Create a typed exact box feature from SI dimensions.",
    keywords = "geometry,cad,box,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::BOX_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_box_builtin(width: f64, depth: f64, height: f64) -> BuiltinResult<Value> {
    primitive(GeometryFeatureOperation::Box {
        width_m: scalar(width),
        depth_m: scalar(depth),
        height_m: scalar(height),
    })
}

#[runtime_builtin(
    name = "geometry.cylinder",
    category = "geometry",
    summary = "Create a typed exact cylinder feature from SI dimensions and vectors.",
    keywords = "geometry,cad,cylinder,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::CYLINDER_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_cylinder_builtin(
    radius: f64,
    height: f64,
    axis: Value,
    origin: Value,
) -> BuiltinResult<Value> {
    primitive(GeometryFeatureOperation::Cylinder {
        radius_m: scalar(radius),
        height_m: scalar(height),
        axis: vector3(&axis, "geometry.cylinder", "axis")?.map(scalar),
        origin_m: vector3(&origin, "geometry.cylinder", "origin")?.map(scalar),
    })
}

#[runtime_builtin(
    name = "geometry.sphere",
    category = "geometry",
    summary = "Create a typed exact sphere feature from an SI radius and center.",
    keywords = "geometry,cad,sphere,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::SPHERE_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_sphere_builtin(radius: f64, center: Value) -> BuiltinResult<Value> {
    primitive(GeometryFeatureOperation::Sphere {
        radius_m: scalar(radius),
        center_m: vector3(&center, "geometry.sphere", "center")?.map(scalar),
    })
}

#[runtime_builtin(
    name = "geometry.transform",
    category = "geometry",
    summary = "Apply one exact affine transform to a typed geometry feature.",
    keywords = "geometry,cad,transform,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::TRANSFORM_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_transform_builtin(source: Value, matrix: Value) -> BuiltinResult<Value> {
    let graph = super::codec::feature_from_value(&source, "geometry.transform")?;
    let matrix = numeric_values(&matrix, "geometry.transform", "matrix")?;
    if matrix.len() != 16 {
        return Err(invalid(
            "geometry.transform",
            "matrix must contain exactly 16 column-major affine components",
        ));
    }
    let operation = GeometryFeatureOperation::Transform {
        source: graph.result.clone(),
        matrix: matrix.into_iter().map(scalar).collect(),
    };
    feature_value(unary_operation(graph, operation)?)
}

#[runtime_builtin(
    name = "geometry.instance",
    category = "geometry",
    summary = "Create exact occurrences of a typed feature at one or more affine transforms.",
    keywords = "geometry,cad,instance,occurrence,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::INSTANCE_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_instance_builtin(
    source: Value,
    transforms: Vec<Value>,
) -> BuiltinResult<Value> {
    let builtin = "geometry.instance";
    if transforms.is_empty() {
        return Err(invalid(builtin, "requires at least one affine transform"));
    }
    let graph = super::codec::feature_from_value(&source, builtin)?;
    let transforms = transforms
        .into_iter()
        .map(|transform| {
            let values = numeric_values(&transform, builtin, "transform")?;
            if values.len() != 16 {
                return Err(invalid(
                    builtin,
                    "each transform must contain exactly 16 column-major affine components",
                ));
            }
            Ok(values.into_iter().map(scalar).collect())
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let operation = GeometryFeatureOperation::Instance {
        source: graph.result.clone(),
        transforms,
    };
    feature_value(unary_operation(graph, operation)?)
}

fn primitive(operation: GeometryFeatureOperation) -> BuiltinResult<Value> {
    feature_value(append_operation(
        GeometryFeatureGraph {
            features: Vec::new(),
            result: runmat_geometry_core::GeometryFeatureRef(String::new()),
        },
        operation,
    )?)
}

pub(super) fn scalar(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

pub(super) fn vector3(value: &Value, builtin: &'static str, name: &str) -> BuiltinResult<[f64; 3]> {
    let values = numeric_values(value, builtin, name)?;
    values
        .try_into()
        .map_err(|_| invalid(builtin, format!("{name} must contain exactly three values")))
}

pub(super) fn numeric_values(
    value: &Value,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) if value.is_finite() => Ok(vec![*value]),
        Value::Int(value) if value.to_f64().abs() <= 9_007_199_254_740_992.0 => {
            Ok(vec![value.to_f64()])
        }
        Value::Int(_) => Err(invalid(
            builtin,
            format!("{name} is not exactly representable as binary64"),
        )),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            let values = tensor.materialize_f64();
            if values.iter().all(|value| value.is_finite()) {
                Ok(values)
            } else {
                Err(invalid(builtin, format!("{name} must be finite")))
            }
        }
        _ => Err(invalid(builtin, format!("{name} must be real numeric"))),
    }
}
