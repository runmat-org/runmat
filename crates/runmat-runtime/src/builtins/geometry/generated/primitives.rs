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
    let builtin = "geometry.box";
    primitive(GeometryFeatureOperation::Box {
        width_m: physical_scalar(width, runmat_types::physical::QuantityKind::Length, builtin)?,
        depth_m: physical_scalar(depth, runmat_types::physical::QuantityKind::Length, builtin)?,
        height_m: physical_scalar(
            height,
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
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
    let builtin = "geometry.cylinder";
    primitive(GeometryFeatureOperation::Cylinder {
        radius_m: physical_scalar(
            radius,
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
        height_m: physical_scalar(
            height,
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
        axis: vector3(&axis, "geometry.cylinder", "axis")?.map(scalar),
        origin_m: position(vector3(&origin, builtin, "origin")?, builtin)?,
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
    let builtin = "geometry.sphere";
    primitive(GeometryFeatureOperation::Sphere {
        radius_m: physical_scalar(
            radius,
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
        center_m: position(vector3(&center, builtin, "center")?, builtin)?,
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
        matrix: affine_matrix(matrix, "geometry.transform")?,
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
            affine_matrix(values, builtin)
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

/// Public generated-geometry builtins accept canonical SI numerical MATLAB
/// values. The physical role is explicit before an authored graph is returned.
pub(super) fn physical_scalar(
    value: f64,
    kind: runmat_types::physical::QuantityKind,
    builtin: &'static str,
) -> BuiltinResult<GeometryScalarExpression> {
    let descriptor = runmat_types::physical::DynamicDimension::try_with_angle(
        kind.dimension(),
        kind.angle_power(),
        Some(kind),
    )
    .map_err(|error| invalid(builtin, error.to_string()))?;
    let value = runmat_types::physical::DimensionedScalar::try_from_descriptor(descriptor, value)
        .map_err(|error| invalid(builtin, error.to_string()))?;
    Ok(GeometryScalarExpression::Canonical { value })
}

pub(super) fn position(
    values: [f64; 3],
    builtin: &'static str,
) -> BuiltinResult<[GeometryScalarExpression; 3]> {
    Ok([
        physical_scalar(
            values[0],
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
        physical_scalar(
            values[1],
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
        physical_scalar(
            values[2],
            runmat_types::physical::QuantityKind::Length,
            builtin,
        )?,
    ])
}

fn affine_matrix(
    values: Vec<f64>,
    builtin: &'static str,
) -> BuiltinResult<Vec<GeometryScalarExpression>> {
    if values.len() != 16 {
        return Err(invalid(builtin, "affine transform requires 16 values"));
    }
    // MATLAB storage is column-major; the geometry/OCCT contract is row-major.
    (0..16)
        .map(|index| {
            let row = index / 4;
            let column = index % 4;
            let value = values[column * 4 + row];
            if matches!(index, 3 | 7 | 11) {
                physical_scalar(value, runmat_types::physical::QuantityKind::Length, builtin)
            } else {
                Ok(scalar(value))
            }
        })
        .collect()
}

#[cfg(test)]
mod quantity_boundary_tests {
    use super::*;
    #[test]
    fn matlab_column_major_translation_becomes_row_major_typed_lengths() {
        let source = vec![
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.012, 0.034, 0.056, 1.,
        ];
        let matrix = affine_matrix(source, "geometry.transform").unwrap();
        for (index, expected) in [(3, 0.012), (7, 0.034), (11, 0.056)] {
            let GeometryScalarExpression::Canonical { value } = &matrix[index] else {
                panic!("translation lost its units")
            };
            assert_eq!(value.canonical_value(), expected);
            assert_eq!(
                value.semantic_kind(),
                Some(runmat_types::physical::QuantityKind::Length)
            );
        }
        assert_eq!(matrix[12], scalar(0.0));
        assert!(physical_scalar(
            f64::NAN,
            runmat_types::physical::QuantityKind::Length,
            "geometry.box"
        )
        .is_err());
    }
}
