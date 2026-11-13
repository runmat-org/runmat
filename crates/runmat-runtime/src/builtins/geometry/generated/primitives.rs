use runmat_geometry_core::{GeometryFeatureGraph, GeometryFeatureOperation};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::args::{at_least, exactly, length, position, transform, vector3};
use super::codec::feature_value;
use super::combine::{append_operation, unary_operation};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.box",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_box_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.box";
    let args = exactly(builtin, args, 3)?;
    primitive(
        GeometryFeatureOperation::Box {
            width_m: length(&args[0], builtin, "width")?,
            depth_m: length(&args[1], builtin, "depth")?,
            height_m: length(&args[2], builtin, "height")?,
        },
        builtin,
    )
}

#[runtime_builtin(
    name = "geometry.cylinder",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_cylinder_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.cylinder";
    let args = exactly(builtin, args, 4)?;
    primitive(
        GeometryFeatureOperation::Cylinder {
            radius_m: length(&args[0], builtin, "radius")?,
            height_m: length(&args[1], builtin, "height")?,
            axis: vector3(&args[2], builtin, "axis")?,
            origin_m: position(&args[3], builtin, "origin")?,
        },
        builtin,
    )
}

#[runtime_builtin(
    name = "geometry.sphere",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_sphere_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.sphere";
    let args = exactly(builtin, args, 2)?;
    primitive(
        GeometryFeatureOperation::Sphere {
            radius_m: length(&args[0], builtin, "radius")?,
            center_m: position(&args[1], builtin, "center")?,
        },
        builtin,
    )
}

#[runtime_builtin(
    name = "geometry.transform",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_transform_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.transform";
    let args = exactly(builtin, args, 2)?;
    let graph = super::codec::feature_from_value(&args[0], builtin)?;
    let operation = GeometryFeatureOperation::Transform {
        source: graph.result.clone(),
        matrix: transform(&args[1], builtin, "matrix")?,
    };
    feature_value(unary_operation(graph, operation, builtin)?, builtin)
}

#[runtime_builtin(
    name = "geometry.instance",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::primitives"
)]
pub async fn geometry_instance_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.instance";
    let args = at_least(
        builtin,
        args,
        2,
        "requires a source feature and at least one affine transform",
    )?;
    let graph = super::codec::feature_from_value(&args[0], builtin)?;
    let transforms = args[1..]
        .iter()
        .map(|value| transform(value, builtin, "transform"))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let operation = GeometryFeatureOperation::Instance {
        source: graph.result.clone(),
        transforms,
    };
    feature_value(unary_operation(graph, operation, builtin)?, builtin)
}

fn primitive(operation: GeometryFeatureOperation, builtin: &'static str) -> BuiltinResult<Value> {
    feature_value(
        append_operation(
            GeometryFeatureGraph {
                features: Vec::new(),
                result: runmat_geometry_core::GeometryFeatureRef(String::new()),
            },
            operation,
            builtin,
        )?,
        builtin,
    )
}
