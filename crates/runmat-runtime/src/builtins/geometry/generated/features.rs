use runmat_geometry_core::{
    GeometryFeatureOperation, GeometryFeatureRegionSelection, GeometryRegionRole,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::args::{angle, exactly, length, position, vector3};
use super::codec::{feature_value, region_output_from_value};
use super::combine::{append_operation, merge_values, unary_operation};
use super::invalid;
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.extrude",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_extrude_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.extrude";
    let args = exactly(builtin, args, 3)?;
    let graph = super::codec::feature_from_value(&args[0], builtin)?;
    let operation = GeometryFeatureOperation::Extrude {
        profile: graph.result.clone(),
        distance_m: length(&args[1], builtin, "distance")?,
        direction: vector3(&args[2], builtin, "direction")?,
    };
    feature_value(unary_operation(graph, operation, builtin)?, builtin)
}

#[runtime_builtin(
    name = "geometry.revolve",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_revolve_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.revolve";
    let args = exactly(builtin, args, 4)?;
    let graph = super::codec::feature_from_value(&args[0], builtin)?;
    let angle = angle(&args[3], builtin, "angle")?;
    let operation = GeometryFeatureOperation::Revolve {
        profile: graph.result.clone(),
        axis: vector3(&args[1], builtin, "axis")?,
        origin_m: position(&args[2], builtin, "origin")?,
        angle_rad: angle,
    };
    feature_value(unary_operation(graph, operation, builtin)?, builtin)
}

#[runtime_builtin(
    name = "geometry.hole",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_hole_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.hole";
    let args = exactly(builtin, args, 5)?;
    let graph = super::codec::feature_from_value(&args[0], builtin)?;
    let operation = GeometryFeatureOperation::Hole {
        target: graph.result.clone(),
        radius_m: length(&args[1], builtin, "radius")?,
        depth_m: length(&args[2], builtin, "depth")?,
        axis: vector3(&args[3], builtin, "axis")?,
        origin_m: position(&args[4], builtin, "origin")?,
    };
    feature_value(unary_operation(graph, operation, builtin)?, builtin)
}

#[runtime_builtin(
    name = "geometry.pocket",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_pocket_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.pocket";
    let mut args = exactly(builtin, args, 4)?;
    let direction = vector3(&args[3], builtin, "direction")?;
    let depth = length(&args[2], builtin, "depth")?;
    args.truncate(2);
    let (graph, operands) = merge_values(args, builtin)?;
    let operation = GeometryFeatureOperation::Pocket {
        target: operands[0].clone(),
        profile: operands[1].clone(),
        depth_m: depth,
        direction,
    };
    feature_value(append_operation(graph, operation, builtin)?, builtin)
}

#[runtime_builtin(
    name = "geometry.fillet",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_fillet_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.fillet";
    let args = exactly(builtin, args, 3)?;
    let radius = length(&args[2], builtin, "radius")?;
    edge_treatment(&args[0], &args[1], builtin, |target, edges| {
        GeometryFeatureOperation::Fillet {
            target,
            edges,
            radius_m: radius,
        }
    })
}

#[runtime_builtin(
    name = "geometry.chamfer",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_chamfer_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.chamfer";
    let args = exactly(builtin, args, 3)?;
    let distance = length(&args[2], builtin, "distance")?;
    edge_treatment(&args[0], &args[1], builtin, |target, edges| {
        GeometryFeatureOperation::Chamfer {
            target,
            edges,
            distance_m: distance,
        }
    })
}

fn edge_treatment(
    target: &Value,
    edges: &Value,
    builtin: &'static str,
    operation: impl FnOnce(
        runmat_geometry_core::GeometryFeatureRef,
        GeometryFeatureRegionSelection,
    ) -> GeometryFeatureOperation,
) -> BuiltinResult<Value> {
    let graph = super::codec::feature_from_value(target, builtin)?;
    let edges = region_output_from_value(edges, builtin)?;
    if edges.role != GeometryRegionRole::Edge {
        return Err(invalid(builtin, "selection must contain edges"));
    }
    let operation = operation(
        graph.result.clone(),
        GeometryFeatureRegionSelection {
            role: edges.role,
            selector: edges.selector,
            cardinality: edges.cardinality,
        },
    );
    feature_value(unary_operation(graph, operation, builtin)?, builtin)
}
