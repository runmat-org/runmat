use runmat_geometry_core::{
    GeometryFeatureOperation, GeometryFeatureRegionSelection, GeometryRegionRole,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::codec::{feature_value, region_output_from_value};
use super::combine::{append_operation, merge_values, unary_operation};
use super::primitives::{scalar, vector3};
use super::{feature_type, invalid};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.extrude",
    category = "geometry",
    summary = "Extrude an exact profile along a finite direction.",
    keywords = "geometry,cad,extrude,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::EXTRUDE_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_extrude_builtin(
    profile: Value,
    distance: f64,
    direction: Value,
) -> BuiltinResult<Value> {
    let builtin = "geometry.extrude";
    let graph = super::codec::feature_from_value(&profile, builtin)?;
    let operation = GeometryFeatureOperation::Extrude {
        profile: graph.result.clone(),
        distance_m: scalar(distance),
        direction: vector3(&direction, builtin, "direction")?.map(scalar),
    };
    feature_value(unary_operation(graph, operation)?)
}

#[runtime_builtin(
    name = "geometry.revolve",
    category = "geometry",
    summary = "Revolve an exact profile about a finite axis and origin.",
    keywords = "geometry,cad,revolve,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::REVOLVE_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_revolve_builtin(
    profile: Value,
    axis: Value,
    origin: Value,
    angle: f64,
) -> BuiltinResult<Value> {
    let builtin = "geometry.revolve";
    let graph = super::codec::feature_from_value(&profile, builtin)?;
    let operation = GeometryFeatureOperation::Revolve {
        profile: graph.result.clone(),
        axis: vector3(&axis, builtin, "axis")?.map(scalar),
        origin_m: vector3(&origin, builtin, "origin")?.map(scalar),
        angle_rad: scalar(angle),
    };
    feature_value(unary_operation(graph, operation)?)
}

#[runtime_builtin(
    name = "geometry.hole",
    category = "geometry",
    summary = "Cut an exact cylindrical hole from a target feature.",
    keywords = "geometry,cad,hole,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::HOLE_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_hole_builtin(
    target: Value,
    radius: f64,
    depth: f64,
    axis: Value,
    origin: Value,
) -> BuiltinResult<Value> {
    let builtin = "geometry.hole";
    let graph = super::codec::feature_from_value(&target, builtin)?;
    let operation = GeometryFeatureOperation::Hole {
        target: graph.result.clone(),
        radius_m: scalar(radius),
        depth_m: scalar(depth),
        axis: vector3(&axis, builtin, "axis")?.map(scalar),
        origin_m: vector3(&origin, builtin, "origin")?.map(scalar),
    };
    feature_value(unary_operation(graph, operation)?)
}

#[runtime_builtin(
    name = "geometry.pocket",
    category = "geometry",
    summary = "Sweep a profile to cut an exact pocket from a target feature.",
    keywords = "geometry,cad,pocket,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::POCKET_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_pocket_builtin(
    target: Value,
    profile: Value,
    depth: f64,
    direction: Value,
) -> BuiltinResult<Value> {
    let builtin = "geometry.pocket";
    let (graph, operands) = merge_values(vec![target, profile], builtin)?;
    let operation = GeometryFeatureOperation::Pocket {
        target: operands[0].clone(),
        profile: operands[1].clone(),
        depth_m: scalar(depth),
        direction: vector3(&direction, builtin, "direction")?.map(scalar),
    };
    feature_value(append_operation(graph, operation)?)
}

#[runtime_builtin(
    name = "geometry.fillet",
    category = "geometry",
    summary = "Round an exact, typed edge selection on a target feature.",
    keywords = "geometry,cad,fillet,edges,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::FILLET_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_fillet_builtin(
    target: Value,
    edges: Value,
    radius: f64,
) -> BuiltinResult<Value> {
    edge_treatment(target, edges, "geometry.fillet", |target, edges| {
        GeometryFeatureOperation::Fillet {
            target,
            edges,
            radius_m: scalar(radius),
        }
    })
}

#[runtime_builtin(
    name = "geometry.chamfer",
    category = "geometry",
    summary = "Bevel an exact, typed edge selection on a target feature.",
    keywords = "geometry,cad,chamfer,edges,generated",
    type_resolver(feature_type),
    descriptor(super::metadata::CHAMFER_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::features"
)]
pub async fn geometry_chamfer_builtin(
    target: Value,
    edges: Value,
    distance: f64,
) -> BuiltinResult<Value> {
    edge_treatment(target, edges, "geometry.chamfer", |target, edges| {
        GeometryFeatureOperation::Chamfer {
            target,
            edges,
            distance_m: scalar(distance),
        }
    })
}

fn edge_treatment(
    target: Value,
    edges: Value,
    builtin: &'static str,
    operation: impl FnOnce(
        runmat_geometry_core::GeometryFeatureRef,
        GeometryFeatureRegionSelection,
    ) -> GeometryFeatureOperation,
) -> BuiltinResult<Value> {
    let graph = super::codec::feature_from_value(&target, builtin)?;
    let edges = region_output_from_value(&edges, builtin)?;
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
    feature_value(unary_operation(graph, operation)?)
}
