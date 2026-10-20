use std::collections::{BTreeMap, HashMap};

use runmat_geometry_core::{
    GeneratedGeometryProduct, GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation,
    GeometryFeatureRef,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::args::exactly;
use super::codec::{feature_from_value, feature_value, generated_value};
use super::invalid;
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.union",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::combine"
)]
pub async fn geometry_union_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    boolean(args, "geometry.union", |operands| {
        GeometryFeatureOperation::Union { operands }
    })
}

#[runtime_builtin(
    name = "geometry.intersect",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::combine"
)]
pub async fn geometry_intersect_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    boolean(args, "geometry.intersect", |operands| {
        GeometryFeatureOperation::Intersect { operands }
    })
}

#[runtime_builtin(
    name = "geometry.subtract",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::combine"
)]
pub async fn geometry_subtract_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    if args.len() < 2 {
        return Err(invalid(
            "geometry.subtract",
            "requires one base and at least one tool",
        ));
    }
    let builtin = "geometry.subtract";
    let (graph, operands) = merge_values(args, builtin)?;
    let operation = GeometryFeatureOperation::Subtract {
        base: operands[0].clone(),
        tools: operands[1..].to_vec(),
    };
    feature_value(append_operation(graph, operation, builtin)?, builtin)
}

#[runtime_builtin(
    name = "geometry.finish",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::combine"
)]
pub async fn geometry_finish_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.finish";
    let args = exactly(builtin, args, 1)?;
    generated_value(
        GeneratedGeometryProduct {
            graph: feature_from_value(&args[0], builtin)?,
            region_outputs: BTreeMap::new(),
        },
        builtin,
    )
}

fn boolean(
    args: Vec<Value>,
    builtin: &'static str,
    operation: impl FnOnce(Vec<GeometryFeatureRef>) -> GeometryFeatureOperation,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(invalid(builtin, "requires at least two operands"));
    }
    let (graph, operands) = merge_values(args, builtin)?;
    feature_value(
        append_operation(graph, operation(operands), builtin)?,
        builtin,
    )
}

pub(super) fn merge_values(
    values: Vec<Value>,
    builtin: &'static str,
) -> BuiltinResult<(GeometryFeatureGraph, Vec<GeometryFeatureRef>)> {
    let mut features = Vec::<GeometryFeature>::new();
    let mut feature_indices = HashMap::<String, usize>::new();
    let mut operands = Vec::with_capacity(values.len());
    for value in values {
        let graph = feature_from_value(&value, builtin)?;
        operands.push(graph.result);
        for feature in graph.features {
            match feature_indices.get(&feature.feature_id).copied() {
                Some(index) if features[index] != feature => {
                    return Err(invalid(
                        builtin,
                        "feature identity collision has conflicting definitions",
                    ));
                }
                Some(_) => {}
                None => {
                    feature_indices.insert(feature.feature_id.clone(), features.len());
                    features.push(feature);
                }
            }
        }
    }
    Ok((
        GeometryFeatureGraph {
            features,
            result: GeometryFeatureRef(String::new()),
        },
        operands,
    ))
}

pub(super) fn unary_operation(
    graph: GeometryFeatureGraph,
    operation: GeometryFeatureOperation,
    builtin: &'static str,
) -> BuiltinResult<GeometryFeatureGraph> {
    append_operation(graph, operation, builtin)
}

pub(super) fn append_operation(
    mut graph: GeometryFeatureGraph,
    operation: GeometryFeatureOperation,
    builtin: &'static str,
) -> BuiltinResult<GeometryFeatureGraph> {
    let digest = operation
        .canonical_digest()
        .map_err(|error| invalid(builtin, error.to_string()))?;
    let digest = digest.bytes();
    let feature_id = format!(
        "feature_{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7]
    );
    let feature = GeometryFeature {
        feature_id: feature_id.clone(),
        operation,
    };
    match graph
        .features
        .iter()
        .find(|candidate| candidate.feature_id == feature_id)
    {
        Some(existing) if existing != &feature => {
            return Err(invalid(
                builtin,
                "feature identity collision has conflicting definitions",
            ));
        }
        Some(_) => {}
        None => graph.features.push(feature),
    }
    graph.result = GeometryFeatureRef(feature_id);
    Ok(graph)
}
