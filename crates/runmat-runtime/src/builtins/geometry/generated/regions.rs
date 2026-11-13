use runmat_geometry_core::{
    GeneratedGeometryRegionOutput, GeometryRegionCardinality, GeometryRegionRole,
    GeometryRegionSelector,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::args::{exactly, text};
use super::codec::{
    feature_from_value, generated_from_value, generated_value, region_output_from_value,
    region_output_value,
};
use super::invalid;
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.solid",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_solid_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    feature_output(
        args,
        "geometry.solid",
        GeometryRegionRole::Solid,
        GeometryRegionCardinality::ExactlyOne,
        false,
    )
}

#[runtime_builtin(
    name = "geometry.faces",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_faces_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    feature_output(
        args,
        "geometry.faces",
        GeometryRegionRole::Face,
        GeometryRegionCardinality::OneOrMore,
        false,
    )
}

#[runtime_builtin(
    name = "geometry.edges",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_edges_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    feature_output(
        args,
        "geometry.edges",
        GeometryRegionRole::Edge,
        GeometryRegionCardinality::OneOrMore,
        false,
    )
}

#[runtime_builtin(
    name = "geometry.facesIntroducedBy",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_faces_introduced_by_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    feature_output(
        args,
        "geometry.facesIntroducedBy",
        GeometryRegionRole::Face,
        GeometryRegionCardinality::OneOrMore,
        true,
    )
}

#[runtime_builtin(
    name = "geometry.edgesIntroducedBy",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_edges_introduced_by_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    feature_output(
        args,
        "geometry.edgesIntroducedBy",
        GeometryRegionRole::Edge,
        GeometryRegionCardinality::OneOrMore,
        true,
    )
}

#[runtime_builtin(
    name = "geometry.withRegion",
    binding_variant = "default",
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_with_region_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    let builtin = "geometry.withRegion";
    let args = exactly(builtin, args, 3)?;
    let name = text(&args[1], builtin, "name")?;
    let (generated, output) = (&args[0], &args[2]);
    if name.trim().is_empty()
        || name.len() > 256
        || name.chars().any(|character| character.is_control())
    {
        return Err(invalid(builtin, "region name is invalid"));
    }
    let mut generated = generated_from_value(generated, builtin)?;
    let output = region_output_from_value(output, builtin)?;
    if generated
        .region_outputs
        .insert(name.clone(), output)
        .is_some()
    {
        return Err(invalid(
            builtin,
            format!("region output `{name}` is already declared"),
        ));
    }
    generated_value(generated, builtin)
}

fn feature_output(
    args: Vec<Value>,
    builtin: &'static str,
    role: GeometryRegionRole,
    cardinality: GeometryRegionCardinality,
    introduced: bool,
) -> BuiltinResult<Value> {
    let args = exactly(builtin, args, 1)?;
    let graph = feature_from_value(&args[0], builtin)?;
    let selector = if introduced {
        GeometryRegionSelector::BoundaryIntroducedBy(graph.result)
    } else {
        GeometryRegionSelector::FeatureResult(graph.result)
    };
    region_output_value(
        &GeneratedGeometryRegionOutput {
            role,
            selector,
            cardinality,
        },
        builtin,
    )
}
