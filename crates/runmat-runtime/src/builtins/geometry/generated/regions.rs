use runmat_geometry_core::{
    GeneratedGeometryRegionOutput, GeometryRegionCardinality, GeometryRegionRole,
    GeometryRegionSelector,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::codec::{
    feature_from_value, generated_from_value, generated_value, region_output_from_value,
    region_output_value,
};
use super::{generated_type, invalid, region_output_type};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "geometry.solid",
    category = "geometry",
    summary = "Select the exactly-one solid result of a typed geometry feature.",
    keywords = "geometry,cad,region,solid,generated",
    type_resolver(region_output_type),
    descriptor(super::metadata::SOLID_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_solid_builtin(feature: Value) -> BuiltinResult<Value> {
    feature_output(
        feature,
        "geometry.solid",
        GeometryRegionRole::Solid,
        GeometryRegionCardinality::ExactlyOne,
        false,
    )
}

#[runtime_builtin(
    name = "geometry.faces",
    category = "geometry",
    summary = "Select one or more final faces belonging to a typed geometry feature result.",
    keywords = "geometry,cad,region,faces,generated",
    type_resolver(region_output_type),
    descriptor(super::metadata::FACES_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_faces_builtin(feature: Value) -> BuiltinResult<Value> {
    feature_output(
        feature,
        "geometry.faces",
        GeometryRegionRole::Face,
        GeometryRegionCardinality::OneOrMore,
        false,
    )
}

#[runtime_builtin(
    name = "geometry.facesIntroducedBy",
    category = "geometry",
    summary = "Select final faces proven to have been introduced by a typed feature.",
    keywords = "geometry,cad,region,faces,history,generated",
    type_resolver(region_output_type),
    descriptor(super::metadata::FACES_INTRODUCED_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_faces_introduced_by_builtin(feature: Value) -> BuiltinResult<Value> {
    feature_output(
        feature,
        "geometry.facesIntroducedBy",
        GeometryRegionRole::Face,
        GeometryRegionCardinality::OneOrMore,
        true,
    )
}

#[runtime_builtin(
    name = "geometry.edgesIntroducedBy",
    category = "geometry",
    summary = "Select final edges proven to have been introduced by a typed feature.",
    keywords = "geometry,cad,region,edges,history,generated",
    type_resolver(region_output_type),
    descriptor(super::metadata::EDGES_INTRODUCED_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_edges_introduced_by_builtin(feature: Value) -> BuiltinResult<Value> {
    feature_output(
        feature,
        "geometry.edgesIntroducedBy",
        GeometryRegionRole::Edge,
        GeometryRegionCardinality::OneOrMore,
        true,
    )
}

#[runtime_builtin(
    name = "geometry.withRegion",
    category = "geometry",
    summary = "Add one uniquely named typed region output to generated geometry.",
    keywords = "geometry,cad,region,output,generated",
    type_resolver(generated_type),
    descriptor(super::metadata::WITH_REGION_DESCRIPTOR),
    integer_audit(super::GENERATED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::generated::regions"
)]
pub async fn geometry_with_region_builtin(
    generated: Value,
    name: String,
    output: Value,
) -> BuiltinResult<Value> {
    let builtin = "geometry.withRegion";
    if name.trim().is_empty()
        || name.len() > 256
        || name.chars().any(|character| character.is_control())
    {
        return Err(invalid(builtin, "region name is invalid"));
    }
    let mut generated = generated_from_value(&generated, builtin)?;
    let output = region_output_from_value(&output, builtin)?;
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
    generated_value(generated)
}

fn feature_output(
    feature: Value,
    builtin: &'static str,
    role: GeometryRegionRole,
    cardinality: GeometryRegionCardinality,
    introduced: bool,
) -> BuiltinResult<Value> {
    let graph = feature_from_value(&feature, builtin)?;
    let selector = if introduced {
        GeometryRegionSelector::BoundaryIntroducedBy(graph.result)
    } else {
        GeometryRegionSelector::FeatureResult(graph.result)
    };
    region_output_value(&GeneratedGeometryRegionOutput {
        role,
        selector,
        cardinality,
    })
}
