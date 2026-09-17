use super::authoring::*;
use super::{GeometryDigest, GeometryScalarExpression, GeometrySourceFormat, UnitSystem};

#[test]
fn imported_geometry_requires_exact_identity_and_persistent_topology() {
    let mut definition = GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "bracket".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Imported(ImportedGeometryDefinition {
            logical_path: "geometry/bracket.step".into(),
            format: GeometrySourceFormat::Step,
            units: UnitSystem::Millimeter,
            content_digest: GeometryDigest::from_bytes([1; 32]),
            topology_digest: GeometryDigest::from_bytes([2; 32]),
            capabilities: GeometryAuthoringCapabilities {
                topology: true,
                persistent_ids: true,
                ..GeometryAuthoringCapabilities::default()
            },
        }),
        regions: vec![GeometryRegionDefinition {
            region_id: "body".into(),
            role: GeometryRegionRole::Solid,
            selector: GeometryRegionSelector::Name("Bracket".into()),
            cardinality: GeometryRegionCardinality::ExactlyOne,
        }],
        cad_parameters: Vec::new(),
    };
    assert!(definition.validate().is_ok());
    let GeometryDefinitionSource::Imported(source) = &mut definition.source else {
        unreachable!()
    };
    source.capabilities.persistent_ids = false;
    assert!(definition.validate().is_err());
}

#[test]
fn feature_graph_is_ordered_and_rejects_forward_references() {
    let mut graph = GeometryFeatureGraph {
        features: vec![
            GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: 1.0.into(),
                    depth_m: 1.0.into(),
                    height_m: 1.0.into(),
                },
            },
            GeometryFeature {
                feature_id: "cut".into(),
                operation: GeometryFeatureOperation::Transform {
                    source: GeometryFeatureRef("stock".into()),
                    matrix: vec![1.0.into(); 16],
                },
            },
        ],
        result: GeometryFeatureRef("cut".into()),
    };
    assert!(graph.validate(None, &Default::default()).is_ok());
    graph.features[1].operation = GeometryFeatureOperation::Transform {
        source: GeometryFeatureRef("future".into()),
        matrix: vec![0.0.into(); 16],
    };
    assert!(graph.validate(None, &Default::default()).is_err());
}

#[test]
fn parameterized_feature_graph_retains_declared_input_identity() {
    let definition = GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "swept_bracket".into(),
        parameters: vec![GeometryParameterDefinition {
            parameter_id: "width".into(),
            parameter_type: GeometryParameterType::Length,
            required: true,
            default: None,
        }],
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: GeometryScalarExpression::Parameter {
                        name: "width".into(),
                    },
                    depth_m: 0.01.into(),
                    height_m: 0.05.into(),
                },
            }],
            result: GeometryFeatureRef("stock".into()),
        }),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    };
    assert!(definition.validate().is_ok());
}

#[test]
fn feature_region_selectors_must_resolve_in_the_authored_graph() {
    let definition = GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "part".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: 1.0.into(),
                    depth_m: 1.0.into(),
                    height_m: 1.0.into(),
                },
            }],
            result: GeometryFeatureRef("stock".into()),
        }),
        regions: vec![GeometryRegionDefinition {
            region_id: "missing_boundary".into(),
            role: GeometryRegionRole::Face,
            selector: GeometryRegionSelector::BoundaryIntroducedBy(GeometryFeatureRef(
                "missing".into(),
            )),
            cardinality: GeometryRegionCardinality::OneOrMore,
        }],
        cad_parameters: Vec::new(),
    };
    assert!(definition.validate().is_err());
}
