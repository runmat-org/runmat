use super::authoring::*;
use super::{GeometryDigest, GeometrySourceFormat, UnitSystem};

#[test]
fn imported_geometry_requires_exact_identity_and_persistent_topology() {
    let mut definition = GeometryDefinition {
        geometry_id: "bracket".into(),
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
                    width_m: 1.0,
                    depth_m: 1.0,
                    height_m: 1.0,
                },
            },
            GeometryFeature {
                feature_id: "cut".into(),
                operation: GeometryFeatureOperation::Transform {
                    source: GeometryFeatureRef("stock".into()),
                    matrix: [1.0; 16],
                },
            },
        ],
        result: GeometryFeatureRef("cut".into()),
    };
    assert!(graph.validate(None).is_ok());
    graph.features[1].operation = GeometryFeatureOperation::Transform {
        source: GeometryFeatureRef("future".into()),
        matrix: [0.0; 16],
    };
    assert!(graph.validate(None).is_err());
}
