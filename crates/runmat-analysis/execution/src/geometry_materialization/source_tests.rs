use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;
use runmat_geometry_core::{
    GeometryAuthoringCapabilities, GeometryDefinition, GeometryDefinitionSource, GeometryDigest,
    GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef,
    GeometryScalarExpression, GeometrySourceFormat, ImportedGeometryDefinition, LengthUnit,
};
use runmat_geometry_ops::{GeometryFeatureCapabilities, GeometryFeatureKind};

use super::*;

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "geometry:test".into(),
        encryption_context: Digest::sha256(b"geometry-test-key"),
    }
}

fn provider() -> GeometryProviderProfile {
    GeometryProviderProfile {
        provider_id: "exact-test".into(),
        provider_version: "1".into(),
        kernel_abi: "exact-test-abi".into(),
        capabilities: GeometryFeatureCapabilities {
            supported: BTreeSet::from([GeometryFeatureKind::Box]),
            maximum_features: 100,
            maximum_instances_per_feature: 100,
            maximum_boolean_operands: 10,
        },
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn scalar(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

#[test]
fn constructed_source_prepares_without_transport_objects() {
    let definition = GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "part".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![GeometryFeature {
                feature_id: "body".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: scalar(1.0),
                    depth_m: scalar(1.0),
                    height_m: scalar(1.0),
                },
            }],
            result: GeometryFeatureRef("body".into()),
        }),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    };
    let source = PreparedGeometrySource::constructed();
    let request = prepare_geometry_materialization_request(
        definition,
        BTreeMap::new(),
        &source,
        provider(),
        Default::default(),
    )
    .unwrap();

    assert!(source.objects.is_empty());
    assert!(source.roots.is_empty());
    assert!(matches!(
        request.source,
        GeometryMaterializationSource::Constructed
    ));
}

#[test]
fn imported_source_is_chunked_and_bound_to_declared_bytes_and_units() {
    let bytes = b"0123456789";
    let digest = Digest::sha256(bytes);
    let definition = GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "part".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Imported(Box::new(ImportedGeometryDefinition {
            preview_extent_m: None,
            unit_evidence: Default::default(),
            logical_path: "project/part.brep".into(),
            format: GeometrySourceFormat::Brep,
            authored_units: Some(LengthUnit::Millimeter),
            content_digest: GeometryDigest::from_bytes(*digest.bytes()),
            topology_digest: GeometryDigest::from_bytes(*Digest::sha256(b"topology").bytes()),
            capabilities: GeometryAuthoringCapabilities {
                topology: true,
                persistent_ids: true,
                ..Default::default()
            },
            exchange: None,
            shared_interfaces: false,
        })),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    };
    let access = access();
    let source = PreparedGeometrySource::imported("part.brep", bytes, 4, &access).unwrap();
    let request = prepare_geometry_materialization_request(
        definition,
        BTreeMap::new(),
        &source,
        provider(),
        Default::default(),
    )
    .unwrap();

    assert_eq!(source.objects.len(), 3);
    assert_eq!(source.roots.len(), 3);
    assert_eq!(
        request.import_options.source_units,
        Some(LengthUnit::Millimeter)
    );
    let GeometryMaterializationSource::Imported {
        source_digest,
        encoded_length,
        chunks,
        ..
    } = &request.source
    else {
        panic!("imported definition must retain imported source evidence");
    };
    assert_eq!(*source_digest, digest);
    assert_eq!(*encoded_length, bytes.len() as u64);
    assert_eq!(chunks.len(), 3);
    assert!(chunks
        .iter()
        .zip(&source.roots)
        .all(|(chunk, root)| chunk.digest == root.logical_digest
            && chunk.encoded_length == root.encoded_length));
    let program =
        prepare_geometry_materialization_program(request, revision(), &access, &source.roots)
            .unwrap();
    assert_eq!(program.program.arguments.len(), 4);
}

#[test]
fn definition_and_prepared_source_kind_must_converge() {
    let bytes = b"brep-source";
    let digest = Digest::sha256(bytes);
    let definition = GeometryDefinition {
        module_path: "project/main.fea".into(),
        geometry_id: "part".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Imported(Box::new(ImportedGeometryDefinition {
            preview_extent_m: None,
            unit_evidence: Default::default(),
            logical_path: "project/part.brep".into(),
            format: GeometrySourceFormat::Brep,
            authored_units: Some(LengthUnit::Meter),
            content_digest: GeometryDigest::from_bytes(*digest.bytes()),
            topology_digest: GeometryDigest::from_bytes(*Digest::sha256(b"topology").bytes()),
            capabilities: GeometryAuthoringCapabilities {
                topology: true,
                persistent_ids: true,
                ..Default::default()
            },
            exchange: None,
            shared_interfaces: false,
        })),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    };

    let error = prepare_geometry_materialization_request(
        definition,
        BTreeMap::new(),
        &PreparedGeometrySource::constructed(),
        provider(),
        Default::default(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("source does not match its definition"));
}
