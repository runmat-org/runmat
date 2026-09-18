use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    build_geometry_task_submission, import_geometry_materialization_output,
    prepare_geometry_materialization_program, GeometryExecutionContext,
    GeometryMaterializationRequest, GeometryMaterializationSource, GeometrySourceChunk,
    DEFAULT_GEOMETRY_MAXIMUM_ATTEMPTS,
};
use runmat_execution::identity::ArtifactId;
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, LogicalObject, ObjectNamespace, ProgramExecutionResponse,
};
use runmat_geometry_core::{
    DerivedGeometryDefinition, GeometryAuthoringCapabilities, GeometryDefinition,
    GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest, GeometryFeature,
    GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef, GeometryRegionCardinality,
    GeometryRegionDefinition, GeometryRegionRole, GeometryRegionSelector, GeometryScalarExpression,
    GeometrySourceFormat, ImportedGeometryDefinition,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_geometry_ops::GeometryFeaturePlanLimits;

use super::*;

#[derive(Default)]
struct MemoryStore(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for MemoryStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

impl CacheExport for MemoryStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        self.0
            .insert(object.descriptor.digest, object.bytes.clone());
        Ok(())
    }
}

#[test]
fn constructed_and_derived_geometry_publish_reimportable_exact_closures() {
    let profile = native_geometry_provider_profile().expect("OCCT profile");
    assert_eq!(
        profile.capabilities.supported,
        BTreeSet::from([
            runmat_geometry_ops::GeometryFeatureKind::Box,
            runmat_geometry_ops::GeometryFeatureKind::Cylinder,
            runmat_geometry_ops::GeometryFeatureKind::Sphere,
            runmat_geometry_ops::GeometryFeatureKind::Transform,
            runmat_geometry_ops::GeometryFeatureKind::Instance,
            runmat_geometry_ops::GeometryFeatureKind::Union,
            runmat_geometry_ops::GeometryFeatureKind::Subtract,
            runmat_geometry_ops::GeometryFeatureKind::Intersect,
        ])
    );
    let request = constructed_request(profile);
    let access = access("geometry:test", b"geometry-test-key");
    let prepared =
        prepare_geometry_materialization_program(request.clone(), revision(), &access, &[])
            .unwrap();
    let submission = build_geometry_task_submission(
        &prepared,
        BTreeSet::new(),
        GeometryExecutionContext {
            scope_id: ExecutionScopeId::derive(&[b"geometry-scope"]),
            pool_id: PoolId::derive(&[b"geometry-pool"]),
            program_artifact_id: ArtifactId::derive(&[prepared.program.artifact.id.0.bytes()]),
            deadline_unix_millis: None,
            priority: 0,
            maximum_attempts: DEFAULT_GEOMETRY_MAXIMUM_ATTEMPTS,
        },
    )
    .unwrap();
    assert!(submission.request.resources.required_capabilities.contains(
        &runmat_analysis_execution::geometry_provider_capability(&request.provider)
    ));
    let mut store = MemoryStore::default();
    prepared.write_to(&mut store).unwrap();
    let response = execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        panic!("geometry materialization failed: {response:?}");
    };
    assert!(result_objects.len() >= 4);
    let ValuePayload::Object(output_root) = &outputs[0] else {
        panic!("output must be externalized");
    };
    let output = import_geometry_materialization_output(&store, output_root, &access).unwrap();
    output.validate_against(&request).unwrap();
    assert_eq!(output.regions["body"].entity_ids.len(), 1);
    assert_eq!(exact_solid_count(&output.document.model), Some(1));

    let exact_source_roots = result_objects
        .into_iter()
        .filter(|reference| reference.logical_digest != output_root.logical_digest)
        .collect::<Vec<_>>();
    let derived_request = derived_request(&request, &output);
    let prepared = prepare_geometry_materialization_program(
        derived_request.clone(),
        revision(),
        &access,
        &exact_source_roots,
    )
    .unwrap();
    prepared.write_to(&mut store).unwrap();
    let response = execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        panic!("derived geometry materialization failed: {response:?}");
    };
    let ValuePayload::Object(output_root) = &outputs[0] else {
        panic!("derived output must be externalized");
    };
    let derived_output =
        import_geometry_materialization_output(&store, output_root, &access).unwrap();
    derived_output.validate_against(&derived_request).unwrap();
    assert_eq!(derived_output.regions["body"].entity_ids.len(), 1);
    assert!(derived_output.feature_history.contains_key("translated"));
}

#[test]
fn imported_brep_is_verified_against_source_and_topology_identity() {
    const BOX: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");
    let options = ExactCadImportOptions::default();
    let admitted = runmat_geometry_io::import_exact_cad(
        "box.brep",
        BOX,
        runmat_geometry_io::GeometryFormat::Brep,
        &options,
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .unwrap();
    let source_digest = Digest::sha256(BOX);
    let request = imported_request(
        options,
        source_digest,
        admitted.build_closure().unwrap().manifest.topology.digest,
        BOX.len() as u64,
    );
    let access = access("geometry:import-test", b"geometry-import-test-key");
    let source = LogicalObject::new(
        ObjectNamespace::InputValue,
        "geometry/sources/box.brep",
        "application/vnd.runmat.geometry-source",
        BOX.to_vec(),
    )
    .unwrap();
    let source_root = access
        .reference(
            &source,
            runmat_execution::value::ValueRefKind::DriverObject,
            "runmat.geometry.source-chunk",
        )
        .unwrap();
    let prepared = prepare_geometry_materialization_program(
        request.clone(),
        revision(),
        &access,
        &[source_root],
    )
    .unwrap();
    let mut store = MemoryStore::default();
    store.write_verified(&source).unwrap();
    prepared.write_to(&mut store).unwrap();
    let response = execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        panic!("imported geometry materialization failed: {response:?}");
    };
    let ValuePayload::Object(output_root) = &outputs[0] else {
        panic!("imported output must be externalized");
    };
    let output = import_geometry_materialization_output(&store, output_root, &access).unwrap();
    output.validate_against(&request).unwrap();
    assert_eq!(output.regions["body"].entity_ids.len(), 1);
}

fn constructed_request(
    provider: runmat_analysis_execution::GeometryProviderProfile,
) -> GeometryMaterializationRequest {
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/constructed.fea".into(),
            geometry_id: "bracket".into(),
            parameters: Vec::new(),
            source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
                features: vec![GeometryFeature {
                    feature_id: "stock".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: scalar(1.0),
                        depth_m: scalar(2.0),
                        height_m: scalar(3.0),
                    },
                }],
                result: GeometryFeatureRef("stock".into()),
            }),
            regions: body_region(),
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::new(),
        source: GeometryMaterializationSource::Constructed,
        import_options: ExactCadImportOptions::default(),
        provider,
        plan_limits: GeometryFeaturePlanLimits::default(),
        resources: Default::default(),
    }
}

fn derived_request(
    source: &GeometryMaterializationRequest,
    output: &runmat_analysis_execution::GeometryMaterializationOutput,
) -> GeometryMaterializationRequest {
    let source_definition = GeometryDefinitionRef {
        module_path: source.definition.module_path.clone(),
        geometry_id: source.definition.geometry_id.clone(),
    };
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/derived.fea".into(),
            geometry_id: "translated_bracket".into(),
            parameters: Vec::new(),
            source: GeometryDefinitionSource::Derived(DerivedGeometryDefinition {
                source: source_definition.clone(),
                graph: GeometryFeatureGraph {
                    features: vec![GeometryFeature {
                        feature_id: "translated".into(),
                        operation: GeometryFeatureOperation::Transform {
                            source: GeometryFeatureRef("source.result".into()),
                            matrix: (0..16)
                                .map(|index| {
                                    scalar(if matches!(index, 0 | 5 | 10 | 15) {
                                        1.0
                                    } else if index == 3 {
                                        2.0
                                    } else {
                                        0.0
                                    })
                                })
                                .collect(),
                        },
                    }],
                    result: GeometryFeatureRef("translated".into()),
                },
            }),
            regions: body_region(),
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::new(),
        source: GeometryMaterializationSource::Derived {
            definition: source_definition,
            document: Box::new(output.document.clone()),
            root: output.document.primary_artifact().clone(),
        },
        import_options: ExactCadImportOptions::default(),
        provider: source.provider.clone(),
        plan_limits: GeometryFeaturePlanLimits::default(),
        resources: Default::default(),
    }
}

fn imported_request(
    import_options: ExactCadImportOptions,
    source_digest: Digest,
    topology_digest: GeometryDigest,
    encoded_length: u64,
) -> GeometryMaterializationRequest {
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/imported.fea".into(),
            geometry_id: "box".into(),
            parameters: Vec::new(),
            source: GeometryDefinitionSource::Imported(ImportedGeometryDefinition {
                logical_path: "box.brep".into(),
                format: GeometrySourceFormat::Brep,
                units: runmat_geometry_core::UnitSystem::Meter,
                content_digest: GeometryDigest::from_bytes(*source_digest.bytes()),
                topology_digest,
                capabilities: GeometryAuthoringCapabilities {
                    topology: true,
                    persistent_ids: true,
                    ..Default::default()
                },
            }),
            regions: body_region(),
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::new(),
        source: GeometryMaterializationSource::Imported {
            source_name: "box.brep".into(),
            source_digest,
            encoded_length,
            chunks: vec![GeometrySourceChunk {
                ordinal: 0,
                digest: source_digest,
                encoded_length,
            }],
        },
        import_options,
        provider: native_geometry_provider_profile().unwrap(),
        plan_limits: GeometryFeaturePlanLimits::default(),
        resources: Default::default(),
    }
}

fn body_region() -> Vec<GeometryRegionDefinition> {
    vec![GeometryRegionDefinition {
        region_id: "body".into(),
        role: GeometryRegionRole::Solid,
        selector: GeometryRegionSelector::Result,
        cardinality: GeometryRegionCardinality::ExactlyOne,
    }]
}

fn scalar(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

fn access(scope: &str, key: &[u8]) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: scope.into(),
        encryption_context: Digest::sha256(key),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"geometry-graph"),
        Digest::sha256(b"geometry-source"),
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

fn exact_solid_count(model: &runmat_geometry_core::GeometryModel) -> Option<u64> {
    match model {
        runmat_geometry_core::GeometryModel::ExactBRep { model } => Some(model.solid_count),
        runmat_geometry_core::GeometryModel::FacetedSolid { .. } => None,
    }
}
