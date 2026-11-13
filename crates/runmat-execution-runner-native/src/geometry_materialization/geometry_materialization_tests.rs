use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    admit_named_geometry_generator, build_geometry_task_submission,
    import_geometry_materialization_output, prepare_geometry_materialization_program,
    GeneratedGeometryExecutionReceipt, GeometryExecutionContext, GeometryMaterializationRequest,
    GeometryMaterializationSource, GeometrySourceChunk, DEFAULT_GEOMETRY_MAXIMUM_ATTEMPTS,
    GENERATED_GEOMETRY_PRODUCT_MEDIA_TYPE, GENERATED_GEOMETRY_PRODUCT_VALUE_SCHEMA,
};
use runmat_execution::identity::ArtifactId;
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, BundleCallable, LogicalObject, ObjectNamespace,
    ProgramExecutionResponse,
};
use runmat_geometry_core::{
    DerivedGeometryDefinition, GeneratedGeometryDefinition, GeneratedGeometryProduct,
    GeneratedGeometryRegionOutput, GeometryAuthoringCapabilities, GeometryDefinition,
    GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest, GeometryFeature,
    GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef, GeometryRegionCardinality,
    GeometryRegionDefinition, GeometryRegionRole, GeometryRegionSelector, GeometryScalarExpression,
    GeometrySourceFormat, ImportedGeometryDefinition,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_geometry_ops::GeometryFeaturePlanLimits;
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{
    CallableFact, CapabilitySet, EffectKind, EffectSet, ObjectFact, ProgramFunctionId, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use super::*;

#[derive(Default)]
struct MemoryStore(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for MemoryStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded_with_admission(digest, maximum_bytes, &mut |_| Ok(()))
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        runmat_execution_artifact::cache::copy_verified_bounded_with_admission(
            digest,
            self.0.get(&digest).map(Vec::as_slice),
            maximum_bytes,
            admit,
        )
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
            runmat_geometry_ops::GeometryFeatureKind::Extrude,
            runmat_geometry_ops::GeometryFeatureKind::Revolve,
            runmat_geometry_ops::GeometryFeatureKind::Union,
            runmat_geometry_ops::GeometryFeatureKind::Subtract,
            runmat_geometry_ops::GeometryFeatureKind::Intersect,
            runmat_geometry_ops::GeometryFeatureKind::Hole,
            runmat_geometry_ops::GeometryFeatureKind::Pocket,
            runmat_geometry_ops::GeometryFeatureKind::Fillet,
            runmat_geometry_ops::GeometryFeatureKind::Chamfer,
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
            host: crate::test_support::host_requirement(),
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
fn constructed_geometry_program_is_preserved_by_the_meshing_root() {
    let request = constructed_request(native_geometry_provider_profile().expect("OCCT profile"));
    let access = access("geometry:mesh-root", b"geometry-mesh-root-key");
    let revision = revision();
    let geometry =
        prepare_geometry_materialization_program(request.clone(), revision.clone(), &access, &[])
            .unwrap();
    let meshing_request = crate::resolve_native_meshing_request(
        &request.import_options,
        runmat_meshing_core::MeshingRequestSettings::default(),
    )
    .unwrap();
    let prepared = crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
        program_revision: revision,
        artifact_access: access.clone(),
        geometry,
        geometry_source_objects: Vec::new(),
        meshing_request: meshing_request.clone(),
        preferred_edges_per_partition: 8,
        preferred_faces_per_partition: 8,
    })
    .unwrap();
    let mut store = MemoryStore::default();
    prepared.write_to(&mut store).unwrap();
    let decoded = crate::decode_geometry_meshing_root(prepared.request(), &store).unwrap();

    assert_eq!(decoded.artifact_access, access);
    assert_eq!(decoded.request.meshing_request, meshing_request);
    assert_eq!(decoded.geometry.arguments.len(), 1);
    assert_eq!(
        decoded.geometry.recipe.execution_mode,
        runmat_analysis_execution::GEOMETRY_MATERIALIZATION_EXECUTION_MODE
    );
    let programs = prepared.required_programs();
    assert_eq!(programs.len(), 3);
    assert_eq!(
        programs
            .iter()
            .map(|program| program.artifact.id)
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
}

#[test]
fn imported_brep_is_verified_against_source_and_topology_identity() {
    const BOX: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");
    let options = ExactCadImportOptions {
        source_units: Some(runmat_geometry_core::LengthUnit::Meter),
        source_units_authority: runmat_geometry_io::SourceUnitsAuthority::Authored,
        ..Default::default()
    };
    let admitted = runmat_geometry_io::import_exact_cad(
        "box.brep",
        BOX,
        runmat_geometry_io::GeometryFormat::Brep,
        &options,
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .unwrap();
    let expected_solid = admitted.topology.solids[0].id.clone();
    let expected_face = admitted.source_face_ids[&0].clone();
    let source_digest = Digest::sha256(BOX);
    let mut request = imported_request(
        options,
        source_digest,
        admitted.build_closure().unwrap().manifest.topology.digest,
        admitted.source_unit_evidence.clone(),
        BOX.len() as u64,
    );
    request.definition.regions = vec![
        GeometryRegionDefinition {
            region_id: "body".into(),
            role: GeometryRegionRole::Solid,
            selector: GeometryRegionSelector::PersistentId(
                expected_solid.analysis_selection_id().unwrap(),
            ),
            cardinality: GeometryRegionCardinality::ExactlyOne,
        },
        GeometryRegionDefinition {
            region_id: "support".into(),
            role: GeometryRegionRole::Face,
            selector: GeometryRegionSelector::PersistentId(
                expected_face.analysis_selection_id().unwrap(),
            ),
            cardinality: GeometryRegionCardinality::ExactlyOne,
        },
    ];
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
    assert_eq!(output.regions["body"].entity_ids, vec![expected_solid]);
    assert_eq!(output.regions["support"].entity_ids, vec![expected_face]);
}

#[test]
fn imported_ap242_is_verified_against_exact_exchange_inventory() {
    let bytes = ap242_box_source();
    // The source's AP242 context declares millimeters. Author the same unit, not a
    // reinterpretation of its original physical geometry as meters.
    let options = ExactCadImportOptions {
        source_units: Some(runmat_geometry_core::LengthUnit::Millimeter),
        source_units_authority: runmat_geometry_io::SourceUnitsAuthority::Authored,
        ..Default::default()
    };
    let admitted = runmat_geometry_io::import_exact_cad(
        "box.step",
        bytes.as_bytes(),
        runmat_geometry_io::GeometryFormat::Step,
        &options,
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .unwrap();
    let exchange = runmat_geometry_io::ap242::inspect_ap242(&bytes).unwrap();
    assert_eq!(exchange.parameters.len(), 1);
    let source_digest = Digest::sha256(bytes.as_bytes());
    let mut request = imported_request(
        options,
        source_digest,
        admitted.build_closure().unwrap().manifest.topology.digest,
        admitted.source_unit_evidence.clone(),
        bytes.len() as u64,
    );
    request.definition.regions[0].cardinality = GeometryRegionCardinality::OneOrMore;
    let GeometryDefinitionSource::Imported(imported) = &mut request.definition.source else {
        unreachable!()
    };
    imported.logical_path = "box.step".into();
    imported.format = GeometrySourceFormat::Step;
    imported.capabilities.parameters = true;
    imported.exchange = Some(exchange.clone());
    let GeometryMaterializationSource::Imported {
        source_name,
        chunks,
        ..
    } = &mut request.source
    else {
        unreachable!()
    };
    *source_name = "box.step".into();
    chunks[0].digest = source_digest;
    let response = execute_imported_request(&request, bytes.as_bytes());
    assert!(
        matches!(
            &response,
            ProgramExecutionResponse::ExternalizedSuccess { .. }
        ),
        "AP242 materialization failed: {response:?}"
    );

    let GeometryDefinitionSource::Imported(imported) = &mut request.definition.source else {
        unreachable!()
    };
    let parameter = &mut imported.exchange.as_mut().unwrap().parameters[0];
    parameter.value = runmat_types::physical::SourceQuantity::from_canonical(
        parameter.value.kind(),
        0.009,
        parameter.value.attribution().clone(),
    )
    .unwrap();
    let response = execute_imported_request(&request, bytes.as_bytes());
    assert!(format!("{response:?}").contains("exchange facts differ"));
}

#[test]
fn generated_geometry_requires_and_materializes_its_frozen_product_receipt() {
    let product = GeneratedGeometryProduct {
        graph: GeometryFeatureGraph {
            features: vec![
                GeometryFeature {
                    feature_id: "stock".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: scalar(2.0),
                        depth_m: scalar(2.0),
                        height_m: scalar(2.0),
                    },
                },
                GeometryFeature {
                    feature_id: "bore".into(),
                    operation: GeometryFeatureOperation::Cylinder {
                        radius_m: scalar(0.25),
                        height_m: scalar(4.0),
                        axis: [scalar(0.0), scalar(0.0), scalar(1.0)],
                        origin_m: [scalar(1.0), scalar(1.0), scalar(-1.0)],
                    },
                },
                GeometryFeature {
                    feature_id: "finished".into(),
                    operation: GeometryFeatureOperation::Subtract {
                        base: GeometryFeatureRef("stock".into()),
                        tools: vec![GeometryFeatureRef("bore".into())],
                    },
                },
            ],
            result: GeometryFeatureRef("finished".into()),
        },
        region_outputs: BTreeMap::from([
            (
                "body".into(),
                GeneratedGeometryRegionOutput {
                    role: GeometryRegionRole::Solid,
                    selector: GeometryRegionSelector::FeatureResult(GeometryFeatureRef(
                        "finished".into(),
                    )),
                    cardinality: GeometryRegionCardinality::ExactlyOne,
                },
            ),
            (
                "hole_wall".into(),
                GeneratedGeometryRegionOutput {
                    role: GeometryRegionRole::Face,
                    selector: GeometryRegionSelector::BoundaryIntroducedBy(GeometryFeatureRef(
                        "bore".into(),
                    )),
                    cardinality: GeometryRegionCardinality::ExactlyOne,
                },
            ),
        ]),
    };
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "geometry/generated/test",
        GENERATED_GEOMETRY_PRODUCT_MEDIA_TYPE,
        product.canonical_bytes().unwrap(),
    )
    .unwrap();
    let access = access("geometry:generated-test", b"geometry-generated-test-key");
    let product_root = access
        .reference(
            &object,
            runmat_execution::value::ValueRefKind::ResultObject,
            GENERATED_GEOMETRY_PRODUCT_VALUE_SCHEMA,
        )
        .unwrap();
    let source_digest = Digest::sha256(b"make-bored-block-source");
    let generated = GeneratedGeometryDefinition {
        module_path: "examples/makeBoredBlock.m".into(),
        function: "makeBoredBlock".into(),
        source_digest: GeometryDigest::from_bytes(*source_digest.bytes()),
        arguments: Vec::new(),
    };
    let callable = BundleCallable {
        owner_identity: "examples".into(),
        qualified_name: "makeBoredBlock".into(),
        source_digest,
    };
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::ClassIdentity::from("geometry.Generated")),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let function = FunctionAnalysis {
        function: ProgramFunctionId(4),
        captures: Vec::new(),
        callable: CallableFact {
            capabilities: Default::default(),
            identity: None,
            parameters: Vec::new(),
            parameters_complete: true,
            outputs: vec![output.clone()],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: true,
        },
        outputs: vec![output],
        effects: EffectSet(BTreeSet::from([EffectKind::MayThrow])),
        cooperative_suspension: false,
        capabilities: CapabilitySet::default(),
        convergence: FunctionConvergence::Exact,
    };
    let admission = admit_named_geometry_generator(
        runmat_geometry_ops::ResolvedGeneratedGeometryRequest {
            module_path: generated.module_path.clone(),
            function: generated.function.clone(),
            source_digest: generated.source_digest,
            arguments: Vec::new(),
        },
        revision(),
        callable,
        runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
            &runmat_execution::ExecutableIdentity {
                program: revision(),
                root_package: "examples".into(),
                entrypoint: "makeBoredBlock".into(),
                entrypoint_function: Some(function.function),
                entrypoint_kind: runmat_execution::ExecutableEntrypointKind::Function,
                declaration_class: None,
                source_digest: Digest::sha256(b"executable"),
            },
            "makeBoredBlock",
        )
        .unwrap(),
        &function,
        &BTreeSet::new(),
    )
    .unwrap();
    let receipt = GeneratedGeometryExecutionReceipt {
        schema_version: 1,
        admission,
        product_digest: product_root.logical_digest,
        product_encoded_length: product_root.encoded_length,
    };
    let request = GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/generated.fea".into(),
            geometry_id: "generated_bored_block".into(),
            parameters: Vec::new(),
            source: GeometryDefinitionSource::Generated(generated),
            regions: vec![
                GeometryRegionDefinition {
                    region_id: "body".into(),
                    role: GeometryRegionRole::Solid,
                    selector: GeometryRegionSelector::GeneratedOutput("body".into()),
                    cardinality: GeometryRegionCardinality::ExactlyOne,
                },
                GeometryRegionDefinition {
                    region_id: "hole_wall".into(),
                    role: GeometryRegionRole::Face,
                    selector: GeometryRegionSelector::GeneratedOutput("hole_wall".into()),
                    cardinality: GeometryRegionCardinality::ExactlyOne,
                },
            ],
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::new(),
        source: GeometryMaterializationSource::Generated {
            receipt: Box::new(receipt),
        },
        import_options: ExactCadImportOptions::default(),
        provider: native_geometry_provider_profile().unwrap(),
        plan_limits: GeometryFeaturePlanLimits::default(),
        resources: Default::default(),
    };
    let prepared = prepare_geometry_materialization_program(
        request.clone(),
        revision(),
        &access,
        std::slice::from_ref(&product_root),
    )
    .unwrap();
    let mut store = MemoryStore::default();
    store.write_verified(&object).unwrap();
    prepared.write_to(&mut store).unwrap();
    let response = execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        panic!("generated geometry materialization failed: {response:?}");
    };
    let ValuePayload::Object(output_root) = &outputs[0] else {
        panic!("generated output must be externalized");
    };
    let output = import_geometry_materialization_output(&store, output_root, &access).unwrap();
    output.validate_against(&request).unwrap();
    assert_eq!(output.regions["body"].entity_ids.len(), 1);
    assert_eq!(output.regions["hole_wall"].entity_ids.len(), 1);
    assert_eq!(
        output.materialization_kind,
        runmat_geometry_ops::GeometryMaterializationKind::Generated
    );
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
    unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence,
    encoded_length: u64,
) -> GeometryMaterializationRequest {
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/imported.fea".into(),
            geometry_id: "box".into(),
            parameters: Vec::new(),
            source: GeometryDefinitionSource::Imported(Box::new(ImportedGeometryDefinition {
                preview_extent_m: None,
                unit_evidence,
                logical_path: "box.brep".into(),
                format: GeometrySourceFormat::Brep,
                authored_units: import_options.source_units,
                content_digest: GeometryDigest::from_bytes(*source_digest.bytes()),
                topology_digest,
                capabilities: GeometryAuthoringCapabilities {
                    topology: true,
                    persistent_ids: true,
                    ..Default::default()
                },
                exchange: None,
                shared_interfaces: false,
            })),
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

fn execute_imported_request(
    request: &GeometryMaterializationRequest,
    bytes: &[u8],
) -> ProgramExecutionResponse {
    let access = access("geometry:ap242-test", b"geometry-ap242-test-key");
    let source = LogicalObject::new(
        ObjectNamespace::InputValue,
        "geometry/sources/box.step",
        "application/vnd.runmat.geometry-source",
        bytes.to_vec(),
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
    execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        NativeGeometryMaterializationLimits::default(),
    )
}

fn ap242_box_source() -> String {
    const SOURCE: &str =
        include_str!("../../../runmat-geometry/io/tests/fixtures/two_box_assembly.step");
    let source = SOURCE.replace(
        "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }",
        "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 0 1 10303 442 1 1 4 }",
    );
    source.replacen(
        "ENDSEC;\nEND-ISO-10303-21;",
        "#1000=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));\n\
#1001=(LENGTH_MEASURE_WITH_UNIT() MEASURE_REPRESENTATION_ITEM() MEASURE_WITH_UNIT(LENGTH_MEASURE(8.0),#1000) REPRESENTATION_ITEM('WallThickness'));\n\
#1002=SHAPE_REPRESENTATION_WITH_PARAMETERS('design parameters',(#1001),#23);\n\
#1003=PROPERTY_DEFINITION('design parameters','',#5);\n\
#1004=PROPERTY_DEFINITION_REPRESENTATION(#1003,#1002);\n\
#1005=ID_ATTRIBUTE('parameter-wall-thickness',#1001);\n\
ENDSEC;\nEND-ISO-10303-21;",
        1,
    )
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
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn exact_solid_count(model: &runmat_geometry_core::GeometryModel) -> Option<u64> {
    match model {
        runmat_geometry_core::GeometryModel::ExactBRep { model } => Some(model.solid_count),
        runmat_geometry_core::GeometryModel::FacetedSolid { .. } => None,
    }
}
