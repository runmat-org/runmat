use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    build_geometry_task_submission, import_geometry_materialization_output,
    prepare_geometry_materialization_program, GeometryExecutionContext,
    GeometryMaterializationOutput, GeometryMaterializationRequest, GeometryMaterializationSource,
    DEFAULT_GEOMETRY_MAXIMUM_ATTEMPTS,
};
use runmat_execution::identity::ArtifactId;
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, LogicalObject, ProgramExecutionResponse,
};
use runmat_geometry_core::{
    DerivedGeometryDefinition, GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef,
    GeometryDefinitionSource, GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation,
    GeometryFeatureRef, GeometryParameterDefinition, GeometryParameterType,
    GeometryRegionCardinality, GeometryRegionDefinition, GeometryRegionRole,
    GeometryRegionSelector, GeometryScalarExpression,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_geometry_ops::GeometryFeaturePlanLimits;
use runmat_meshing_core::{resolve_meshing_request, MeshingRequestSettings};
use runmat_meshing_execution::{import_exact_geometry_input, ExactMeshingDagPlanner};

use super::{
    execute_geometry_materialization_program_request, native_geometry_provider_profile,
    NativeGeometryMaterializationLimits,
};

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
fn derived_geometry_sweep_reuses_its_source_and_invalidates_every_downstream_identity() {
    let access = ArtifactAccess {
        authorization_scope: "geometry:sweep-test".into(),
        encryption_context: Digest::sha256(b"geometry-sweep-test-key"),
    };
    let mut store = MemoryStore::default();
    let base_request = base_request();
    let (base, base_objects, base_output_root) = execute(&base_request, &access, &[], &mut store);
    let source_objects = base_objects
        .into_iter()
        .filter(|reference| reference.logical_digest != base_output_root.logical_digest)
        .collect::<Vec<_>>();

    let narrow_request = swept_request(&base_request, &base, 0.25);
    let wide_request = swept_request(&base_request, &base, 1.25);
    assert_ne!(
        narrow_request.identity().unwrap(),
        wide_request.identity().unwrap()
    );
    let narrow_program = prepare_geometry_materialization_program(
        narrow_request.clone(),
        revision(),
        &access,
        &source_objects,
    )
    .unwrap();
    let wide_program = prepare_geometry_materialization_program(
        wide_request.clone(),
        revision(),
        &access,
        &source_objects,
    )
    .unwrap();
    let scope_id = ExecutionScopeId::derive(&[b"geometry-sweep-scope"]);
    let pool_id = PoolId::derive(&[b"geometry-sweep-pool"]);
    let submission =
        |program: &runmat_analysis_execution::PreparedGeometryMaterializationProgram| {
            build_geometry_task_submission(
                program,
                BTreeSet::new(),
                GeometryExecutionContext {
                    host: crate::test_support::host_requirement(),
                    scope_id,
                    pool_id,
                    program_artifact_id: ArtifactId::derive(&[program
                        .program
                        .artifact
                        .id
                        .0
                        .bytes()]),
                    deadline_unix_millis: None,
                    priority: 0,
                    maximum_attempts: DEFAULT_GEOMETRY_MAXIMUM_ATTEMPTS,
                },
            )
            .unwrap()
        };
    let narrow_submission = submission(&narrow_program);
    let wide_submission = submission(&wide_program);
    assert_ne!(narrow_submission.request.id, wide_submission.request.id);
    assert_eq!(
        narrow_submission.request.resources.required_capabilities,
        wide_submission.request.resources.required_capabilities
    );
    let (narrow, _, _) = execute(&narrow_request, &access, &source_objects, &mut store);
    let (wide, _, _) = execute(&wide_request, &access, &source_objects, &mut store);
    let (narrow_repeat, _, _) = execute(&narrow_request, &access, &source_objects, &mut store);

    assert_eq!(narrow.geometry_root, narrow_repeat.geometry_root);
    assert_ne!(
        narrow.geometry_root.logical_digest,
        wide.geometry_root.logical_digest
    );
    assert_ne!(
        narrow.document.primary_artifact().digest,
        wide.document.primary_artifact().digest
    );
    let narrow_source = source_root(&narrow_request);
    let wide_source = source_root(&wide_request);
    assert_eq!(narrow_source, wide_source);
    assert_eq!(
        narrow_source.digest,
        base.document.primary_artifact().digest
    );

    let inventory = ObjectInventoryLimits::default();
    let narrow_geometry = import_exact_geometry_input(
        &store,
        narrow.document.clone(),
        &narrow.geometry_root,
        access.clone(),
        inventory,
    )
    .unwrap();
    let wide_geometry = import_exact_geometry_input(
        &store,
        wide.document.clone(),
        &wide.geometry_root,
        access.clone(),
        inventory,
    )
    .unwrap();
    let narrow_mesh =
        resolve_meshing_request(narrow.document.tolerance, MeshingRequestSettings::default())
            .unwrap();
    let wide_mesh =
        resolve_meshing_request(wide.document.tolerance, MeshingRequestSettings::default())
            .unwrap();
    let narrow_plan = ExactMeshingDagPlanner::new(
        &narrow_geometry,
        narrow_mesh,
        runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
        access.clone(),
        None,
    )
    .unwrap()
    .initial_curve_pass(&narrow_geometry.geometry_objects().topology, 64)
    .unwrap();
    let wide_plan = ExactMeshingDagPlanner::new(
        &wide_geometry,
        wide_mesh,
        runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
        access,
        None,
    )
    .unwrap()
    .initial_curve_pass(&wide_geometry.geometry_objects().topology, 64)
    .unwrap();
    assert_ne!(
        narrow_plan.partitions()[0]
            .host()
            .workload
            .stage_identity_digest,
        wide_plan.partitions()[0]
            .host()
            .workload
            .stage_identity_digest
    );
    assert_ne!(
        narrow_plan.partitions()[0].host().workload.inputs,
        wide_plan.partitions()[0].host().workload.inputs
    );
}

fn execute(
    request: &GeometryMaterializationRequest,
    access: &ArtifactAccess,
    source_roots: &[runmat_execution::value::ValueRef],
    store: &mut MemoryStore,
) -> (
    GeometryMaterializationOutput,
    Vec<runmat_execution::value::ValueRef>,
    runmat_execution::value::ValueRef,
) {
    let prepared =
        prepare_geometry_materialization_program(request.clone(), revision(), access, source_roots)
            .unwrap();
    prepared.write_to(store).unwrap();
    let response = execute_geometry_materialization_program_request(
        &prepared.program,
        store,
        NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        panic!("geometry sweep case failed: {response:?}");
    };
    let ValuePayload::Object(root) = &outputs[0] else {
        panic!("geometry sweep output must be externalized");
    };
    let output_root = root.as_ref().clone();
    let output = import_geometry_materialization_output(store, root, access).unwrap();
    output.validate_against(request).unwrap();
    (output, result_objects, output_root)
}

fn base_request() -> GeometryMaterializationRequest {
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/sweep.fea".into(),
            geometry_id: "stock".into(),
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
            regions: body_region(),
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::new(),
        source: GeometryMaterializationSource::Constructed,
        import_options: ExactCadImportOptions::default(),
        provider: native_geometry_provider_profile().unwrap(),
        plan_limits: GeometryFeaturePlanLimits::default(),
        resources: Default::default(),
    }
}

fn swept_request(
    source: &GeometryMaterializationRequest,
    output: &GeometryMaterializationOutput,
    offset_m: f64,
) -> GeometryMaterializationRequest {
    let source_definition = GeometryDefinitionRef {
        module_path: source.definition.module_path.clone(),
        geometry_id: source.definition.geometry_id.clone(),
    };
    let mut matrix = (0..16)
        .map(|index| {
            scalar(if matches!(index, 0 | 5 | 10 | 15) {
                1.0
            } else {
                0.0
            })
        })
        .collect::<Vec<_>>();
    matrix[3] = GeometryScalarExpression::Parameter {
        name: "offset".into(),
    };
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/sweep.fea".into(),
            geometry_id: "translated".into(),
            parameters: vec![GeometryParameterDefinition {
                parameter_id: "offset".into(),
                parameter_type: GeometryParameterType::Length,
                required: true,
                default: None,
            }],
            source: GeometryDefinitionSource::Derived(DerivedGeometryDefinition {
                source: source_definition.clone(),
                graph: GeometryFeatureGraph {
                    features: vec![GeometryFeature {
                        feature_id: "translated".into(),
                        operation: GeometryFeatureOperation::Transform {
                            source: GeometryFeatureRef("source.result".into()),
                            matrix,
                        },
                    }],
                    result: GeometryFeatureRef("translated".into()),
                },
            }),
            regions: body_region(),
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::from([(
            "offset".into(),
            GeometryAuthoringValue::Scalar(scalar(offset_m)),
        )]),
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

fn source_root(
    request: &GeometryMaterializationRequest,
) -> &runmat_geometry_core::GeometryObjectRef {
    let GeometryMaterializationSource::Derived { root, .. } = &request.source else {
        panic!("expected derived geometry sweep request");
    };
    root
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

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"geometry-sweep-graph"),
        Digest::sha256(b"geometry-sweep-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}
