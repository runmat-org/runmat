use std::collections::{BTreeMap, BTreeSet};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_execution::{
    import_geometry_materialization_output, prepare_geometry_materialization_program,
    GeometryMaterializationRequest, GeometryMaterializationSource,
};
use runmat_execution::resource::ResourceInventory;
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ExecutionBundleBuilder};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_transport_native::control::ResourceRequest;
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_geometry_core::{
    GeometryDefinition, GeometryDefinitionSource, GeometryFeature, GeometryFeatureGraph,
    GeometryFeatureOperation, GeometryFeatureRef, GeometryRegionCardinality,
    GeometryRegionDefinition, GeometryRegionRole, GeometryRegionSelector, GeometryScalarExpression,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_geometry_ops::GeometryFeaturePlanLimits;
use runmat_meshing_execution::exact_meshing_worker_capabilities;

use super::*;

#[tokio::test(flavor = "current_thread")]
async fn constructed_geometry_reaches_a_terminal_mesh_over_encrypted_quic() {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-geometry-mesh\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let identity = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*identity.graph_digest.bytes()),
        Digest::from_bytes(*identity.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let access = ArtifactAccess {
        authorization_scope: "remote-geometry-mesh-run".into(),
        encryption_context: Digest::sha256(b"remote-geometry-mesh-encryption"),
    };
    let geometry_request = constructed_request();
    let geometry = prepare_geometry_materialization_program(
        geometry_request.clone(),
        revision.clone(),
        &access,
        &[],
    )
    .unwrap();
    let mut meshing_request = crate::resolve_native_meshing_request(
        &geometry_request.import_options,
        runmat_meshing_core::MeshingRequestSettings::default(),
    )
    .unwrap();
    make_coarse(&mut meshing_request);
    let prepared = prepare_geometry_meshing_root(GeometryMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        geometry: geometry.clone(),
        geometry_source_objects: Vec::new(),
        meshing_request: meshing_request.clone(),
        preferred_edges_per_partition: 3,
        preferred_faces_per_partition: 2,
    })
    .unwrap();

    let local_directory = tempfile::tempdir().unwrap();
    let mut local_store =
        crate::FilesystemObjectStore::open(local_directory.path().join("objects"), 512 << 20)
            .unwrap();
    geometry.write_to(&mut local_store).unwrap();
    let local_response = crate::execute_geometry_materialization_program_request(
        &geometry.program,
        &mut local_store,
        crate::NativeGeometryMaterializationLimits::default(),
    );
    let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
        outputs, ..
    } = local_response
    else {
        panic!("local geometry materialization did not complete")
    };
    let [ValuePayload::Object(output_root)] = outputs.as_slice() else {
        panic!("local geometry materialization returned an invalid root")
    };
    let output =
        import_geometry_materialization_output(&local_store, output_root, &access).unwrap();

    let mut builder = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
            &revision,
        ))
        .unwrap()
        .with_compiled_package_closure();
    for program in prepared.required_programs() {
        builder = builder.with_materialized_program(
            program.recipe.clone(),
            program.artifact.form,
            program.artifact.executable_bytes.clone(),
        );
    }
    let mut bundle = Vec::new();
    write_bundle(
        &builder.build().unwrap(),
        &mut bundle,
        ArchiveLimits::default(),
    )
    .unwrap();

    let cohort = crate::meshing_platform::native_meshing_capability_cohort();
    let mut capabilities =
        exact_meshing_worker_capabilities(&output.document, &meshing_request, Some(&cohort))
            .unwrap();
    capabilities.insert(runmat_analysis_execution::geometry_provider_capability(
        &geometry_request.provider,
    ));
    let resources = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        accelerators: Vec::new(),
        maximum_wall_millis: 180_000,
    };
    let inventory = ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        accelerators: Vec::new(),
        capabilities,
    };
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"geometry-mesh-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"geometry-mesh-pool"]);
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[b"geometry-mesh-worker"]),
        pool_id,
        resources: inventory.clone(),
    };
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let authority = listener.local_addr().unwrap();
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([41; 32]).unwrap();
    let server = crate::run_remote_meshing_worker_quic(crate::RemoteMeshingWorkerQuicRequest {
        worker: crate::RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 53,
            session_id: [31; 16],
            run_key: run_key.clone(),
            limits: FrameLimits::default(),
        },
        kernel: Arc::new(crate::native_meshing_kernel_dispatcher()),
        meshing_limits: crate::NativeMeshingHostLimits::default(),
    });
    let client = async {
        let channel = crate::QuicRemoteWorkerChannel::connect(
            crate::RemoteWorkerChannelConfig {
                run_identity: access.authorization_scope.clone(),
                node_identity: "geometry-mesh-node".into(),
                worker: worker.clone(),
                driver_fence: 53,
                session_id: [31; 16],
                run_key,
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority,
                server_name: "runmat.execution".into(),
                certificate_der,
            },
        )
        .await
        .unwrap();
        let pool = crate::RemotePoolDriver::new_with_value_scope(
            scope_id,
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: inventory,
            },
            53,
            bundle,
            access.authorization_scope.clone(),
        )
        .unwrap();
        pool.add_worker(channel.clone()).await.unwrap();
        for argument in &prepared.request().arguments {
            let ValuePayload::Object(reference) = argument else {
                panic!("geometry meshing root input was not externalized")
            };
            assert_eq!(reference.kind, ValueRefKind::DriverObject);
            let object = prepared
                .objects()
                .iter()
                .find(|object| object.descriptor.digest == reference.logical_digest)
                .unwrap();
            pool.register_execution_object(reference.as_ref().clone(), object.bytes.clone())
                .unwrap();
        }
        let outcome = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_meshing::execute_geometry_root(
                Arc::clone(&pool),
                prepared.into_request(),
                access.authorization_scope.clone(),
                resources,
            ),
        )
        .await
        .expect("remote geometry meshing root timed out")
        .unwrap();
        let Ok(result) = outcome else {
            panic!("remote geometry meshing root did not complete")
        };
        let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = result.response
        else {
            panic!("remote geometry meshing root did not publish a terminal mesh")
        };
        let [ValuePayload::Object(root)] = outputs.as_slice() else {
            panic!("remote geometry meshing root returned an invalid terminal root")
        };
        assert!(result_objects.contains(root));
        let terminal = runmat_meshing_execution::import_exact_meshing_result(
            pool.as_ref(),
            root.as_ref().clone(),
            result_objects,
            access,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        assert!(!terminal.artifact.topology.volume_elements.is_empty());
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
}

fn constructed_request() -> GeometryMaterializationRequest {
    GeometryMaterializationRequest {
        schema_version: 1,
        definition: GeometryDefinition {
            module_path: "examples/constructed.fea".into(),
            geometry_id: "block".into(),
            parameters: Vec::new(),
            source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
                features: vec![GeometryFeature {
                    feature_id: "stock".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: scalar(1.0),
                        depth_m: scalar(1.0),
                        height_m: scalar(1.0),
                    },
                }],
                result: GeometryFeatureRef("stock".into()),
            }),
            regions: vec![GeometryRegionDefinition {
                region_id: "body".into(),
                role: GeometryRegionRole::Solid,
                selector: GeometryRegionSelector::Result,
                cardinality: GeometryRegionCardinality::ExactlyOne,
            }],
            cad_parameters: Vec::new(),
        },
        bindings: BTreeMap::new(),
        source: GeometryMaterializationSource::Constructed,
        import_options: ExactCadImportOptions::default(),
        provider: crate::native_geometry_provider_profile().unwrap(),
        plan_limits: GeometryFeaturePlanLimits::default(),
        resources: Default::default(),
    }
}

fn scalar(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

fn make_coarse(request: &mut runmat_meshing_core::MeshingRequest) {
    request.metric.global_metric =
        runmat_meshing_core::MetricTensor3::isotropic_length_m(10.0).unwrap();
    request.quality.curve.maximum_chordal_deviation_m = runmat_types::physical::Length::new(0.1);
    request.quality.curve.maximum_tangent_change_degrees = 180.0;
    request.quality.curve.minimum_metric_edge_length = 0.01;
    request.quality.curve.maximum_metric_edge_length = 10.0;
    request.quality.surface.minimum_metric_angle_degrees = 0.1;
    request.quality.surface.maximum_physical_aspect_ratio = 1_000.0;
    request.quality.surface.maximum_chordal_deviation_m = runmat_types::physical::Length::new(0.1);
    request.quality.surface.maximum_normal_deviation_degrees = 180.0;
    request.quality.volume.maximum_metric_edge_length = 2.0;
    request.quality.volume.maximum_radius_edge_ratio = 10.0;
    request.quality.volume.minimum_scaled_jacobian = 0.01;
    request.limits.maximum_nodes = 10_000;
    request.limits.maximum_elements = 10_000;
}
