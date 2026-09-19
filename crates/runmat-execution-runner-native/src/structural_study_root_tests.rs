use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation, PreparedGeometrySource,
};
use runmat_analysis_fea::study::{
    StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

use super::*;

#[test]
fn typed_study_root_binds_bounded_accelerators_to_the_planned_backend() {
    let request = runmat_execution::resource::AcceleratorRequest {
        class: "gpu.test".into(),
        count: 1,
        memory_bytes_each: 1024,
    };
    assert!(validate_accelerator_requests(
        runmat_analysis_fea::ComputeBackend::Cpu,
        std::slice::from_ref(&request),
    )
    .is_err());
    assert!(validate_accelerator_requests(runmat_analysis_fea::ComputeBackend::Gpu, &[]).is_err());
    let excessive = vec![request; 17];
    assert!(
        validate_accelerator_requests(runmat_analysis_fea::ComputeBackend::Gpu, &excessive,)
            .is_err()
    );
}

#[test]
fn typed_study_root_round_trips_exact_scientific_intent() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    std::fs::write(&path, SOURCE).unwrap();
    let checked =
        runmat_runtime::analysis::check_fea_path_in_root(temporary.path(), &path).unwrap();
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked.resolve_study("project/main.fea", "static").unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();

    let revision = revision();
    let access = ArtifactAccess {
        authorization_scope: "typed-study-root".into(),
        encryption_context: Digest::sha256(b"typed-study-root-encryption"),
    };
    let source = PreparedGeometrySource::constructed();
    let geometry_request = prepare_geometry_materialization_request(
        resolved.geometry,
        realized.geometry.bindings.clone(),
        &source,
        contract_geometry_provider_profile(),
        GeometryMaterializationPreparation::default(),
    )
    .unwrap();
    let geometry = prepare_geometry_materialization_program(
        geometry_request.clone(),
        revision.clone(),
        &access,
        &source.roots,
    )
    .unwrap();
    let meshing_request = crate::resolve_native_meshing_request(
        &geometry_request.import_options,
        resolved.mesh.settings,
    )
    .unwrap();
    let geometry_meshing =
        crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            geometry,
            geometry_source_objects: source.objects,
            meshing_request,
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
        })
        .unwrap();
    let planning = StructuralStudyPlanningPolicy {
        backend: runmat_analysis_fea::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        numerics: StructuralNumericalPlanningPolicy::production(),
    };
    let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )
    .unwrap();
    let prepared = prepare_structural_study_root(StructuralStudyRootPreparation {
        program_revision: revision,
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest: Some(Digest::sha256(b"frozen-project")),
        study: realized.clone(),
        materials: resolved.material_catalog.clone(),
        planning,
        analysis_accelerator_requests: Vec::new(),
        geometry_meshing,
    })
    .unwrap();
    assert!(prepared
        .required_programs()
        .iter()
        .any(|program| program.recipe.execution_mode == STRUCTURAL_STUDY_ROOT_EXECUTION_MODE));
    let expectation = prepared.result_expectation();
    let encoded = serde_json::to_vec(&expectation).unwrap();
    let decoded_expectation: StructuralStudyResultExpectation =
        serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded_expectation, expectation);
    let mut substituted_material = expectation.clone();
    substituted_material.materials[0].material.name = "substituted steel".into();
    assert!(substituted_material.validate().is_err());
    let mut substituted_geometry = expectation.clone();
    substituted_geometry.geometry_request.definition.geometry_id = "other-part".into();
    assert!(substituted_geometry.validate().is_err());
    let directory = tempfile::tempdir().unwrap();
    let mut store = crate::FilesystemObjectStore::open(directory.path(), 64 << 20).unwrap();
    prepared.write_to(&mut store).unwrap();
    let decoded = decode_structural_study_root(prepared.request(), &store).unwrap();
    assert_eq!(decoded.study, realized);
    assert_eq!(decoded.materials, resolved.material_catalog);
    assert_eq!(decoded.planning, planning);
    assert!(decoded.analysis_accelerator_requests.is_empty());
    assert_eq!(
        prepared.transport_objects().unwrap().len(),
        prepared.objects().len()
    );
}

#[cfg(feature = "occt-native")]
#[tokio::test(flavor = "current_thread")]
async fn typed_sweep_root_executes_geometry_mesh_compilation_and_solve_over_quic() {
    use std::collections::BTreeSet;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::sync::Arc;
    use std::time::Duration;

    use rcgen::{generate_simple_self_signed, CertifiedKey};
    use runmat_execution::resource::{Capability, ResourceInventory};
    use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
    use runmat_execution_artifact::encryption::RunKeyMaterial;
    use runmat_execution_artifact::ExecutionBundleBuilder;
    use runmat_execution_runner::{PoolSpec, WorkerSpec};
    use runmat_execution_transport_native::control::ResourceRequest;
    use runmat_execution_transport_native::frame::FrameLimits;
    use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};

    let project_root = tempfile::tempdir().unwrap();
    let source_root = project_root.path().join("src");
    std::fs::create_dir(&source_root).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"typed-study-root\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let path = source_root.join("main.fea");
    std::fs::write(&path, SOURCE).unwrap();
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
        authorization_scope: "typed-study-root-quic".into(),
        encryption_context: Digest::sha256(b"typed-study-root-quic-encryption"),
    };
    let prepared = prepare_source(&source_root, &path, revision.clone(), access.clone());
    let study_identity = runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
        prepared.authored_study_digest(),
        None,
    )
    .unwrap();
    let definition = runmat_analysis_fea::execution::SweepDefinition::new(
        Digest::sha256(b"typed-study-root-quic-sweep"),
        vec![study_identity],
        runmat_analysis_fea::execution::SweepCompositionPolicy::CollectAll,
    )
    .unwrap();
    let prepared = crate::prepare_structural_sweep_root(crate::StructuralSweepRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        definition,
        studies: vec![prepared],
    })
    .unwrap();
    let expectation = prepared.result_expectation();
    let encoded = serde_json::to_vec(&expectation).unwrap();
    let decoded_expectation: crate::StructuralSweepResultExpectation =
        serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded_expectation, expectation);
    let mut builder = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure();
    for program in prepared.required_programs() {
        builder = builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
        );
    }
    let mut bundle = Vec::new();
    write_bundle(
        &builder.build().unwrap(),
        &mut bundle,
        ArchiveLimits::default(),
    )
    .unwrap();

    let resources = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 2 << 30,
        scratch_bytes: 2 << 30,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
        maximum_wall_millis: 180_000,
    };
    let mut capabilities = runmat_analysis_execution::analysis_worker_capabilities(
        std::iter::empty(),
        std::iter::empty(),
        false,
    );
    capabilities.insert(Capability::ProcessIsolation);
    capabilities.extend(
        runmat_meshing_execution::exact_meshing_worker_capability_inventory(
            &[runmat_geometry_io::native_exact_kernel_abi().unwrap()],
            &runmat_meshing_core::production_algorithm_versions(),
            &[
                runmat_meshing_core::ElementOrder::Tet4,
                runmat_meshing_core::ElementOrder::Tet10,
            ],
            [crate::meshing_platform::native_meshing_capability_cohort().as_str()],
        )
        .unwrap(),
    );
    capabilities.insert(runmat_analysis_execution::geometry_provider_capability(
        &crate::native_geometry_provider_profile().unwrap(),
    ));
    let inventory = ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        accelerators: Vec::new(),
        capabilities,
    };
    let pool_id = runmat_execution::PoolId::derive(&[b"typed-study-root-pool"]);
    let worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"typed-study-root-worker"]),
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
    let run_key = RunKeyMaterial::from_entropy([73; 32]).unwrap();
    let server = crate::run_remote_meshing_worker_quic(crate::RemoteMeshingWorkerQuicRequest {
        worker: crate::RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 71,
            session_id: [19; 16],
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
                node_identity: "typed-study-root-node".into(),
                worker: worker.clone(),
                driver_fence: 71,
                session_id: [19; 16],
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
            runmat_execution::ExecutionScopeId::derive(&[b"typed-study-root-scope"]),
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: inventory,
            },
            71,
            bundle,
            access.authorization_scope.clone(),
        )
        .unwrap();
        pool.add_worker(channel.clone()).await.unwrap();
        for object in prepared.transport_objects() {
            pool.register_execution_object(object.reference, object.bytes)
                .unwrap();
        }
        let request = prepared.into_request();
        let outcome = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_sweep::execute(
                Arc::clone(&pool),
                request,
                access.authorization_scope.clone(),
                resources.clone(),
            ),
        )
        .await
        .expect("typed structural sweep timed out")
        .unwrap();
        match outcome {
            crate::remote::pool_execution::RemotePoolExecutionOutcome::Completed(result) => {
                let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
                    result_objects,
                    ..
                } = &result.response
                else {
                    panic!("typed structural sweep did not publish a result closure");
                };
                let cache = TestResultCache(
                    result_objects
                        .iter()
                        .map(|reference| {
                            (
                                reference.logical_digest,
                                result
                                    .read_result_object(reference, &access.authorization_scope)
                                    .unwrap(),
                            )
                        })
                        .collect(),
                );
                let imported = crate::import_structural_sweep_result(
                    &cache,
                    result.response,
                    &expectation,
                    runmat_execution_artifact::object::ObjectInventoryLimits::default(),
                )
                .unwrap();
                let [entry] = imported.entries.as_slice() else {
                    panic!("typed structural sweep changed its case count");
                };
                let crate::ExactStructuralSweepTerminal::Succeeded(study) = &entry.terminal else {
                    panic!("typed structural sweep case did not succeed");
                };
                let compiled = study
                    .compiled_study
                    .as_ref()
                    .expect("typed study result omitted its compiled scientific contract");
                compiled.validate_against(&study.mesh.artifact).unwrap();
                assert_eq!(compiled.study_id, "static");
            }
            crate::remote::pool_execution::RemotePoolExecutionOutcome::Cancelled => {
                panic!("typed structural sweep was cancelled")
            }
            crate::remote::pool_execution::RemotePoolExecutionOutcome::Indeterminate(message) => {
                panic!("typed structural sweep was indeterminate: {message}")
            }
        }
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
}

#[cfg(feature = "occt-native")]
struct TestResultCache(std::collections::BTreeMap<Digest, Vec<u8>>);

#[cfg(feature = "occt-native")]
impl runmat_execution_artifact::cache::CacheImport for TestResultCache {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

#[cfg(feature = "occt-native")]
fn prepare_source(
    source_root: &std::path::Path,
    path: &std::path::Path,
    revision: ProgramRevision,
    access: ArtifactAccess,
) -> PreparedStructuralStudyRoot {
    let checked = runmat_runtime::analysis::check_fea_path_in_root(source_root, path).unwrap();
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked.resolve_study("project/main.fea", "static").unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let source = PreparedGeometrySource::constructed();
    let geometry_request = prepare_geometry_materialization_request(
        resolved.geometry,
        realized.geometry.bindings.clone(),
        &source,
        crate::native_geometry_provider_profile().unwrap(),
        GeometryMaterializationPreparation::default(),
    )
    .unwrap();
    let geometry = prepare_geometry_materialization_program(
        geometry_request.clone(),
        revision.clone(),
        &access,
        &source.roots,
    )
    .unwrap();
    let mut meshing_request = crate::resolve_native_meshing_request(
        &geometry_request.import_options,
        resolved.mesh.settings,
    )
    .unwrap();
    meshing_request.resources.maximum_wall_time_ms = 180_000;
    let geometry_meshing =
        crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            geometry,
            geometry_source_objects: source.objects,
            meshing_request,
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
        })
        .unwrap();
    let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )
    .unwrap();
    prepare_structural_study_root(StructuralStudyRootPreparation {
        program_revision: revision,
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest: Some(Digest::sha256(b"frozen-project")),
        study: realized,
        materials: resolved.material_catalog,
        planning: StructuralStudyPlanningPolicy {
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            numerics: StructuralNumericalPlanningPolicy::production(),
        },
        analysis_accelerator_requests: Vec::new(),
        geometry_meshing,
    })
    .unwrap()
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"typed-study-graph"),
        Digest::sha256(b"typed-study-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn contract_geometry_provider_profile() -> runmat_analysis_execution::GeometryProviderProfile {
    runmat_analysis_execution::GeometryProviderProfile {
        provider_id: "exact-contract-test".into(),
        provider_version: "1".into(),
        kernel_abi: "exact-contract-test-abi".into(),
        capabilities: runmat_geometry_ops::GeometryFeatureCapabilities {
            supported: std::collections::BTreeSet::from([
                runmat_geometry_ops::GeometryFeatureKind::Box,
            ]),
            maximum_features: 16,
            maximum_instances_per_feature: 16,
            maximum_boolean_operands: 8,
        },
    }
}

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region support : face {
        select result
        expect one_or_more
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 10 m
        maximum_deviation = 0.1 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
model structure {
    assign material.steel to geometry.part.region.body
    constraint base {
        structural.fixed { target = geometry.part.region.support }
    }
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {}
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
