//! Every authored family through the portable study root on a loopback multi-node cluster.
//!
//! Each study runs once locally on process-isolated workers and once through the remote driver
//! composition on QUIC loopback nodes. The client imports the remote closure, publishes it and
//! compares the solved numerical digest with the local run under the same artifact authority.
use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_execution::resource::{Capability, ResourceInventory};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactResult, ExecutionBundleBuilder};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_runner_native::{
    execute_prepared_fea_study_locally, execute_study_root_on_pool, import_family_study_result,
    prepare_checked_fea_study, prepare_structural_study_root, run_remote_meshing_worker_quic,
    LocalFeaExecutionOptions, QuicRemoteWorkerChannel, RemoteMeshingWorkerQuicRequest,
    RemotePoolDriver, RemoteStudyRootOutcome, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_runtime::analysis::{FeaRootSelector, ResolvedFeaRoot};

const BOX_BREP: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");

const THERMAL: &str = r#"
geometry part {
 file = "./box.brep"; units = m
 region body : solid { select id "solid_000001"; expect exactly_one }
 region cold : face { select id "face_000001"; expect exactly_one }
 region hot : face { select id "face_000002"; expect exactly_one }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 1.5 m; maximum_deviation = 0.02 m } }
material conductor {
 mass_density = 2 kg/m^3
 thermal { reference_temperature = 300 K; modulus_temperature_coefficient = 0 K^-1
  conductivity = 3 W/m/K; specific_heat = 4 J/kg/K; expansion_coefficient = 0 K^-1 }
}
model bar {
 thermal { reference_temperature = 300 K }
 assign material.conductor to geometry.part.region.body
 constraint cold { thermal.temperature { target = geometry.part.region.cold; value = 300 K } }
 constraint hot { thermal.temperature { target = geometry.part.region.hot; value = 350 K } }
}
study response {
 geometry = geometry.part; mesh = mesh.volume; model = model.bar
 analysis heat { thermal steady {} }
 output temperature = analysis.heat.field.node.temperature
}
"#;

const ELECTRICAL: &str = r#"
geometry part {
 file = "./box.brep"; units = m
 region body : solid { select id "solid_000001"; expect exactly_one }
 region low : face { select id "face_000001"; expect exactly_one }
 region high : face { select id "face_000002"; expect exactly_one }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 1.5 m; maximum_deviation = 0.02 m } }
material dielectric {
 electrical { reference_temperature = 293.15 K; conductivity = 3 A/V/m
  resistive_heating_coefficient = 0 K^-1; relative_permittivity = 4; relative_permeability = 1 }
}
model electric {
 electromagnetic { equation = steady_conduction }
 assign material.dielectric to geometry.part.region.body
 constraint low { electrical.potential { target = geometry.part.region.low; value = 0 V } }
 constraint high { electrical.potential { target = geometry.part.region.high; value = 2 V } }
}
study response {
 geometry = geometry.part; mesh = mesh.volume; model = model.electric
 analysis potential { electromagnetic static {}; accuracy { algebraic_relative_tolerance = 1e-11 } }
 output potential = analysis.potential.field.node.electric_potential
}
"#;

const MAGNETIC: &str = r#"
geometry part {
 construct { feature body = box(width = 1 m, depth = 1 m, height = 1 m); result = feature.body }
 region bulk : solid { select result; expect exactly_one }
 region exterior : face { select result; expect one_or_more }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 0.5 m; maximum_deviation = 0.02 m } }
material vacuum {
 electrical { reference_temperature = 293.15 K; conductivity = 0 A/V/m
  resistive_heating_coefficient = 0 K^-1; relative_permittivity = 1; relative_permeability = 1 }
}
model coil {
 electromagnetic { equation = magnetostatics }
 assign material.vacuum to geometry.part.region.bulk
 constraint ground { magnetic.vector_potential_ground { target = geometry.part.region.exterior } }
 constraint gauge { magnetic.coulomb_gauge { target = geometry.part.region.bulk } }
 load current { magnetic_volume_current_density on geometry.part.region.bulk = [0, 0, 1] A/m^2 }
 load_case drive { include load.current }
}
study response {
 geometry = geometry.part; mesh = mesh.volume; model = model.coil
 analysis potential { electromagnetic static {}; accuracy { algebraic_relative_tolerance = 1e-10 } }
 output flux = analysis.potential.field.integration_point.magnetic_flux_density
}
"#;

const MAXWELL: &str = r#"
geometry part {
 construct { feature body = box(width = 1 m, depth = 1 m, height = 1 m); result = feature.body }
 region bulk : solid { select result; expect exactly_one }
 region exterior : face { select result; expect one_or_more }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 0.5 m; maximum_deviation = 0.02 m } }
material conductor {
 electrical { reference_temperature = 293.15 K; conductivity = 3 A/V/m
  resistive_heating_coefficient = 0 K^-1; relative_permittivity = 4; relative_permeability = 1 }
}
model field {
 electromagnetic { equation = harmonic_maxwell; reference_frequency = 159154.94309189534 Hz }
 assign material.conductor to geometry.part.region.bulk
 constraint wall { electrical.perfect_conductor { target = geometry.part.region.exterior } }
 signal current : electric_current_density over frequency {
  samples = [{ at = 159154.94309189534 Hz,
   value = { real = [0 A/m^2, 0.6 A/m^2, 0 A/m^2], imaginary = [0 A/m^2, 0.8 A/m^2, 0 A/m^2] } }]
  interpolation = linear; extrapolation = reject; frame = global
 }
 excitation drive { magnetic_volume_current_density { target = geometry.part.region.bulk; signal = signal.current } }
 load_case driven { include excitation.drive }
}
study response {
 geometry = geometry.part; mesh = mesh.volume; model = model.field
 analysis phasor {
  electromagnetic harmonic { frequencies = [159154.94309189534 Hz] }
  accuracy { algebraic_relative_tolerance = 1e-10 }
 }
 output electric = analysis.phasor.field.integration_point.electric_field
}
"#;

const ACOUSTIC: &str = r#"
geometry part {
 file = "./box.brep"; units = m
 region body : solid { select id "solid_000001"; expect exactly_one }
 region inlet : face { select id "face_000001"; expect exactly_one }
 region outlet : face { select id "face_000002"; expect exactly_one }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 1.5 m; maximum_deviation = 0.02 m } }
material fluid { mass_density = 3 kg/m^3; acoustic { speed_of_sound = 2 m/s; damping_ratio = 0 } }
model duct {
 assign material.fluid to geometry.part.region.body
 load termination { acoustic.impedance { target = geometry.part.region.outlet; impedance = 6 Pa*s/m } }
 signal drive : velocity over frequency {
  samples = [
   { at = 0.2 Hz, value = { real = -0.16666666666666666 m/s, imaginary = -0.06666666666666667 m/s } },
   { at = 0.4 Hz, value = { real = -0.16666666666666666 m/s, imaginary = -0.06666666666666667 m/s } }
  ]
  interpolation = linear; extrapolation = reject
 }
 excitation inlet { normal_velocity { target = geometry.part.region.inlet; signal = signal.drive } }
 load_case driven { include excitation.inlet }
}
study response {
 geometry = geometry.part; mesh = mesh.volume; model = model.duct
 analysis sound { acoustic harmonic { frequencies = [0.2 Hz, 0.3 Hz, 0.4 Hz] } }
}
"#;

const FLUID: &str = r#"
geometry part {
 file = "./box.brep"; units = m
 region bulk : cell { select result; expect exactly_one }
 region outlet : face { select id "face_000002"; expect exactly_one }
 region lid : face { select id "face_000001"; expect exactly_one }
 region wall3 : face { select id "face_000003"; expect exactly_one }
 region wall4 : face { select id "face_000004"; expect exactly_one }
 region wall5 : face { select id "face_000005"; expect exactly_one }
 region wall6 : face { select id "face_000006"; expect exactly_one }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 1.5 m; maximum_deviation = 0.02 m } }
material liquid { mass_density = 1 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 1 Pa*s } }
model flow {
 assign material.liquid to geometry.part.region.bulk
 constraint lid { fluid.velocity { target = geometry.part.region.lid; value = [0.001 m/s, 0 m/s, 0 m/s] } }
 constraint wall3 { fluid.no_slip_wall { target = geometry.part.region.wall3 } }
 constraint wall4 { fluid.no_slip_wall { target = geometry.part.region.wall4 } }
 constraint wall5 { fluid.no_slip_wall { target = geometry.part.region.wall5 } }
 constraint wall6 { fluid.no_slip_wall { target = geometry.part.region.wall6 } }
 load outlet { fluid.outlet_pressure { target = geometry.part.region.outlet; value = 0 Pa } }
}
study response {
 geometry = geometry.part; mesh = mesh.volume; model = model.flow
 analysis solve { fluid static {}; accuracy { algebraic_relative_tolerance = 1e-8 } }
 output velocity = analysis.solve.field.cell.fluid_velocity
}
"#;

/// Replace the box fixture's source ordinals (`face_00000N`, `solid_000001`) with the canonical
/// persistent selectors of `box.brep`; the solver matches canonical identities only.
fn box_selectors(source: &str) -> String {
    let imported = runmat_geometry_io::import_exact_cad(
        "box.brep",
        BOX_BREP,
        runmat_geometry_io::GeometryFormat::Brep,
        &runmat_geometry_io::ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..runmat_geometry_io::ExactCadImportOptions::default()
        },
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .unwrap();
    let mut source = source.replace(
        "\"solid_000001\"",
        &format!(
            "\"{}\"",
            imported.topology.regions[0]
                .id
                .analysis_selection_id()
                .unwrap()
        ),
    );
    for (ordinal, face) in &imported.source_face_ids {
        source = source.replace(
            &format!("\"face_{:06}\"", ordinal + 1),
            &format!("\"{}\"", face.analysis_selection_id().unwrap()),
        );
    }
    assert!(!source.contains("\"face_0"), "unmapped box face ordinal");
    source
}

fn nonlinear_source() -> String {
    include_str!("../../../../docs/simulation/examples/small-strain-j2.fea")
        .replace("./geometry/box.brep", "./box.brep")
        .replace("study plastic", "study response")
}

/// Evidence for one family: local versus loopback-cluster digests and wall times.
pub(super) struct RemoteFamilyEvidence {
    pub family: &'static str,
    pub nodes: u32,
    pub local_digest: String,
    pub remote_digest: String,
    pub local_seconds: f64,
    pub remote_seconds: f64,
}

pub(super) async fn run() {
    let nodes = std::env::var("RUNMAT_REMOTE_FAMILY_NODES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let selected = std::env::var("RUNMAT_REMOTE_FAMILY_ONLY").ok();
    let families: Vec<(&'static str, String)> = vec![
        ("thermal", box_selectors(THERMAL)),
        ("electrical", box_selectors(ELECTRICAL)),
        ("magnetic", box_selectors(MAGNETIC)),
        ("maxwell", box_selectors(MAXWELL)),
        ("acoustic", box_selectors(ACOUSTIC)),
        ("fluid", box_selectors(FLUID)),
        ("nonlinear", nonlinear_source()),
    ];
    let mut evidence = Vec::new();
    for (family, source) in families {
        if selected.as_deref().is_some_and(|only| only != family) {
            continue;
        }
        eprintln!("remote family {family}: starting on {nodes} loopback nodes");
        evidence.push(run_family(family, &source, nodes).await);
    }
    println!("| family | nodes | local digest | remote digest | local s | remote s |");
    println!("|---|---|---|---|---|---|");
    for row in &evidence {
        println!(
            "| {} | {} | {} | {} | {:.2} | {:.2} |",
            row.family,
            row.nodes,
            &row.local_digest[..16],
            &row.remote_digest[..16],
            row.local_seconds,
            row.remote_seconds
        );
        assert_eq!(
            row.local_digest, row.remote_digest,
            "{} remote result differs from local",
            row.family
        );
    }
}

async fn run_family(family: &'static str, source: &str, nodes: u32) -> RemoteFamilyEvidence {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    std::env::set_var("RUNMAT_EXECUTION_STATE_DIR", root.join("native-state"));
    std::env::set_var("RUNMAT_FEA_ARTIFACT_ROOT", root.join("runtime-artifacts"));
    std::env::set_var(
        "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        root.join("study-artifacts"),
    );
    runmat_runtime::analysis::storage::configure_artifact_store_from_env().unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("runmat.toml"),
        "[package]\nname = \"remote-family\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    std::fs::write(root.join("src/box.brep"), BOX_BREP).unwrap();
    let path = root.join("src/main.fea");
    std::fs::write(&path, source).unwrap();
    let project =
        runmat_package::build_frozen_project(&root.join("runmat.toml"), BTreeSet::new()).unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .await
        .unwrap();
    let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
    assert_eq!(
        checked.checked().check_result().status,
        runmat_runtime::analysis::FeaCheckStatus::Valid,
        "{family}: {:?}",
        checked.checked().check_result().diagnostics
    );
    let options = LocalFeaExecutionOptions {
        root: Some(FeaRootSelector::Study {
            id: "response".into(),
        }),
        workers: Some(nodes),
        ..Default::default()
    };
    let resolve = || {
        let ResolvedFeaRoot::Study(study) = checked.resolve_root(options.root.as_ref()).unwrap()
        else {
            panic!("{family}: expected a study root")
        };
        study
    };
    // One artifact authority for both placements: object identities then agree exactly.
    let authority = format!("remote-family-{family}");

    let started = Instant::now();
    let prepared = prepare_checked_fea_study(&checked, resolve(), &[], &authority, &options)
        .await
        .unwrap();
    let local = execute_prepared_fea_study_locally(prepared, &options)
        .await
        .unwrap_or_else(|error| panic!("{family}: local execution failed: {error}"));
    let local_seconds = started.elapsed().as_secs_f64();
    let local_digest = local.result.numerical_digest().unwrap().to_string();

    let started = Instant::now();
    let prepared = prepare_checked_fea_study(&checked, resolve(), &[], &authority, &options)
        .await
        .unwrap();
    let study_root = prepare_structural_study_root(prepared.root).unwrap();
    let expectation = study_root.result_expectation();
    let mut builder = ExecutionBundleBuilder::native(&project, prepared.revision.clone())
        .unwrap()
        .with_compiled_package_closure();
    for program in study_root.required_programs() {
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
    let cpu_millicores = expectation
        .local_execution
        .checked_cpu_millicores()
        .unwrap();
    let allocation = AllocationResources {
        cpu_millicores: cpu_millicores.into(),
        memory_bytes: 4 << 30,
        scratch_bytes: 4 << 30,
        accelerators: Vec::new(),
        maximum_wall_millis: 1_800_000,
    };
    let outcome = loopback_cluster(
        nodes,
        &authority,
        bundle,
        cpu_millicores,
        allocation.memory_bytes,
        |pool| {
            let study_root = &study_root;
            let allocation = &allocation;
            async move {
                for object in study_root.transport_objects().unwrap() {
                    pool.register_execution_object(object.reference, object.bytes)
                        .unwrap();
                }
                execute_study_root_on_pool(pool, study_root.request(), "family-study", allocation)
                    .await
                    .unwrap()
            }
        },
    )
    .await;
    let RemoteStudyRootOutcome::Completed { response, objects } = outcome else {
        panic!("{family}: remote study did not complete: {outcome:?}")
    };
    let cache = DownloadedObjects(Arc::new(
        objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    ));
    let remote = import_family_study_result(
        cache,
        response,
        &expectation,
        ObjectInventoryLimits::default(),
    )
    .unwrap_or_else(|error| panic!("{family}: remote import failed: {error}"));
    let remote_digest = remote.numerical_digest().unwrap();
    // The imported closure publishes through the same owner as a local run.
    let publication =
        runmat_execution_runner_native::prepare_study_publication(&remote, "typed-remote")
            .unwrap_or_else(|error| panic!("{family}: remote publication failed: {error}"));
    runmat_execution_runner_native::complete_frozen_fea_assessments(publication, &[])
        .await
        .unwrap_or_else(|error| panic!("{family}: remote publication commit failed: {error}"));
    RemoteFamilyEvidence {
        family,
        nodes,
        local_digest,
        remote_digest: remote_digest.to_string(),
        local_seconds,
        remote_seconds: started.elapsed().as_secs_f64(),
    }
}

#[derive(Clone)]
struct DownloadedObjects(Arc<std::collections::BTreeMap<runmat_execution::Digest, Vec<u8>>>);

impl CacheImport for DownloadedObjects {
    fn read_verified(&self, digest: runmat_execution::Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }

    fn read_verified_bounded(
        &self,
        digest: runmat_execution::Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        match self.0.get(&digest) {
            Some(bytes) if bytes.len() as u64 > maximum_bytes => Err(
                runmat_execution_artifact::ArtifactError::Limit("object exceeds read limit".into()),
            ),
            other => Ok(other.cloned()),
        }
    }
}

fn capabilities() -> BTreeSet<Capability> {
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
            [runmat_execution_runner_native::native_meshing_capability_cohort().as_str()],
        )
        .unwrap(),
    );
    capabilities.insert(runmat_analysis_execution::geometry_provider_capability(
        &runmat_execution_runner_native::native_geometry_provider_profile().unwrap(),
    ));
    capabilities
}

/// Start `nodes` QUIC loopback worker nodes, connect one pool to all of them and run `body`.
pub(super) async fn loopback_cluster<F, Fut, T>(
    nodes: u32,
    run_identity: &str,
    bundle: Vec<u8>,
    cpu_millicores_per_node: u32,
    memory_bytes_per_node: u64,
    body: F,
) -> T
where
    F: FnOnce(Arc<RemotePoolDriver>) -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let signing_key_der = signing_key.serialize_der();
    let pool_id = runmat_execution::PoolId::derive(&[run_identity.as_bytes(), b"/pool"]);
    let inventory = ResourceInventory {
        cpu_millicores: cpu_millicores_per_node,
        memory_bytes: memory_bytes_per_node,
        scratch_bytes: memory_bytes_per_node,
        accelerators: Vec::new(),
        capabilities: capabilities(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([61; 32]).unwrap();
    let mut servers = Vec::new();
    let mut endpoints = Vec::new();
    for node in 0..nodes {
        let listener = QuicOverlayListener::bind(
            SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            vec![certificate_der.clone()],
            signing_key_der.clone(),
            FrameLimits::default(),
        )
        .unwrap();
        let authority = listener.local_addr().unwrap();
        let worker = WorkerSpec {
            host: super::phase_execution_support::worker_host_inventory(),
            id: runmat_execution::identity::WorkerId::derive(&[
                run_identity.as_bytes(),
                &node.to_le_bytes(),
            ]),
            pool_id,
            resources: inventory.clone(),
        };
        let session_id = [u8::try_from(node % 200).unwrap() + 1; 16];
        servers.push(Box::pin(run_remote_meshing_worker_quic(
            RemoteMeshingWorkerQuicRequest {
                worker: RemoteWorkerQuicRequest {
                    listener,
                    run_identity: run_identity.into(),
                    worker: worker.clone(),
                    driver_fence: 91,
                    session_id,
                    run_key: run_key.clone(),
                    limits: FrameLimits::default(),
                },
                kernel: Arc::new(
                    runmat_execution_runner_native::native_meshing_kernel_dispatcher(),
                ),
                meshing_limits: runmat_execution_runner_native::NativeMeshingHostLimits::default(),
            },
        )));
        endpoints.push((authority, worker, session_id));
    }
    let mut resource_limit = inventory.clone();
    resource_limit.cpu_millicores *= nodes;
    resource_limit.memory_bytes *= u64::from(nodes);
    resource_limit.scratch_bytes *= u64::from(nodes);
    let client = async {
        let pool = RemotePoolDriver::new_with_value_scope(
            runmat_execution::ExecutionScopeId::derive(&[run_identity.as_bytes(), b"/scope"]),
            PoolSpec {
                id: pool_id,
                min_workers: nodes,
                max_workers: nodes,
                max_in_flight: nodes,
                resource_limit,
            },
            91,
            bundle,
            run_identity,
        )
        .unwrap();
        for (node, (authority, worker, session_id)) in endpoints.into_iter().enumerate() {
            let channel = QuicRemoteWorkerChannel::connect(
                RemoteWorkerChannelConfig {
                    run_identity: run_identity.into(),
                    node_identity: format!("{run_identity}-node-{node}"),
                    worker,
                    driver_fence: 91,
                    session_id,
                    run_key: run_key.clone(),
                    limits: FrameLimits::default(),
                },
                &PinnedQuicEndpoint {
                    authority,
                    server_name: "runmat.execution".into(),
                    certificate_der: certificate_der.clone(),
                },
            )
            .await
            .unwrap();
            pool.add_worker(channel).await.unwrap();
        }
        tokio::time::timeout(Duration::from_secs(3_600), body(Arc::clone(&pool)))
            .await
            .expect("loopback cluster run timed out")
    };
    // Worker servers are not Send; drive them on this task beside the client.
    let servers = futures::future::join_all(servers);
    tokio::select! {
        result = client => result,
        outcomes = servers => panic!("loopback worker nodes stopped early: {:?}", outcomes.into_iter().map(|outcome| outcome.err().map(|error| error.to_string())).collect::<Vec<_>>()),
    }
}
