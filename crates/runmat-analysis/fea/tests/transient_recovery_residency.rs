//! Isolated allocator evidence for the completed physical transient recovery preflight.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use runmat_analysis_core::{AnalysisModel, ExcitationTarget, FormulationRequirement};
use runmat_analysis_fea::{
    execution::{
        StructuralTransientBlockPlan, StructuralTransientBlockRequest,
        StructuralTransientNumericalReceipt, StructuralTransientResultAssemblyRequest,
        StructuralTransientSolverPlan, StructuralTransientSolverResult,
        StructuralTransientWorkload,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::transient::{
        prepare_transient_checkpoint_scope, recover_completed_transient_run, run_transient_block,
        CompletedStructuralTransientRunInput, StructuralTransientRunBlockInput,
    },
    planning::resource::build_structural_transient_recovery_resource_plan,
    solve::transient::StructuralTimeStepPolicy,
    ComputeBackend,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    PersistentEntityKind, SolverMeshArtifact, StableDigest,
};

struct CountingAllocator;

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);
static TRACKING: AtomicBool = AtomicBool::new(false);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(bytes(layout.size()), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            LIVE_BYTES.fetch_sub(bytes(layout.size()), Ordering::Relaxed);
            record_allocation(new_size);
        }
        replacement
    }
}

fn bytes(size: usize) -> u64 {
    u64::try_from(size).unwrap_or(u64::MAX)
}

fn record_allocation(size: usize) {
    let live = LIVE_BYTES
        .fetch_add(bytes(size), Ordering::Relaxed)
        .saturating_add(bytes(size));
    if TRACKING.load(Ordering::Relaxed) {
        PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
    }
}

#[cfg(unix)]
fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is writable storage for one `rusage` value.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: successful `getrusage` initialized the complete value.
    let maximum = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    Some(if cfg!(target_os = "macos") {
        maximum
    } else {
        maximum.saturating_mul(1024)
    })
}

#[cfg(not(unix))]
fn process_peak_rss_bytes() -> Option<u64> {
    None
}

fn add_face_region(mesh: &mut SolverMeshArtifact, face: usize, region: &str) {
    let provenance = &mut mesh.topology.boundary_faces[face].provenance;
    let entity = PersistentEntityId {
        kind: PersistentEntityKind::Face,
        source_topology_id: region.to_owned(),
        assembly_path: vec!["root".to_owned()],
    };
    if !provenance.contains(&entity) {
        provenance.push(entity);
        provenance.sort();
    }
}

fn physical_fixture_mesh(model: &AnalysisModel) -> SolverMeshArtifact {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    if let Some(assignment) = model.material_assignments.first() {
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: assignment.selected_region_id().unwrap().to_owned(),
            assembly_path: vec!["root".to_owned()],
        };
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
    }
    for region in model
        .constraints
        .iter()
        .filter_map(|condition| condition.target())
    {
        match region.region_id.as_str() {
            "free_vertex_y_support" => add_face_region(&mut mesh, 1, &region.region_id),
            "free_vertex_z_support" => add_face_region(&mut mesh, 2, &region.region_id),
            other => add_face_region(&mut mesh, 0, other),
        }
    }
    for action in &model.excitations {
        if let ExcitationTarget::Region(target) = &action.excitation().target {
            if target.region_id != "dynamic_force" {
                add_face_region(&mut mesh, 1, &target.region_id);
            }
        }
    }
    for assignment in &model.material_assignments {
        add_face_region(&mut mesh, 1, assignment.selected_region_id().unwrap());
        add_face_region(&mut mesh, 1, &assignment.assigned_material_id);
    }
    add_face_region(&mut mesh, 1, "tip");
    add_face_region(&mut mesh, 1, "dynamic_force");
    add_face_region(&mut mesh, 3, "dynamic_force");
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

fn connector_observation_model() -> AnalysisModel {
    use runmat_analysis_core::*;

    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.connectors = vec![AnalysisConnector {
        connector_id: "root_to_tip_dashpot".into(),
        kind: ConnectorKind::AxialDashpot {
            primary: RegionTarget::point("node:2"),
            secondary: RegionTarget::point("node:4"),
            damping: Damping::new(75.0),
        },
    }];
    model.sequence.steps[0].observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId("dashpot_force_history".into()),
        quantity: ObservationQuantity::ReactionForce,
        value_shape: ObservationQuantity::ReactionForce.canonical_value_shape(),
        target: ObservationTarget::Connector {
            connector_id: "root_to_tip_dashpot".into(),
        },
        source_location: FieldLocation::Connector,
        output_location: FieldLocation::Connector,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::EveryAcceptedIncrement,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    })];
    model
}

fn mixed_observation_model() -> AnalysisModel {
    use runmat_analysis_core::{
        FieldLocation, ObservationId, ObservationQuantity, ObservationRequest, ObservationTarget,
        RetentionPolicy,
    };

    let mut model = connector_observation_model();
    let ObservationRequest::History(connector_force) =
        model.sequence.steps[0].observations[0].clone()
    else {
        unreachable!("connector fixture has one history")
    };
    let mut displacement = connector_force.clone();
    displacement.observation_id = ObservationId("displacement_history".into());
    displacement.quantity = ObservationQuantity::Displacement;
    displacement.value_shape = displacement.quantity.canonical_value_shape();
    displacement.target = ObservationTarget::WholeModel;
    displacement.source_location = FieldLocation::Node;
    displacement.output_location = FieldLocation::Node;

    let mut stress = connector_force.clone();
    stress.observation_id = ObservationId("stress_history".into());
    stress.quantity = ObservationQuantity::Stress;
    stress.value_shape = stress.quantity.canonical_value_shape();
    stress.target = ObservationTarget::WholeModel;
    stress.source_location = FieldLocation::Element;
    stress.output_location = FieldLocation::Element;

    let mut connector_energy = connector_force;
    connector_energy.observation_id = ObservationId("dashpot_energy_history".into());
    connector_energy.quantity = ObservationQuantity::StrainEnergy;
    connector_energy.value_shape = connector_energy.quantity.canonical_value_shape();
    connector_energy.storage.retention = RetentionPolicy::LatestSamples { count: 2 };
    model.sequence.steps[0].observations.extend([
        ObservationRequest::History(displacement),
        ObservationRequest::History(stress),
        ObservationRequest::History(connector_energy),
    ]);
    model
}

fn measure_physical_recovery(
    case: &'static str,
    model: AnalysisModel,
    interval_count: usize,
) -> (u64, u64) {
    let mesh = physical_fixture_mesh(&model);
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        solver: StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
        residual_target: 1.0e-10,
    };
    let scope = prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
    let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let terminals = StructuralTransientBlockPlan::new(49)
        .unwrap()
        .terminal_schedule_ordinals(terminal)
        .unwrap();
    let mut predecessor = None;
    let mut segments = Vec::new();
    for terminal in terminals {
        let request =
            StructuralTransientBlockRequest::new(scope.clone(), predecessor.as_ref(), terminal)
                .unwrap();
        let segment = run_transient_block(StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &request,
            predecessor: predecessor.as_ref(),
            modal_state: None,
            observation_cursors: Vec::new(),
        })
        .unwrap();
        predecessor = Some(segment.manifest().terminal_checkpoint.clone());
        segments.push(segment);
    }
    let request = StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
    let result = StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
    let receipt = StructuralTransientNumericalReceipt::new(&result, &request, &segments).unwrap();
    drop(segments);
    drop(predecessor);
    drop(request);
    drop(scope);

    let plan = build_structural_transient_recovery_resource_plan(
        &workload, &mesh, None, &result, &receipt,
    )
    .unwrap();
    let baseline = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    TRACKING.store(true, Ordering::Relaxed);
    let recovered = recover_completed_transient_run(CompletedStructuralTransientRunInput {
        workload: &workload,
        solver_mesh: &mesh,
        modal_state: None,
        result,
        numerical_receipt: &receipt,
    })
    .unwrap();
    TRACKING.store(false, Ordering::Relaxed);
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    assert!(recovered.time_points_s.len() > interval_count);
    assert_eq!(
        recovered.run.observation_histories.len(),
        workload.model.sequence.steps[0].observations.len()
    );
    for (history, request) in recovered
        .run
        .observation_histories
        .iter()
        .zip(&workload.model.sequence.steps[0].observations)
    {
        let runmat_analysis_core::ObservationRequest::History(spec) = request else {
            unreachable!("recovery fixture only authors histories")
        };
        let expected = match &spec.storage.retention {
            runmat_analysis_core::RetentionPolicy::Canonical => recovered.time_points_s.len(),
            runmat_analysis_core::RetentionPolicy::LatestSamples { count } => {
                recovered.time_points_s.len().min(*count)
            }
            _ => unreachable!("recovery fixture only uses canonical or latest retention"),
        };
        assert_eq!(history.samples().len(), expected);
    }
    assert!(
        peak <= plan.request.memory_bytes,
        "measured live heap peak {peak} exceeds recovery request {} (baseline {baseline})",
        plan.request.memory_bytes
    );
    eprintln!(
        "transient recovery ({case}, {interval_count} intervals): baseline_live_heap={baseline} peak_live_heap={peak} process_peak_rss={:?} planned_memory={}",
        process_peak_rss_bytes(), plan.request.memory_bytes
    );
    (peak, plan.request.memory_bytes)
}

#[test]
fn completed_physical_recovery_heap_peak_fits_planned_memory() {
    let (short_peak, short_plan) = measure_physical_recovery(
        "direct",
        fixture_model(FixtureId::StructuralTransientTet4Reference),
        24,
    );
    let (long_peak, long_plan) = measure_physical_recovery(
        "direct",
        fixture_model(FixtureId::StructuralTransientTet4Reference),
        240,
    );
    assert!(long_peak > short_peak);
    assert!(long_plan > short_plan);
    let (_, connector_observation_plan) =
        measure_physical_recovery("connector observation", connector_observation_model(), 24);
    assert!(connector_observation_plan > short_plan);
    let (mixed_peak, mixed_plan) =
        measure_physical_recovery("mixed observations", mixed_observation_model(), 240);
    assert!(mixed_peak > long_peak);
    assert!(mixed_plan > long_plan);
}
