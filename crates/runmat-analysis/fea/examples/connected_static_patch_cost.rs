//! End-to-end physical Tet4/Tet10 static patch timing on a conforming connected mesh.
//!
//! Run with `cargo run --release -p runmat-analysis-fea --example connected_static_patch_cost -- 3 3 4 tet10`.
//! Refinement and resource planning are outside the timed solve. The printed factorization
//! reservation is the conservative coercivity-admission envelope, not observed factor storage.
//! The counted Rust heap peak begins after refinement and planning, but does not include native
//! allocations. Peak RSS includes process startup and all retained data; neither measure alone
//! proves the production worker's complete resource accounting.

use std::{
    collections::{BTreeSet, HashMap},
    env,
    error::Error,
    time::Instant,
};

#[path = "../support/peak_allocator.rs"]
mod peak_allocator;
use peak_allocator::{begin_measurement, end_measurement, process_peak_rss_bytes};

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, MaterialElasticityModel,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::StructuralLinearWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{build_structural_resource_plan, LocalExecutionPlan, LocalThreadCount},
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    run_linear_static_with_options, ComputeBackend, FeaRunError, FeaRunResult,
    LinearStaticSolveOptions, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
};
use runmat_meshing_core::{
    fixtures::{connected_tetrahedron_solver_mesh, elevate_connected_tet4_fixture},
    solver_volume_element_identity, CanonicalMeshingContract, ElementOrder, FieldTopologyLocation,
    MeshNeighbor, SolverMeshArtifact, SolverMeshNode, StableDigest,
};

fn face_key(nodes: &[u64], omitted: usize) -> [u64; 3] {
    let mut face = [0; 3];
    let mut output = 0;
    for (index, &node) in nodes.iter().enumerate() {
        if index != omitted {
            face[output] = node;
            output += 1;
        }
    }
    face.sort_unstable();
    face
}

fn refine(
    mut mesh: SolverMeshArtifact,
    levels: usize,
) -> Result<SolverMeshArtifact, Box<dyn Error>> {
    if levels > 0 {
        let expected_elements =
            (mesh.topology.volume_elements.len() as u64) * 4_u64.pow(levels as u32);
        let added_nodes = (expected_elements - mesh.topology.volume_elements.len() as u64) / 3;
        mesh.resolved_request.limits.maximum_nodes = mesh.topology.nodes.len() as u64 + added_nodes;
        mesh.resolved_request.limits.maximum_elements = expected_elements;
    }
    for _ in 0..levels {
        let parents = std::mem::take(&mut mesh.topology.volume_elements);
        let mut children = Vec::with_capacity(parents.len() * 4);
        for parent in parents {
            let mut center = [0.0; 3];
            for &node_id in &parent.node_ids {
                let point = mesh.topology.nodes[node_id as usize - 1]
                    .coordinates_m
                    .canonical_value();
                for axis in 0..3 {
                    center[axis] += point[axis] / 4.0;
                }
            }
            let center_id = mesh.topology.nodes.len() as u64 + 1;
            let mut identity_bytes = [0xa5; 32];
            identity_bytes[..8].copy_from_slice(&center_id.to_le_bytes());
            mesh.topology.nodes.push(SolverMeshNode {
                node_id: center_id,
                stable_identity: StableDigest::from_bytes(identity_bytes),
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(center),
                provenance: parent.provenance.clone(),
                exact_parameters: Vec::new(),
            });
            for omitted in 0..4 {
                let mut child = parent.clone();
                child.node_ids[omitted] = center_id;
                child.element_id = children.len() as u64 + 1;
                child.stable_identity =
                    solver_volume_element_identity(std::array::from_fn(|index| {
                        mesh.topology.nodes[child.node_ids[index] as usize - 1].stable_identity
                    }));
                children.push(child);
            }
        }
        mesh.topology.volume_elements = children;

        let mut owners: HashMap<[u64; 3], Vec<(u64, u8)>> = HashMap::new();
        for element in &mesh.topology.volume_elements {
            for omitted in 0..4 {
                owners
                    .entry(face_key(&element.node_ids, omitted))
                    .or_default()
                    .push((element.element_id, omitted as u8));
            }
        }
        mesh.topology.neighbors = mesh
            .topology
            .volume_elements
            .iter()
            .flat_map(|element| {
                (0..4).map(|omitted| {
                    let face_owners = &owners[&face_key(&element.node_ids, omitted)];
                    MeshNeighbor {
                        element_id: element.element_id,
                        local_face_index: omitted as u8,
                        adjacent_element_id: face_owners
                            .iter()
                            .find(|(id, _)| *id != element.element_id)
                            .map(|(id, _)| *id),
                    }
                })
            })
            .collect();
        for face in &mut mesh.topology.boundary_faces {
            let mut key = [face.node_ids[0], face.node_ids[1], face.node_ids[2]];
            key.sort_unstable();
            face.adjacent_volume_element_ids = owners[&key].iter().map(|(id, _)| *id).collect();
        }
        mesh.topology.regions[0].element_ids =
            (1..=mesh.topology.volume_elements.len() as u64).collect();
        let topo = &mut *mesh.topology;
        for topology in &mut topo.field_topologies {
            match topology.location {
                FieldTopologyLocation::Node => {
                    topology.ordered_entity_ids = (1..=topo.nodes.len() as u64).collect();
                }
                FieldTopologyLocation::VolumeElement => {
                    topology.ordered_entity_ids = (1..=topo.volume_elements.len() as u64).collect();
                }
                FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => {}
            }
        }
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest()?;
        mesh.validate_canonical()?;
    }
    Ok(mesh)
}

fn patch_model(mesh: &SolverMeshArtifact) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let MaterialElasticityModel::IsotropicLinear(elasticity) = model.materials[0]
        .elasticity
        .as_mut()
        .expect("fixture elasticity")
    else {
        panic!("expected isotropic linear elasticity");
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.25);
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    let boundary_nodes = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    model.constraints = boundary_nodes
        .into_iter()
        .map(|node_id| {
            let coordinates = mesh.topology.nodes[node_id as usize - 1]
                .coordinates_m
                .canonical_value();
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("affine_boundary_node_{node_id}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{node_id}")),
                    translations: [
                        StructuralTranslationDof::Ux,
                        StructuralTranslationDof::Uy,
                        StructuralTranslationDof::Uz,
                    ]
                    .into_iter()
                    .zip([0.1, 0.2, 0.3])
                    .enumerate()
                    .map(|(axis, (dof, gradient))| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(
                            gradient * coordinates[axis],
                        )),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    model
}

fn solve_patch(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
    local_execution: LocalExecutionPlan,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_options(
        model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            solver_mesh_artifact_path: Some("connected-static-patch-cost.rmmesh".into()),
            local_execution,
            ..LinearStaticSolveOptions::default()
        },
    )
}

fn plan_patch(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
    local_execution: LocalExecutionPlan,
) -> Result<runmat_analysis_fea::planning::resource::FeaResourcePlan, Box<dyn Error>> {
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(model, Some(mesh))?;
    let fields = structural_static_observable_field_definitions()?;
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        model,
        Some(mesh),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )?;
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: ComputeBackend::Cpu,
        preconditioner: runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend:
            runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    Ok(build_structural_resource_plan(
        Some(mesh),
        &constraints,
        &observations,
        &workload,
        &[],
        local_execution,
        None,
    )?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let levels: usize = args.next().ok_or("expected refinement levels")?.parse()?;
    let repetitions: usize = args.next().ok_or("expected repetition count")?.parse()?;
    let threads: u16 = args.next().unwrap_or_else(|| "1".into()).parse()?;
    let order = match args.next().as_deref() {
        None | Some("tet4") => ElementOrder::Tet4,
        Some("tet10") => ElementOrder::Tet10,
        Some(_) => return Err("expected element order tet4 or tet10".into()),
    };
    if args.next().is_some() || levels > 5 || repetitions == 0 {
        return Err("expected refinement levels 0..=5, positive repetitions, optional admitted thread count, and optional tet4/tet10 order".into());
    }
    let local_execution = LocalExecutionPlan::from_thread_count(LocalThreadCount::new(threads)?)?;
    let tet4_mesh = refine(connected_tetrahedron_solver_mesh(), levels)?;
    let mesh = if order == ElementOrder::Tet10 {
        let elevated = elevate_connected_tet4_fixture(tet4_mesh);
        elevated.validate_canonical()?;
        elevated
    } else {
        tet4_mesh
    };
    let model = patch_model(&mesh);
    let plan = plan_patch(&model, &mesh, local_execution)?;
    let baseline_live_heap_bytes = begin_measurement();
    let mut timings = Vec::with_capacity(repetitions);
    let mut maximum_displacement_error_m = 0.0_f64;
    for _ in 0..repetitions {
        let start = Instant::now();
        let result = solve_patch(&model, &mesh, local_execution)?;
        timings.push(start.elapsed().as_secs_f64());
        let displacement = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .ok_or("missing displacement field")?
            .as_host_f64()
            .ok_or("non-host displacement field")?;
        for (actual, node) in displacement.chunks_exact(3).zip(&mesh.topology.nodes) {
            for axis in 0..3 {
                let expected = [0.1, 0.2, 0.3][axis] * node.coordinates_m.canonical_value()[axis];
                let error = (actual[axis] - expected).abs();
                maximum_displacement_error_m = maximum_displacement_error_m.max(error);
                if error > 1.0e-6 {
                    return Err(format!(
                        "node {} axis {axis}: {} != {expected}; first displacements: {:?}",
                        node.node_id,
                        actual[axis],
                        &displacement[..displacement.len().min(15)]
                    )
                    .into());
                }
            }
        }
    }
    let peak_live_heap_bytes = end_measurement();
    let incremental_heap_peak_bytes = peak_live_heap_bytes.saturating_sub(baseline_live_heap_bytes);
    let numerical_request_bytes = plan
        .request
        .memory_bytes
        .checked_sub(plan.breakdown.worker_overhead_bytes.0)
        .ok_or("worker overhead exceeds admitted memory")?;
    if incremental_heap_peak_bytes > numerical_request_bytes {
        return Err(format!(
            "counted Rust heap growth {incremental_heap_peak_bytes} exceeds admitted numerical memory {numerical_request_bytes}"
        )
        .into());
    }
    timings.sort_by(f64::total_cmp);
    println!("os={},arch={},order={order:?},levels={levels},nodes={},elements={},threads={threads},repetitions={repetitions},min_s={:.6},median_s={:.6},max_displacement_error_m={maximum_displacement_error_m:.3e}",
        env::consts::OS, env::consts::ARCH, mesh.topology.nodes.len(), mesh.topology.volume_elements.len(), timings[0], timings[repetitions / 2]);
    println!(
        "resource_plan_memory_bytes={},resource_plan_scratch_bytes={},resource_plan_wall_millis={}",
        plan.request.memory_bytes, plan.request.scratch_bytes, plan.request.max_wall_millis
    );
    println!(
        "resource_plan_breakdown_topology_bytes={},csr_value_bytes={},assembly_contribution_bytes={},factorization_bytes={},constraint_bytes={},worker_overhead_bytes={}",
        plan.breakdown.topology_bytes.0,
        plan.breakdown.csr_value_bytes.0,
        plan.breakdown.assembly_contribution_bytes.0,
        plan.breakdown.factorization_bytes.0,
        plan.breakdown.constraint_bytes.0,
        plan.breakdown.worker_overhead_bytes.0,
    );
    println!(
        "rust_heap_baseline_live_bytes={baseline_live_heap_bytes},rust_heap_peak_live_bytes={peak_live_heap_bytes},rust_heap_incremental_peak_bytes={incremental_heap_peak_bytes},resource_plan_numerical_memory_bytes={numerical_request_bytes}"
    );
    if let Some(bytes) = process_peak_rss_bytes() {
        println!("process_peak_rss_bytes={bytes}");
    }
    Ok(())
}

#[cfg(test)]
#[path = "connected_static_patch_cost/tests.rs"]
mod tests;
