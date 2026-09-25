//! End-to-end physical Tet4/Tet10 static patch timing on a conforming connected mesh.
//!
//! Run with `cargo run --release -p runmat-analysis-fea --example connected_static_patch_cost -- 3 3 4 tet10`.
//! Refinement is outside the timed solve. Peak RSS includes process startup and all retained data,
//! so it is not an isolated FEA allocation or a resource-admission pass.

use std::{
    collections::{BTreeSet, HashMap},
    env,
    error::Error,
    mem::MaybeUninit,
    time::Instant,
};

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, FormulationRequirement,
    MaterialElasticityModel, PrescribedStructuralDofs, PrescribedValue, RegionTarget,
    StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    planning::resource::{LocalExecutionPlan, LocalThreadCount},
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
        mesh.resolved_request.resources.maximum_nodes =
            mesh.topology.nodes.len() as u64 + added_nodes;
        mesh.resolved_request.resources.maximum_elements = expected_elements;
    }
    for _ in 0..levels {
        let parents = std::mem::take(&mut mesh.topology.volume_elements);
        let mut children = Vec::with_capacity(parents.len() * 4);
        for parent in parents {
            let mut center = [0.0; 3];
            for &node_id in &parent.node_ids {
                let point = mesh.topology.nodes[node_id as usize - 1].coordinates_m;
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
                coordinates_m: center,
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
        for topology in &mut mesh.topology.field_topologies {
            match topology.location {
                FieldTopologyLocation::Node => {
                    topology.ordered_entity_ids = (1..=mesh.topology.nodes.len() as u64).collect();
                }
                FieldTopologyLocation::VolumeElement => {
                    topology.ordered_entity_ids =
                        (1..=mesh.topology.volume_elements.len() as u64).collect();
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
    elasticity.youngs_modulus_pa = 100.0;
    elasticity.poisson_ratio = 0.25;
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
            let coordinates = mesh.topology.nodes[node_id as usize - 1].coordinates_m;
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

#[cfg(unix)]
fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // getrusage writes the complete structure before it is read below.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let raw = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(raw)
    } else {
        raw.checked_mul(1024)
    }
}

#[cfg(not(unix))]
fn process_peak_rss_bytes() -> Option<u64> {
    None
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
            formulation_requirement: FormulationRequirement::Experimental,
            solver_mesh: Some(mesh.clone()),
            solver_mesh_artifact_path: Some("connected-static-patch-cost.rmmesh".into()),
            local_execution,
            ..LinearStaticSolveOptions::default()
        },
    )
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
                let expected = [0.1, 0.2, 0.3][axis] * node.coordinates_m[axis];
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
    timings.sort_by(f64::total_cmp);
    println!("os={},arch={},order={order:?},levels={levels},nodes={},elements={},threads={threads},repetitions={repetitions},min_s={:.6},median_s={:.6},max_displacement_error_m={maximum_displacement_error_m:.3e}",
        env::consts::OS, env::consts::ARCH, mesh.topology.nodes.len(), mesh.topology.volume_elements.len(), timings[0], timings[repetitions / 2]);
    if let Some(bytes) = process_peak_rss_bytes() {
        println!("process_peak_rss_bytes={bytes}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_fea::{
        constraints::CanonicalStructuralConstraintPlan,
        execution::StructuralLinearWorkload,
        planning::resource::{
            build_structural_resource_plan, ResourcePlanningError, StructuralSymbolicOperators,
        },
        post::fields::structural_observable_field_definitions,
        results::observation::{
            resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
        },
        solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
        topology::AnalysisProjection,
        FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    };

    #[test]
    fn connected_preflight_tracks_expanded_proof_and_upper_pattern_fallback() {
        for (levels, expected_dofs, expected_nonzeros) in [(4, 1_035, 27_765), (5, 4_107, 110_709)]
        {
            let mesh = refine(connected_tetrahedron_solver_mesh(), levels).unwrap();
            let model = patch_model(&mesh);
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            let constraints =
                CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
            let symbolic =
                StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh)
                    .unwrap();
            assert_eq!(symbolic.stiffness_upper_bound().rows(), expected_dofs);
            assert_eq!(
                symbolic.stiffness_upper_bound().column_indices().len(),
                expected_nonzeros
            );
            let field_definitions = structural_observable_field_definitions().unwrap();
            let observations = resolve_structural_observation_plan(
                &model.sequence.steps[0],
                &model,
                Some(&mesh),
                &constraints,
                &field_definitions,
                ObservationLimits::default(),
            )
            .unwrap();
            let ceiling = model.sequence.steps[0]
                .accuracy
                .resources
                .maximum_resident_bytes;
            let workload = StructuralLinearWorkload {
                schema_version: 1,
                step_id: model.sequence.steps[0].step_id.clone(),
                model,
                backend: ComputeBackend::Cpu,
                formulation_requirement: FormulationRequirement::Experimental,
                preconditioner: SpdPreconditionerKind::Jacobi,
                algebra_backend: LinearAlgebraBackendKind::CpuReference,
                observation_execution: ObservationExecutionPolicy::scalable(),
                require_solver_mesh_for_solid: true,
                initial_guess_transition: None,
            };
            let plan = build_structural_resource_plan(
                Some(&mesh),
                &constraints,
                &observations,
                &workload,
                &[],
                LocalExecutionPlan::serial(),
            )
            .unwrap();
            assert!(plan.request.memory_bytes <= ceiling);
            assert!(
                plan.request.memory_bytes > 32 * 1024 * 1024,
                "levels {levels}: {} bytes",
                plan.request.memory_bytes
            );

            if levels == 4 {
                let mut capped = workload;
                capped.model.sequence.steps[0]
                    .accuracy
                    .resources
                    .maximum_resident_bytes = 32 * 1024 * 1024;
                let error = build_structural_resource_plan(
                    Some(&mesh),
                    &constraints,
                    &observations,
                    &capped,
                    &[],
                    LocalExecutionPlan::serial(),
                )
                .expect_err("a ceiling below the resource reservation must reject early");
                assert!(matches!(
                    error,
                    ResourcePlanningError::AccuracyCeiling { .. }
                ));
            }
        }
    }

    #[test]
    fn refined_connected_tet10_preflight_reserves_and_rejects_a_lower_ceiling() {
        let mesh =
            elevate_connected_tet4_fixture(refine(connected_tetrahedron_solver_mesh(), 3).unwrap());
        mesh.validate_canonical().unwrap();
        let model = patch_model(&mesh);
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
        let fields = structural_observable_field_definitions().unwrap();
        let observations = resolve_structural_observation_plan(
            &model.sequence.steps[0],
            &model,
            Some(&mesh),
            &constraints,
            &fields,
            ObservationLimits::default(),
        )
        .unwrap();
        let mut workload = StructuralLinearWorkload {
            schema_version: 1,
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution: ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            initial_guess_transition: None,
        };
        let plan = build_structural_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            &[],
            LocalExecutionPlan::serial(),
        )
        .unwrap();
        assert!(plan.request.memory_bytes > 1);
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = plan.request.memory_bytes - 1;
        assert!(matches!(
            build_structural_resource_plan(
                Some(&mesh),
                &constraints,
                &observations,
                &workload,
                &[],
                LocalExecutionPlan::serial(),
            ),
            Err(ResourcePlanningError::AccuracyCeiling { .. })
        ));
    }

    #[test]
    fn connected_patch_serial_and_four_worker_fields_match_the_affine_solution() {
        let mesh = refine(connected_tetrahedron_solver_mesh(), 4).unwrap();
        let model = patch_model(&mesh);
        let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
        let scoped = solve_patch(
            &model,
            &mesh,
            LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
        )
        .unwrap();
        assert_static_field_parity(&serial, &scoped);
        assert_affine_displacement(&scoped, &mesh, 1.0e-7);
        assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
    }

    #[test]
    fn connected_patch_4096_elements_solves_with_parallel_parity_and_affine_accuracy() {
        let mesh = refine(connected_tetrahedron_solver_mesh(), 5).unwrap();
        let model = patch_model(&mesh);
        let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
        let scoped = solve_patch(
            &model,
            &mesh,
            LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
        )
        .unwrap();
        assert_static_field_parity(&serial, &scoped);
        assert_affine_displacement(&scoped, &mesh, 2.0e-7);
        assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
    }

    #[test]
    fn refined_connected_tet10_patch_solves_with_parallel_parity_and_affine_accuracy() {
        let mesh =
            elevate_connected_tet4_fixture(refine(connected_tetrahedron_solver_mesh(), 3).unwrap());
        mesh.validate_canonical().unwrap();
        assert_eq!(mesh.topology.volume_elements.len(), 256);
        let model = patch_model(&mesh);
        let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
        let scoped = solve_patch(
            &model,
            &mesh,
            LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
        )
        .unwrap();
        assert_static_field_parity(&serial, &scoped);
        assert_affine_displacement(&scoped, &mesh, 2.0e-7);
        assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
    }

    fn assert_static_field_parity(serial: &FeaRunResult, scoped: &FeaRunResult) {
        assert_eq!(serial.fields, scoped.fields, "published field parity");
        assert_eq!(serial.observations, scoped.observations);
        assert_eq!(
            serial.structural_static_balance,
            scoped.structural_static_balance
        );
        assert_eq!(
            serial.structural_constraint_result,
            scoped.structural_constraint_result
        );
        assert_eq!(
            serial.structural_connector_result,
            scoped.structural_connector_result
        );
    }

    fn assert_affine_displacement(
        result: &FeaRunResult,
        mesh: &SolverMeshArtifact,
        tolerance: f64,
    ) {
        let displacement = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        for (actual, node) in displacement.chunks_exact(3).zip(&mesh.topology.nodes) {
            for axis in 0..3 {
                let expected = [0.1, 0.2, 0.3][axis] * node.coordinates_m[axis];
                assert!(
                    (actual[axis] - expected).abs() <= tolerance,
                    "node {} axis {axis}: {} != {expected}",
                    node.node_id,
                    actual[axis]
                );
            }
        }
    }

    fn assert_affine_element_fields(
        result: &FeaRunResult,
        mesh: &SolverMeshArtifact,
        strain_tolerance: f64,
        stress_tolerance: f64,
    ) {
        let strain = result
            .field(FEA_FIELD_STRUCTURAL_STRAIN)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_STRESS)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let energy = result
            .field(FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(strain.len(), mesh.topology.volume_elements.len() * 6);
        assert_eq!(stress.len(), mesh.topology.volume_elements.len() * 6);
        assert_eq!(energy.len(), mesh.topology.volume_elements.len());
        for (element, (actual_strain, actual_stress)) in strain
            .chunks_exact(6)
            .zip(stress.chunks_exact(6))
            .enumerate()
        {
            for (component, (&actual, expected)) in actual_strain
                .iter()
                .zip([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= strain_tolerance,
                    "element {element} strain component {component}: {actual} != {expected}"
                );
            }
            for (component, (&actual, expected)) in actual_stress
                .iter()
                .zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0])
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= stress_tolerance,
                    "element {element} stress component {component}: {actual} != {expected}"
                );
            }
            assert!((energy[element] - 12.8).abs() <= stress_tolerance);
        }
        let total = result
            .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(total.len(), 1);
        assert!((total[0] - 12.8 / 6.0).abs() <= stress_tolerance);
    }
}
