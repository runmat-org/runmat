//! End-to-end physical Tet4 static patch timing on a conforming connected mesh.
//!
//! Run with `cargo run --release -p runmat-analysis-fea --example connected_static_patch_cost -- 3 3 4`.
//! Refinement is outside the timed solve. Peak RSS includes process startup and all retained data,
//! so it is not an isolated FEA allocation or a resource-admission pass.

use std::{collections::HashMap, env, error::Error, mem::MaybeUninit, time::Instant};

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
    fixtures::connected_tetrahedron_solver_mesh, solver_volume_element_identity,
    CanonicalMeshingContract, FieldTopologyLocation, MeshNeighbor, SolverMeshArtifact,
    SolverMeshNode, StableDigest,
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

fn patch_model() -> runmat_analysis_core::AnalysisModel {
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
    model.constraints = (1..=4)
        .map(|node_id| {
            let coordinates = match node_id {
                1 => [0.0, 0.0, 0.0],
                2 => [1.0, 0.0, 0.0],
                3 => [0.0, 1.0, 0.0],
                4 => [0.0, 0.0, 1.0],
                _ => unreachable!(),
            };
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
    if args.next().is_some() || levels > 5 || repetitions == 0 {
        return Err("expected refinement levels 0..=5, positive repetitions, and optional admitted thread count".into());
    }
    let local_execution = LocalExecutionPlan::from_thread_count(LocalThreadCount::new(threads)?)?;
    let mesh = refine(connected_tetrahedron_solver_mesh(), levels)?;
    let model = patch_model();
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
    println!("os={},arch={},levels={levels},nodes={},elements={},threads={threads},repetitions={repetitions},min_s={:.6},median_s={:.6},max_displacement_error_m={maximum_displacement_error_m:.3e}",
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
        FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_STRAIN,
        FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    };

    #[test]
    fn connected_patch_serial_and_four_worker_fields_match_the_affine_solution() {
        let mesh = refine(connected_tetrahedron_solver_mesh(), 4).unwrap();
        let model = patch_model();
        let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
        let scoped = solve_patch(
            &model,
            &mesh,
            LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
        )
        .unwrap();
        for field_id in [
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            FEA_FIELD_STRUCTURAL_STRAIN,
            FEA_FIELD_STRUCTURAL_STRESS,
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
        ] {
            let serial_values = serial.field(field_id).unwrap().as_host_f64().unwrap();
            let scoped_values = scoped.field(field_id).unwrap().as_host_f64().unwrap();
            assert_eq!(serial_values.len(), scoped_values.len());
            assert_eq!(serial_values, scoped_values, "{field_id} parity");
        }
        let displacement = scoped
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        for (actual, node) in displacement.chunks_exact(3).zip(&mesh.topology.nodes) {
            for axis in 0..3 {
                let expected = [0.1, 0.2, 0.3][axis] * node.coordinates_m[axis];
                assert!(
                    (actual[axis] - expected).abs() <= 1.0e-7,
                    "node {} axis {axis}: {} != {expected}",
                    node.node_id,
                    actual[axis]
                );
            }
        }
    }

    #[test]
    fn connected_patch_rejects_unsupported_solve_size_instead_of_publishing_zeros() {
        let mesh = refine(connected_tetrahedron_solver_mesh(), 5).unwrap();
        let error = solve_patch(&patch_model(), &mesh, LocalExecutionPlan::serial())
            .expect_err("an unsupported solve must not publish a zero displacement field");
        assert!(
            error.to_string().contains("FEA_INVALID_REAL_SYSTEM"),
            "{error}"
        );
    }
}
