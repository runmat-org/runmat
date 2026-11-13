//! Parallel structural solid assembly scaling on a refined connected Tet4/Tet10 mesh.
//!
//! `cargo run --release -p runmat-analysis-fea --example assembly_scaling -- <levels> <tet4|tet10> <threads,..> [reps]`
//! Prints, per thread count, the minimum wall time of symbolic+numeric operator assembly
//! (stiffness, consistent mass, lumped mass) and checks that every thread count produces
//! bitwise-identical operators.

use std::{collections::HashMap, env, error::Error, time::Instant};

use runmat_analysis_fea::{
    assembly::solver_solid::assemble_solver_solid_operators_with_execution,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{LocalExecutionPlan, LocalThreadCount},
    properties::compile_solid_property_map,
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
        let node_count = mesh.topology.nodes.len() as u64;
        let element_count = mesh.topology.volume_elements.len() as u64;
        mesh.topology.regions[0].element_ids = (1..=element_count).collect();
        for topology in &mut mesh.topology.field_topologies {
            match topology.location {
                FieldTopologyLocation::Node => {
                    topology.ordered_entity_ids = (1..=node_count).collect();
                }
                FieldTopologyLocation::VolumeElement => {
                    topology.ordered_entity_ids = (1..=element_count).collect();
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

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let levels: usize = args.next().ok_or("expected refinement levels")?.parse()?;
    let order = match args.next().as_deref() {
        Some("tet4") => ElementOrder::Tet4,
        Some("tet10") => ElementOrder::Tet10,
        _ => return Err("expected tet4 or tet10".into()),
    };
    let threads = args
        .next()
        .ok_or("expected comma-separated thread counts")?
        .split(',')
        .map(str::parse::<u16>)
        .collect::<Result<Vec<_>, _>>()?;
    let repetitions: usize = args.next().map_or(Ok(3), |value| value.parse())?;
    let started = Instant::now();
    let tet4 = refine(connected_tetrahedron_solver_mesh(), levels)?;
    let mesh = if order == ElementOrder::Tet10 {
        elevate_connected_tet4_fixture(tet4)
    } else {
        tet4
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.materials.truncate(1);
    model.geometry_revision = u32::try_from(mesh.geometry.geometry_revision)?;
    model.material_assignments.truncate(1);
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::all_solids();
    let properties = compile_solid_property_map(&model, &mesh)?;
    println!(
        "mesh order={order:?} levels={levels} nodes={} dofs={} elements={} build_s={:.2}",
        mesh.topology.nodes.len(),
        mesh.topology.nodes.len() * 3,
        mesh.topology.volume_elements.len(),
        started.elapsed().as_secs_f64()
    );
    let mut reference = None;
    for &count in &threads {
        let plan = LocalExecutionPlan::from_thread_count(LocalThreadCount::new(count)?)?;
        let mut best = f64::INFINITY;
        for _ in 0..repetitions {
            let start = Instant::now();
            let operators =
                assemble_solver_solid_operators_with_execution(&mesh, &properties, plan)
                    .map_err(|error| format!("{error:?}"))?;
            best = best.min(start.elapsed().as_secs_f64());
            let digest = (
                operators.stiffness.values_identity()?,
                operators.consistent_mass.values_identity()?,
                operators
                    .lumped_mass_diag
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            match &reference {
                None => reference = Some(digest),
                Some(expected) if *expected != digest => {
                    return Err(format!("{count} threads changed the assembled operators").into())
                }
                Some(_) => {}
            }
            if count == threads[0] && best == start.elapsed().as_secs_f64() {
                println!("nnz={}", operators.stiffness.values().len());
            }
        }
        println!("threads={count} assembly_min_s={best:.4}");
    }
    Ok(())
}
