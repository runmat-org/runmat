//! Independent structural recovery references.
use super::*;
use runmat_meshing_core::StableDigest;

pub(super) fn curved_interior_tet10_patch_recovers_affine_stress_and_energy() {
    use crate::assembly::elements::solid::tetrahedron10::Tetrahedron10ElementGeometry;

    let mut mesh = structured_cantilever(2, ElementOrder::Tet10);
    let corner_count = 3_usize.pow(3);
    let moved = mesh
        .topology
        .nodes
        .iter()
        .position(|node| {
            node.node_id as usize > corner_count
                && node
                    .coordinates_m
                    .canonical_value()
                    .iter()
                    .all(|coordinate| *coordinate > 0.0 && *coordinate < 1.0)
        })
        .expect("structured Tet10 mesh has an interior midside node");
    let containing_element = mesh
        .topology
        .volume_elements
        .iter()
        .find(|element| {
            element
                .node_ids
                .contains(&mesh.topology.nodes[moved].node_id)
        })
        .expect("interior midside node has an owning element")
        .node_ids
        .clone();
    let volume = |mesh: &SolverMeshArtifact| {
        let nodes_m = std::array::from_fn(|index| {
            mesh.topology.nodes[containing_element[index] as usize - 1]
                .coordinates_m
                .canonical_value()
        });
        Tetrahedron10ElementGeometry { nodes_m }
            .volume_m3()
            .unwrap()
    };
    let straight_volume = volume(&mesh);
    {
        let quantity_change = 0.02;
        let quantity_index = 0;
        let quantity_slot = &mut mesh.topology.nodes[moved].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    {
        let quantity_change = 0.015;
        let quantity_index = 1;
        let quantity_slot = &mut mesh.topology.nodes[moved].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] -= quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    {
        let quantity_change = 0.01;
        let quantity_index = 2;
        let quantity_slot = &mut mesh.topology.nodes[moved].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!((volume(&mesh) - straight_volume).abs() > 1.0e-8);
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

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
    replace_fixture_excitations(&mut model, Vec::new());
    let boundary_nodes = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(boundary_nodes.len() < mesh.topology.nodes.len());
    model.constraints = boundary_nodes
        .into_iter()
        .map(|node_id| {
            let point = mesh.topology.nodes[node_id as usize - 1]
                .coordinates_m
                .canonical_value();
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("affine_boundary_{node_id}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{node_id}")),
                    translations: [
                        (StructuralTranslationDof::Ux, 0.1 * point[0]),
                        (StructuralTranslationDof::Uy, 0.2 * point[1]),
                        (StructuralTranslationDof::Uz, 0.3 * point[2]),
                    ]
                    .into_iter()
                    .map(|(dof, value)| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(value)),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("curved interior Tet10 patch should solve");
    let displacement = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(displacement.len(), mesh.topology.nodes.len() * 3);
    for (actual, node) in displacement.chunks_exact(3).zip(&mesh.topology.nodes) {
        for (&actual, expected) in actual.iter().zip([
            0.1 * node.coordinates_m.canonical_value()[0],
            0.2 * node.coordinates_m.canonical_value()[1],
            0.3 * node.coordinates_m.canonical_value()[2],
        ]) {
            assert!(
                (actual - expected).abs() < 1.0e-8,
                "u {actual} vs {expected}"
            );
        }
    }
    let stress = result
        .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(stress.len(), mesh.topology.volume_elements.len() * 4 * 6);
    for point in stress.chunks_exact(6) {
        for (&actual, expected) in point.iter().zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0]) {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "stress {actual} vs {expected}"
            );
        }
    }
    let energy_density = result
        .field(FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(energy_density.len(), mesh.topology.volume_elements.len());
    assert!(energy_density
        .iter()
        .all(|value| (value - 12.8).abs() < 1.0e-5));
    let total_energy = result
        .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(total_energy.len(), 1);
    assert!((total_energy[0] - 12.8).abs() < 1.0e-5);
}
