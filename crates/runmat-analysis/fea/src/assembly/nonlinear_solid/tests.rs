use super::*;
use crate::{
    algebra::spmv,
    assembly::{assemble_structural_system_borrowed, StructuralFormulationRequirement},
    excitation::RealizedExcitations,
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
};
use runmat_analysis_core::{
    MaterialElasticityModel, MaterialMechanicalModel, MaterialPlasticModel,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, BoundaryTriangleOrder};

mod heterogeneous;
mod integration;
mod nonuniform;
mod total_lagrangian;

pub(crate) fn fixture(
    order: ElementOrder,
    plastic: bool,
) -> (AnalysisModel, SolverMeshArtifact, AssemblySummary) {
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints.clear();
    model.materials[0].elasticity = Some(MaterialElasticityModel::IsotropicLinear(
        MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(1200.0),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
        },
    ));
    model.materials[0].plastic = if plastic {
        Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(5.0),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(30.0),
        })
    } else {
        None
    };
    let initial = assemble_structural_system_borrowed(
        &model,
        &RealizedExcitations::default(),
        Some(&mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    (model, mesh, initial)
}

fn affine_displacement(
    mesh: &SolverMeshArtifact,
    initial: &AssemblySummary,
    strain: [f64; 6],
) -> Vec<f64> {
    let mut u = vec![0.0; initial.dof_count];
    let gradient = [
        [strain[0], strain[5] / 2.0, strain[4] / 2.0],
        [strain[5] / 2.0, strain[1], strain[3] / 2.0],
        [strain[4] / 2.0, strain[3] / 2.0, strain[2]],
    ];
    for (i, node) in mesh.topology.nodes.iter().enumerate() {
        for (axis, kind) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .enumerate()
        {
            u[initial.structural_dof_layout.index(i, kind).unwrap()] = gradient[axis]
                .iter()
                .zip(node.coordinates_m.canonical_value())
                .map(|(g, x)| g * x)
                .sum();
        }
    }
    u
}

// Independent planar-face consistent traction integral. Tri3: area/3 at corners;
// affine Tri6: zero at corners, area/3 at each midside node. No volume B/tangent oracle.
fn constant_stress_force(
    mesh: &SolverMeshArtifact,
    initial: &AssemblySummary,
    stress: [f64; 6],
) -> Vec<f64> {
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.node_id, (i, n.coordinates_m.canonical_value())))
        .collect::<BTreeMap<_, _>>();
    let mut force = vec![0.0; initial.dof_count];
    let sigma = [
        [stress[0], stress[5], stress[4]],
        [stress[5], stress[1], stress[3]],
        [stress[4], stress[3], stress[2]],
    ];
    for face in &mesh.topology.boundary_faces {
        let x = [
            nodes[&face.node_ids[0]].1,
            nodes[&face.node_ids[1]].1,
            nodes[&face.node_ids[2]].1,
        ];
        let a: [f64; 3] = std::array::from_fn(|i| x[1][i] - x[0][i]);
        let b: [f64; 3] = std::array::from_fn(|i| x[2][i] - x[0][i]);
        let mut area = [
            (a[1] * b[2] - a[2] * b[1]) / 2.0,
            (a[2] * b[0] - a[0] * b[2]) / 2.0,
            (a[0] * b[1] - a[1] * b[0]) / 2.0,
        ];
        let owner = mesh
            .topology
            .volume_elements
            .iter()
            .find(|e| e.element_id == face.adjacent_volume_element_ids[0])
            .unwrap();
        let centroid: [f64; 3] = std::array::from_fn(|i| {
            owner.node_ids[..4]
                .iter()
                .map(|id| nodes[id].1[i])
                .sum::<f64>()
                / 4.0
        });
        if area
            .iter()
            .enumerate()
            .map(|(i, n)| n * (centroid[i] - x[0][i]))
            .sum::<f64>()
            > 0.0
        {
            area = area.map(|x| -x);
        }
        for (local, id) in face.node_ids.iter().enumerate() {
            let weight = match face.order {
                BoundaryTriangleOrder::Tri3 => 1.0 / 3.0,
                BoundaryTriangleOrder::Tri6 if local >= 3 => 1.0 / 3.0,
                BoundaryTriangleOrder::Tri6 => 0.0,
            };
            for (axis, kind) in [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ]
            .into_iter()
            .enumerate()
            {
                force[initial
                    .structural_dof_layout
                    .index(nodes[id].0, kind)
                    .unwrap()] += weight
                    * sigma[axis]
                        .iter()
                        .zip(area)
                        .map(|(s, n)| s * n)
                        .sum::<f64>();
            }
        }
    }
    force
}

#[test]
fn elastic_constant_strain_matches_independent_stress_energy_and_traction_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, false);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let strain = [0.01, -0.002, 0.003, 0.004, -0.006, 0.008];
        // lambda=mu=480 Pa; engineering shear stress=mu*gamma.
        let stress = [14.88, 3.36, 8.16, 1.92, -2.88, 3.84];
        let committed = basis.zero_state();
        let before = committed.clone();
        let evaluation = basis
            .evaluate(&affine_displacement(&mesh, &initial, strain), &committed)
            .unwrap();
        assert_eq!(committed, before);
        for point in &evaluation.point_evidence {
            for i in 0..6 {
                assert!((point.total_engineering_strain[i] - strain[i]).abs() < 1e-12);
                assert!((point.stress_pa[i] - stress[i]).abs() < 1e-9);
            }
            assert_eq!(point.plastic_dissipation_increment_pa, 0.0);
            assert_eq!(point.hardening_energy_density_pa, 0.0);
        }
        let expected_force = constant_stress_force(&mesh, &initial, stress);
        for (actual, expected) in evaluation.internal_force.iter().zip(expected_force) {
            assert!(
                (actual - expected).abs() < 1e-9,
                "order={order:?}, actual={actual}, expected={expected}"
            );
        }
        let energy_density = 0.5 * strain.iter().zip(stress).map(|(e, s)| e * s).sum::<f64>();
        let evidence = &evaluation.element_evidence[0];
        assert!((evidence.volume_m3 - 1.0 / 6.0).abs() < 1e-13);
        assert!((evidence.elastic_energy_j.canonical_value() - energy_density / 6.0).abs() < 1e-12);
        assert_eq!(evaluation.trial_state, committed);
    }
}

#[test]
fn assembled_consistent_tangent_differentiates_internal_force_from_same_committed_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, true);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let committed = basis.zero_state();
        let before = committed.clone();
        let u = affine_displacement(&mesh, &initial, [0.08, 0.0, 0.0, 0.01, 0.02, -0.01]);
        let evaluation = basis.evaluate(&u, &committed).unwrap();
        assert!(evaluation
            .trial_state
            .points
            .iter()
            .all(|p| p.equivalent_plastic_strain > 0.0));
        let direction = (0..u.len())
            .map(|i| (i as f64 + 0.3).sin())
            .collect::<Vec<_>>();
        let h = 1e-7;
        let plus = u
            .iter()
            .zip(&direction)
            .map(|(u, d)| u + h * d)
            .collect::<Vec<_>>();
        let minus = u
            .iter()
            .zip(&direction)
            .map(|(u, d)| u - h * d)
            .collect::<Vec<_>>();
        let plus = basis.evaluate(&plus, &committed).unwrap();
        let minus = basis.evaluate(&minus, &committed).unwrap();
        let exact = spmv(&evaluation.tangent, &direction).unwrap();
        let error = exact
            .iter()
            .zip(plus.internal_force.iter().zip(&minus.internal_force))
            .map(|(expected, (p, m))| ((p - m) / (2.0 * h) - expected).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = exact.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            error / norm < 1e-7,
            "order={order:?}, relative={}",
            error / norm
        );
        assert_eq!(committed, before);
        assert_eq!(
            basis.evaluate(&u, &committed).unwrap().trial_state,
            evaluation.trial_state
        );
    }
}

#[test]
fn foreign_history_and_changed_material_authority_are_rejected() {
    let (model, mesh, initial) = fixture(ElementOrder::Tet10, true);
    let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
    let mut history = basis.zero_state();
    history.point_identities.reverse();
    assert!(basis.validate_state(&history).is_err());
    let mut history = basis.zero_state();
    history.points[0].equivalent_plastic_strain = -1.0;
    assert!(basis.validate_state(&history).is_err());
    let mut changed = model.clone();
    changed.materials[0].plastic = None;
    assert!(CanonicalNonlinearSolidBasis::prepare(&changed, &mesh, &initial).is_err());
    let mut history = basis.zero_state();
    history.points.pop();
    assert!(basis.validate_state(&history).is_err());
    assert!(basis.evaluate(&[0.0; 1], &basis.zero_state()).is_err());
}

#[test]
fn accumulated_work_and_accepted_replay_are_material_owned_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, true);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let u = affine_displacement(&mesh, &initial, [0.08, 0.0, 0.0, 0.01, 0.02, -0.01]);
        let evaluated = basis.evaluate(&u, &basis.zero_state()).unwrap();
        basis
            .validate_replayed_history(&u, &evaluated.trial_state)
            .unwrap();
        let integrated_increment = evaluated
            .element_evidence
            .iter()
            .map(|e| e.plastic_dissipation_increment_j)
            .sum::<f64>();
        let accumulated = basis
            .accumulated_plastic_dissipation_j(&evaluated.trial_state)
            .unwrap();
        assert!((accumulated - integrated_increment).abs() < 1e-12);
        let continued = basis.evaluate(&u, &evaluated.trial_state).unwrap();
        assert!(
            (basis
                .accumulated_plastic_dissipation_j(&continued.trial_state)
                .unwrap()
                - accumulated)
                .abs()
                < 1e-12
        );
        // Zero committed history cannot be relabeled accepted at a plastically active strain.
        assert!(basis
            .validate_replayed_history(&u, &basis.zero_state())
            .is_err());
        let cold: NonlinearSolidState =
            serde_json::from_slice(&serde_json::to_vec(&evaluated.trial_state).unwrap()).unwrap();
        basis.validate_replayed_history(&u, &cold).unwrap();
        assert_eq!(cold, evaluated.trial_state);
    }
}
