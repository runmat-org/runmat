//! Connected two-material patch with a conformal plane at x = 1/2.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, EvidenceConfidence, MaterialAssignment,
    MaterialElasticityModel, PrescribedStructuralDofs, PrescribedValue, RegionTarget,
    StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{
    CanonicalMeshingContract, ElementOrder, MeshRegion, PersistentEntityKind, SolverMeshArtifact,
    StableDigest,
};

use super::{entity, replace_fixture_excitations, structured_cantilever};
use crate::{
    contracts::{
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    },
    fixtures::{fixture_model, FixtureExcitation, FixtureExcitationKind, FixtureId},
    ComputeBackend, LinearStaticSolveOptions,
};

#[path = "interface/manufactured.rs"]
mod manufactured;

fn constrained_modulus(youngs_modulus_pa: f64, poisson_ratio: f64) -> f64 {
    youngs_modulus_pa * (1.0 - poisson_ratio)
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
}

fn transverse_stress(youngs_modulus_pa: f64, poisson_ratio: f64, axial_strain: f64) -> f64 {
    youngs_modulus_pa * poisson_ratio * axial_strain
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
}

fn interface_mesh(divisions: usize, order: ElementOrder) -> SolverMeshArtifact {
    assert!(divisions > 0 && divisions.is_multiple_of(2));
    let mut mesh = structured_cantilever(divisions, order);
    let left_region = entity(PersistentEntityKind::Region, "left_layer");
    let right_region = entity(PersistentEntityKind::Region, "right_layer");
    let mut groups: BTreeMap<_, Vec<u64>> = BTreeMap::new();
    let topo = &mut *mesh.topology;
    for element in &mut topo.volume_elements {
        let x = element.node_ids[..4]
            .iter()
            .map(|id| topo.nodes[*id as usize - 1].coordinates_m.canonical_value()[0])
            .collect::<Vec<_>>();
        element.region_id = if x.iter().all(|value| *value <= 0.5) {
            left_region.clone()
        } else if x.iter().all(|value| *value >= 0.5) {
            right_region.clone()
        } else {
            panic!("an element crosses the analytical material interface");
        };
        groups
            .entry(element.region_id.clone())
            .or_default()
            .push(element.element_id);
    }
    assert_eq!(groups.len(), 2);
    assert!(groups
        .values()
        .all(|elements| elements.len() == 3 * divisions.pow(3)));
    mesh.topology.regions = groups
        .into_iter()
        .map(|(region_id, element_ids)| MeshRegion {
            region_id,
            element_ids,
        })
        .collect();
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();
    mesh
}

fn interface_model(
    left_poisson_ratio: f64,
    right_poisson_ratio: f64,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut left_material = model.materials[0].clone();
    left_material.material_id = "left_material".into();
    let MaterialElasticityModel::IsotropicLinear(left_elasticity) =
        left_material.elasticity.as_mut().unwrap()
    else {
        panic!("the interface reference requires isotropic materials");
    };
    left_elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0);
    left_elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(left_poisson_ratio);
    let mut right_material = left_material.clone();
    right_material.material_id = "right_material".into();
    let MaterialElasticityModel::IsotropicLinear(right_elasticity) =
        right_material.elasticity.as_mut().unwrap()
    else {
        unreachable!();
    };
    right_elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(200.0);
    right_elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(right_poisson_ratio);
    model.materials = vec![left_material, right_material];
    model.material_assignments = vec![
        MaterialAssignment::to_solid_region(
            runmat_meshing_core::fixtures::fixture_region("left_layer"),
            "left_material",
            EvidenceConfidence::Verified,
        ),
        MaterialAssignment::to_solid_region(
            runmat_meshing_core::fixtures::fixture_region("right_layer"),
            "right_material",
            EvidenceConfidence::Verified,
        ),
    ];
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model
}

#[test]
fn connected_tet4_and_tet10_material_interface_preserves_traction_and_energy() {
    const LEFT_STRAIN: f64 = 0.1;
    const AXIAL_STRESS_PA: f64 = 12.0;
    let right_strain = AXIAL_STRESS_PA / constrained_modulus(200.0, 0.3);
    let left_transverse = transverse_stress(100.0, 0.25, LEFT_STRAIN);
    let right_transverse = transverse_stress(200.0, 0.3, right_strain);
    assert!((left_transverse - right_transverse).abs() > 1.0);
    let exact_displacement = |x: f64| {
        if x <= 0.5 {
            LEFT_STRAIN * x
        } else {
            LEFT_STRAIN * 0.5 + right_strain * (x - 0.5)
        }
    };

    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = interface_mesh(2, order);
        let left_region = entity(PersistentEntityKind::Region, "left_layer");

        let regions = mesh
            .topology
            .volume_elements
            .iter()
            .map(|element| (element.element_id, &element.region_id))
            .collect::<BTreeMap<_, _>>();
        let interface_neighbors = mesh
            .topology
            .neighbors
            .iter()
            .filter(|neighbor| {
                neighbor
                    .adjacent_element_id
                    .is_some_and(|adjacent| regions[&neighbor.element_id] != regions[&adjacent])
            })
            .count();
        assert!(interface_neighbors > 0);

        let boundary_nodes = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|face| face.node_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        let free_interface_nodes = mesh
            .topology
            .nodes
            .iter()
            .filter(|node| {
                !boundary_nodes.contains(&node.node_id)
                    && (node.coordinates_m.canonical_value()[0] - 0.5).abs() < 1.0e-12
            })
            .count();
        assert!(
            free_interface_nodes > 0,
            "{order:?} has no free interface DOF"
        );

        let mut model = interface_model(0.25, 0.3);
        model.constraints = boundary_nodes
            .into_iter()
            .map(|node_id| {
                let x = mesh.topology.nodes[node_id as usize - 1]
                    .coordinates_m
                    .canonical_value()[0];
                AnalysisConstraint::Structural(StructuralConstraint {
                    constraint_id: format!("exact_interface_boundary_{node_id}"),
                    frame: CoordinateFrame::Global,
                    kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                        target: RegionTarget::point(format!("node:{node_id}")),
                        translations: [
                            (StructuralTranslationDof::Ux, exact_displacement(x)),
                            (StructuralTranslationDof::Uy, 0.0),
                            (StructuralTranslationDof::Uz, 0.0),
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

        let mut missing_assignment = model.clone();
        missing_assignment
            .material_assignments
            .retain(|assignment| assignment.assigned_material_id != "right_material");
        assert!(
            crate::run_linear_static_with_options(
                &missing_assignment,
                ComputeBackend::Cpu,
                LinearStaticSolveOptions {
                    solver_mesh: Some(mesh.clone()),
                    require_solver_mesh_for_solid: true,
                    ..LinearStaticSolveOptions::default()
                },
            )
            .is_err(),
            "{order:?} must reject an unassigned material region"
        );

        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("the conformal two-material patch must solve");
        let displacement = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        for (node, actual) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
            let expected = [
                exact_displacement(node.coordinates_m.canonical_value()[0]),
                0.0,
                0.0,
            ];
            for (&actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1.0e-8,
                    "{order:?} node {}",
                    node.node_id
                );
            }
        }
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(stress.len(), mesh.topology.volume_elements.len() * 4 * 6);
        for (index, element) in mesh.topology.volume_elements.iter().enumerate() {
            let transverse = if element.region_id == left_region {
                left_transverse
            } else {
                right_transverse
            };
            for point in 0..4 {
                let actual = &stress[(index * 4 + point) * 6..][..6];
                for (&actual, expected) in
                    actual
                        .iter()
                        .zip([AXIAL_STRESS_PA, transverse, transverse, 0.0, 0.0, 0.0])
                {
                    assert!(
                        (actual - expected).abs() < 1.0e-6,
                        "{order:?} element {index}"
                    );
                }
            }
        }
        let energy = result
            .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
            .unwrap()
            .as_host_f64()
            .unwrap()[0];
        let expected_energy = 0.25 * AXIAL_STRESS_PA * (LEFT_STRAIN + right_strain);
        assert!(
            (energy - expected_energy).abs() < 1.0e-7,
            "{order:?} energy {energy}"
        );
        let reactions = result
            .field(FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .unwrap()
            .as_host_f64()
            .unwrap();
        for axis in 0..3 {
            let resultant = reactions
                .chunks_exact(3)
                .map(|node| node[axis])
                .sum::<f64>();
            assert!(
                resultant.abs() < 1.0e-6,
                "{order:?} axis {axis} reaction {resultant}"
            );
        }
    }
}

#[test]
fn connected_tet4_and_tet10_material_interface_transmits_physical_tip_traction() {
    const TRACTION_PA: f64 = 12.0;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = interface_mesh(2, order);
        let mut model = interface_model(0.0, 0.0);
        model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "fixed_root".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
                translations: [
                    StructuralTranslationDof::Ux,
                    StructuralTranslationDof::Uy,
                    StructuralTranslationDof::Uz,
                ]
                .into_iter()
                .map(|dof| TranslationDofValue {
                    dof,
                    value: PrescribedValue::Constant(Displacement::new(0.0)),
                })
                .collect(),
                rotations: Vec::new(),
            }),
        })];
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "positive_x_tip_traction".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Traction([TRACTION_PA, 0.0, 0.0]),
            }],
        );

        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("the two-material physical traction patch must solve");
        let displacement = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        for (node, actual) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
            let x = node.coordinates_m.canonical_value()[0];
            let expected_x = if x <= 0.5 {
                TRACTION_PA * x / 100.0
            } else {
                TRACTION_PA * (0.5 / 100.0 + (x - 0.5) / 200.0)
            };
            for (&actual, expected) in actual.iter().zip([expected_x, 0.0, 0.0]) {
                assert!(
                    (actual - expected).abs() < 1.0e-7,
                    "{order:?} node {}: {actual} vs {expected}",
                    node.node_id
                );
            }
        }
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(stress.len(), mesh.topology.volume_elements.len() * 4 * 6);
        for actual in stress.chunks_exact(6) {
            for (&actual, expected) in actual.iter().zip([TRACTION_PA, 0.0, 0.0, 0.0, 0.0, 0.0]) {
                assert!(
                    (actual - expected).abs() < 1.0e-5,
                    "{order:?} stress {actual}"
                );
            }
        }
        let energy = result
            .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
            .unwrap()
            .as_host_f64()
            .unwrap()[0];
        let expected_energy = 0.5 * TRACTION_PA * TRACTION_PA * (0.5 / 100.0 + 0.5 / 200.0);
        assert!(
            (energy - expected_energy).abs() < 1.0e-6,
            "{order:?} energy {energy}"
        );
        let reaction = result
            .field(FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let total_reaction = std::array::from_fn::<_, 3, _>(|axis| {
            reaction.chunks_exact(3).map(|node| node[axis]).sum::<f64>()
        });
        for (&actual, expected) in total_reaction.iter().zip([-TRACTION_PA, 0.0, 0.0]) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "{order:?} reaction {actual}"
            );
        }

        let mut reordered = model.clone();
        reordered.materials.reverse();
        reordered.material_assignments.reverse();
        let reordered_result = crate::run_linear_static_with_options(
            &reordered,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("material declaration order must not change the loaded interface solution");
        assert_eq!(reordered_result.fields, result.fields);
    }
}
