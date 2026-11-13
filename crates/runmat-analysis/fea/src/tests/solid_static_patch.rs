//! Independently check physical solid patch solves against affine elasticity solutions.

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, MaterialElasticityModel,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{
    fixtures::{
        canonical_tetrahedron_solver_mesh, connected_quadratic_tetrahedron_solver_mesh,
        connected_tetrahedron_solver_mesh,
    },
    ElementOrder,
};

use super::{
    assert_vectors_within_tolerance, baseline_linear_static_options, fixture_model, host_field,
    replace_fixture_excitations, ComputeBackend, FixtureExcitation, FixtureExcitationKind,
    FixtureId, LinearStaticSolveOptions, ParityTolerance, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
};

fn patch_model() -> runmat_analysis_core::AnalysisModel {
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
    model
}

#[test]
fn connected_tet4_static_dirichlet_patch_recovers_affine_interior_and_boundary_tractions() {
    let mut model = patch_model();
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

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(connected_tetrahedron_solver_mesh()),
            solver_mesh_artifact_path: Some("connected-static-patch.rmmesh".into()),
            ..baseline_linear_static_options()
        },
    )
    .expect("connected affine Dirichlet patch should solve");

    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_eq!(displacement.len(), 15);
    for (actual, expected) in displacement[12..15].iter().zip([0.025, 0.05, 0.075]) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "{actual} != {expected}"
        );
    }

    // E=100 Pa, nu=0.25 gives lambda=mu=40 Pa. The affine strain is
    // diag(0.1,0.2,0.3), hence sigma=diag(32,40,48) Pa and w=12.8 J/m^3.
    let strain = host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN);
    let stress = host_field(&result, FEA_FIELD_STRUCTURAL_STRESS);
    let energy_density = host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY);
    assert_eq!(strain.len(), 24);
    assert_eq!(stress.len(), 24);
    assert_eq!(energy_density.len(), 4);
    for element in 0..4 {
        for (actual, expected) in strain[element * 6..][..6]
            .iter()
            .zip([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "strain {actual} != {expected}"
            );
        }
        for (actual, expected) in stress[element * 6..][..6]
            .iter()
            .zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0])
        {
            assert!(
                (actual - expected).abs() < 1.0e-8,
                "stress {actual} != {expected}"
            );
        }
        assert!((energy_density[element] - 12.8).abs() < 1.0e-9);
    }
    assert!(
        (host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] - 12.8 / 6.0).abs()
            < 1.0e-9
    );

    // The analytical exterior nodal traction is integral(N_i sigma*n) dA.
    // The published reaction field uses the same sign as K*u, and the
    // unconstrained centroid has none.
    let reaction = host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    for (actual, traction) in reaction.chunks_exact(3).zip([
        [-16.0 / 3.0, -20.0 / 3.0, -8.0],
        [16.0 / 3.0, 0.0, 0.0],
        [0.0, 20.0 / 3.0, 0.0],
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 0.0],
    ]) {
        for (&actual, traction) in actual.iter().zip(traction) {
            assert!(
                (actual - traction).abs() < 1.0e-8,
                "reaction {actual} vs traction {traction}"
            );
        }
    }
}

pub(super) fn traction_patch_model() -> runmat_analysis_core::AnalysisModel {
    let mut model = patch_model();
    // Six zero-valued DOFs remove rigid translations and rotations without
    // imposing a strain incompatible with u=(0.1x,0.2y,0.3z).
    model.constraints = [
        (
            1,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ][..],
        ),
        (
            2,
            &[StructuralTranslationDof::Uy, StructuralTranslationDof::Uz][..],
        ),
        (3, &[StructuralTranslationDof::Uz][..]),
    ]
    .into_iter()
    .map(|(node_id, dofs)| {
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: format!("rigid_mode_{node_id}"),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point(format!("node:{node_id}")),
                translations: dofs
                    .iter()
                    .copied()
                    .map(|dof| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    })
                    .collect(),
                rotations: Vec::new(),
            }),
        })
    })
    .collect();

    // Outward tractions t=sigma*n for sigma=diag(32,40,48) Pa.
    // The inclined face has unit normal (1,1,1)/sqrt(3).
    let root_three = 3.0_f64.sqrt();
    replace_fixture_excitations(
        &mut model,
        vec![
            FixtureExcitation {
                id: "z_negative".into(),
                region_id: "face:0".into(),
                kind: FixtureExcitationKind::Traction([0.0, 0.0, -48.0]),
            },
            FixtureExcitation {
                id: "y_negative".into(),
                region_id: "face:1".into(),
                kind: FixtureExcitationKind::Traction([0.0, -40.0, 0.0]),
            },
            FixtureExcitation {
                id: "x_negative".into(),
                region_id: "face:2".into(),
                kind: FixtureExcitationKind::Traction([-32.0, 0.0, 0.0]),
            },
            FixtureExcitation {
                id: "inclined".into(),
                region_id: "face:3".into(),
                kind: FixtureExcitationKind::Traction([
                    32.0 / root_three,
                    40.0 / root_three,
                    48.0 / root_three,
                ]),
            },
        ],
    );
    model
}

#[test]
fn connected_tet4_traction_patch_recovers_affine_solution_and_nodal_resultants() {
    let model = traction_patch_model();
    let mesh = connected_tetrahedron_solver_mesh();
    let mut unrestrained = model.clone();
    unrestrained.constraints.clear();
    let summary = crate::assembly::assemble_structural_system(
        &unrestrained,
        &super::static_excitations(&unrestrained),
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("traction patch should assemble");
    assert_eq!(summary.operator.load().len(), 15);
    for (actual, expected) in summary.operator.load().chunks_exact(3).zip([
        [-16.0 / 3.0, -20.0 / 3.0, -8.0],
        [16.0 / 3.0, 0.0, 0.0],
        [0.0, 20.0 / 3.0, 0.0],
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 0.0],
    ]) {
        for (&actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "load {actual} vs {expected}"
            );
        }
    }

    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh),
        solver_mesh_artifact_path: Some("connected-traction-patch.rmmesh".into()),
        ..baseline_linear_static_options()
    };
    let result =
        crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options.clone())
            .expect("connected traction patch should solve");
    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_eq!(displacement.len(), 15);
    for (actual, expected) in displacement.chunks_exact(3).zip([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.3],
        [0.025, 0.05, 0.075],
    ]) {
        for (&actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "u {actual} vs {expected}"
            );
        }
    }
    let stress_field = host_field(&result, FEA_FIELD_STRUCTURAL_STRESS);
    assert_eq!(stress_field.len(), 24);
    for stress in stress_field.chunks_exact(6) {
        for (&actual, expected) in stress.iter().zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0]) {
            assert!(
                (actual - expected).abs() < 1.0e-7,
                "stress {actual} vs {expected}"
            );
        }
    }
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE)
        .iter()
        .all(|value| value.abs() < 1.0e-8));
    assert!(
        (host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] - 12.8 / 6.0).abs()
            < 1.0e-8
    );

    let parallel = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            local_execution: crate::planning::resource::LocalExecutionPlan::fixed(
                crate::planning::resource::LocalThreadCount::new(2).unwrap(),
            )
            .unwrap(),
            ..options
        },
    )
    .expect("parallel traction patch should solve");
    assert_eq!(result.fields.len(), parallel.fields.len());
    for (serial, parallel) in result.fields.iter().zip(&parallel.fields) {
        assert_eq!(serial.field_id, parallel.field_id);
        assert_eq!(serial.shape, parallel.shape);
        assert_vectors_within_tolerance(
            serial.as_host_f64().unwrap(),
            parallel.as_host_f64().unwrap(),
            ParityTolerance {
                abs: 1.0e-9,
                rel: 1.0e-10,
            },
        );
    }
}

#[test]
fn tet10_traction_patch_recovers_quadratic_face_loads_and_affine_stress() {
    let model = traction_patch_model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut unrestrained = model.clone();
    unrestrained.constraints.clear();
    let summary = crate::assembly::assemble_structural_system(
        &unrestrained,
        &super::static_excitations(&unrestrained),
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("Tet10 traction patch should assemble");

    // A straight-sided quadratic triangle integrates its three corner basis
    // functions to zero and each midside basis to area/3. Check the full
    // assembled vector without reusing the production face quadrature.
    let root_three = 3.0_f64.sqrt();
    let tractions = [
        [0.0, 0.0, -48.0],
        [0.0, -40.0, 0.0],
        [-32.0, 0.0, 0.0],
        [32.0 / root_three, 40.0 / root_three, 48.0 / root_three],
    ];
    let areas = [0.5, 0.5, 0.5, root_three / 2.0];
    let mut expected_load = vec![0.0; 30];
    for ((face, traction), area) in mesh
        .topology
        .boundary_faces
        .iter()
        .zip(tractions)
        .zip(areas)
    {
        for &node_id in &face.node_ids[3..] {
            for axis in 0..3 {
                expected_load[(node_id as usize - 1) * 3 + axis] += area * traction[axis] / 3.0;
            }
        }
    }
    assert_vectors_within_tolerance(
        summary.operator.load(),
        &expected_load,
        ParityTolerance {
            abs: 1.0e-10,
            rel: 1.0e-10,
        },
    );
    let mut resultant = [0.0; 3];
    let mut moment = [0.0; 3];
    for (node, force) in mesh
        .topology
        .nodes
        .iter()
        .zip(expected_load.chunks_exact(3))
    {
        let [x, y, z] = node.coordinates_m.canonical_value();
        resultant
            .iter_mut()
            .zip(force)
            .for_each(|(sum, value)| *sum += value);
        moment[0] += y * force[2] - z * force[1];
        moment[1] += z * force[0] - x * force[2];
        moment[2] += x * force[1] - y * force[0];
    }
    assert!(resultant.into_iter().all(|value| value.abs() < 1.0e-10));
    assert!(moment.into_iter().all(|value| value.abs() < 1.0e-10));

    let solve_summary = crate::assembly::assemble_structural_system(
        &model,
        &super::static_excitations(&model),
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("Tet10 constrained patch should assemble");
    let exact = mesh
        .topology
        .nodes
        .iter()
        .flat_map(|node| {
            [
                0.1 * node.coordinates_m.canonical_value()[0],
                0.2 * node.coordinates_m.canonical_value()[1],
                0.3 * node.coordinates_m.canonical_value()[2],
            ]
        })
        .collect::<Vec<_>>();
    let internal = crate::algebra::apply_k(&solve_summary.operator, &exact);
    assert_vectors_within_tolerance(
        &internal,
        &solve_summary.physical_load,
        ParityTolerance {
            abs: 1.0e-10,
            rel: 1.0e-10,
        },
    );

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            solver_mesh_artifact_path: Some("tet10-traction-patch.rmmesh".into()),
            ..baseline_linear_static_options()
        },
    )
    .expect("Tet10 traction patch should solve");
    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_eq!(displacement.len(), 30);
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
    let stress = host_field(&result, FEA_FIELD_STRUCTURAL_STRESS);
    assert_eq!(stress.len(), 6);
    for (&actual, expected) in stress.iter().zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0]) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "stress {actual} vs {expected}"
        );
    }
    let integration_stress = host_field(
        &result,
        crate::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
    );
    assert_eq!(integration_stress.len(), 24);
    for values in integration_stress.chunks_exact(6) {
        for (&actual, expected) in values.iter().zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0]) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "integration stress {actual} vs {expected}"
            );
        }
    }
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE)
        .iter()
        .all(|value| value.abs() < 1.0e-7));
    assert!(
        (host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] - 12.8 / 6.0).abs()
            < 1.0e-7
    );
}

#[test]
fn connected_tet10_traction_patch_recovers_affine_solution_and_matches_parallel() {
    let model = traction_patch_model();
    let mesh = connected_quadratic_tetrahedron_solver_mesh();
    let exact = mesh
        .topology
        .nodes
        .iter()
        .flat_map(|node| {
            [
                0.1 * node.coordinates_m.canonical_value()[0],
                0.2 * node.coordinates_m.canonical_value()[1],
                0.3 * node.coordinates_m.canonical_value()[2],
            ]
        })
        .collect::<Vec<_>>();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &super::static_excitations(&model),
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("connected Tet10 patch should assemble");
    let internal = crate::algebra::apply_k(&summary.operator, &exact);
    assert_vectors_within_tolerance(
        &internal,
        &summary.physical_load,
        ParityTolerance {
            abs: 1.0e-9,
            rel: 1.0e-10,
        },
    );
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        solver_mesh_artifact_path: Some("connected-tet10-traction-patch.rmmesh".into()),
        ..baseline_linear_static_options()
    };
    let serial =
        crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options.clone())
            .expect("connected Tet10 patch should solve");
    assert_vectors_within_tolerance(
        host_field(&serial, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        &exact,
        ParityTolerance {
            abs: 1.0e-8,
            rel: 1.0e-8,
        },
    );
    let stress = host_field(&serial, FEA_FIELD_STRUCTURAL_STRESS);
    assert_eq!(stress.len(), 4 * 6);
    for values in stress.chunks_exact(6) {
        assert_vectors_within_tolerance(
            values,
            &[32.0, 40.0, 48.0, 0.0, 0.0, 0.0],
            ParityTolerance {
                abs: 1.0e-6,
                rel: 1.0e-8,
            },
        );
    }
    assert!(
        (host_field(&serial, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] - 12.8 / 6.0).abs()
            < 1.0e-7
    );
    let parallel = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            local_execution: crate::planning::resource::LocalExecutionPlan::fixed(
                crate::planning::resource::LocalThreadCount::new(2).unwrap(),
            )
            .unwrap(),
            ..options
        },
    )
    .expect("parallel connected Tet10 patch should solve");
    assert_eq!(serial.fields, parallel.fields);
    assert_eq!(serial.observations, parallel.observations);
}
