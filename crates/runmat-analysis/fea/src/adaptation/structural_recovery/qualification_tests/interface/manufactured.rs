//! Independent quadratic axial solution with a body force across two bonded materials.

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, PrescribedStructuralDofs, PrescribedValue,
    RegionTarget, StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof,
    TranslationDofValue,
};
use runmat_meshing_core::ElementOrder;

use super::{interface_mesh, interface_model, replace_fixture_excitations};
use crate::{
    contracts::{
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    },
    fixtures::{FixtureExcitation, FixtureExcitationKind},
    ComputeBackend, LinearStaticSolveOptions,
};

const ROOT_STRESS_PA: f64 = 12.0;
const STRESS_SLOPE_PA_PER_M: f64 = 6.0;
// (1/2) ∫_0^0.5 (12+6x)^2/100 dx + (1/2) ∫_0.5^1 (12+6x)^2/200 dx.
const EXACT_ENERGY_J: f64 = 0.79875;

fn exact_displacement(x: f64) -> f64 {
    let primitive = |position: f64| {
        ROOT_STRESS_PA * position + 0.5 * STRESS_SLOPE_PA_PER_M * position * position
    };
    if x <= 0.5 {
        primitive(x) / 100.0
    } else {
        primitive(0.5) / 100.0 + (primitive(x) - primitive(0.5)) / 200.0
    }
}

fn solve_manufactured_interface(divisions: usize, order: ElementOrder) -> (f64, f64) {
    let mesh = interface_mesh(divisions, order);
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
        vec![
            FixtureExcitation {
                id: "left_body_force".into(),
                region_id: "left_layer".into(),
                kind: FixtureExcitationKind::BodyForce([-STRESS_SLOPE_PA_PER_M, 0.0, 0.0]),
            },
            FixtureExcitation {
                id: "right_body_force".into(),
                region_id: "right_layer".into(),
                kind: FixtureExcitationKind::BodyForce([-STRESS_SLOPE_PA_PER_M, 0.0, 0.0]),
            },
            FixtureExcitation {
                id: "tip_traction".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Traction([
                    ROOT_STRESS_PA + STRESS_SLOPE_PA_PER_M,
                    0.0,
                    0.0,
                ]),
            },
        ],
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
    .expect("the manufactured two-material body-force case must solve");
    let displacement = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let nodal_error = mesh
        .topology
        .nodes
        .iter()
        .zip(displacement.chunks_exact(3))
        .map(|(node, actual)| {
            let exact = exact_displacement(node.coordinates_m.canonical_value()[0]);
            (actual[0] - exact).powi(2) + actual[1].powi(2) + actual[2].powi(2)
        })
        .sum::<f64>();
    let nodal_rms = (nodal_error / mesh.topology.nodes.len() as f64).sqrt();

    let stress = result
        .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(stress.len(), mesh.topology.volume_elements.len() * 4 * 6);
    let mut stress_error = 0.0;
    let a = 0.585_410_196_624_968_5;
    let b = 0.138_196_601_125_010_5;
    for (element, points) in mesh
        .topology
        .volume_elements
        .iter()
        .zip(stress.chunks_exact(4 * 6))
    {
        let corners = std::array::from_fn::<_, 4, _>(|index| {
            mesh.topology.nodes[element.node_ids[index] as usize - 1]
                .coordinates_m
                .canonical_value()[0]
        });
        for point in 0..4 {
            let quadrature_x = corners
                .iter()
                .enumerate()
                .map(|(index, x)| if index == point { a * x } else { b * x })
                .sum::<f64>();
            let actual = &points[point * 6..][..6];
            stress_error +=
                (actual[0] - (ROOT_STRESS_PA + STRESS_SLOPE_PA_PER_M * quadrature_x)).powi(2);
            stress_error += actual[1..].iter().map(|value| value.powi(2)).sum::<f64>();
        }
    }
    let stress_rms = (stress_error / (mesh.topology.volume_elements.len() * 4) as f64).sqrt();
    let energy = result
        .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    let energy_tolerance = if order == ElementOrder::Tet10 {
        1.0e-6
    } else {
        2.0e-2
    };
    assert!(
        (energy - EXACT_ENERGY_J).abs() < energy_tolerance,
        "{order:?} divisions {divisions}: energy {energy}"
    );
    let reaction = result
        .field(FEA_FIELD_STRUCTURAL_REACTION_FORCE)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let total_reaction = std::array::from_fn::<_, 3, _>(|axis| {
        reaction.chunks_exact(3).map(|node| node[axis]).sum::<f64>()
    });
    for (&actual, expected) in total_reaction.iter().zip([-ROOT_STRESS_PA, 0.0, 0.0]) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "{order:?} divisions {divisions}: reaction {actual}"
        );
    }
    (nodal_rms, stress_rms)
}

#[test]
fn manufactured_body_force_interface_has_tet4_stress_convergence_and_tet10_exactness() {
    let tet4_coarse = solve_manufactured_interface(2, ElementOrder::Tet4);
    let tet4_fine = solve_manufactured_interface(4, ElementOrder::Tet4);
    assert!(
        tet4_fine.1 < 0.75 * tet4_coarse.1,
        "Tet4 stress errors: coarse={tet4_coarse:?}, fine={tet4_fine:?}"
    );
    let tet10 = solve_manufactured_interface(2, ElementOrder::Tet10);
    assert!(tet10.0 < 1.0e-7, "Tet10 nodal error: {tet10:?}");
    assert!(tet10.1 < 1.0e-5, "Tet10 stress error: {tet10:?}");
}
