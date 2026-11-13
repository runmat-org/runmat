//! Independent structural recovery references.
use super::*;

pub(super) fn manufactured_quadratic_solid_solution_converges_for_tet4_and_is_exact_for_tet10() {
    // u=(0.1*x^2,0,0) in a unit cube with E=100 Pa and nu=0.25 has
    // lambda=mu=40 Pa, sigma_xx=24*x Pa, and -div(sigma)=(-24,0,0) N/m^3.
    // Exact displacement is imposed only on exterior mesh nodes; interior
    // values and all recovered stresses are solved, not prescribed.
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
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "manufactured_body_force".into(),
            region_id: "region".into(),
            kind: FixtureExcitationKind::BodyForce([-24.0, 0.0, 0.0]),
        }],
    );

    let mut tet4_stress_errors = Vec::new();
    for (order, divisions) in [
        (ElementOrder::Tet4, 1),
        (ElementOrder::Tet4, 2),
        (ElementOrder::Tet4, 3),
        (ElementOrder::Tet10, 1),
        (ElementOrder::Tet10, 2),
    ] {
        let mesh = structured_cantilever(divisions, order);
        let mut unrestrained = model.clone();
        unrestrained.constraints.clear();
        let excitations = crate::excitation::realize_step_excitations(
            &unrestrained,
            &unrestrained.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .expect("manufactured body force should realize");
        let summary = crate::assembly::assemble_structural_system(
            &unrestrained,
            &excitations,
            Some(mesh.clone()),
            crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("manufactured body force should assemble");
        assert_eq!(summary.physical_load.len(), mesh.topology.nodes.len() * 3);
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        for (node, nodal) in mesh
            .topology
            .nodes
            .iter()
            .zip(summary.physical_load.chunks_exact(3))
        {
            let [x, y, z] = node.coordinates_m.canonical_value();
            for axis in 0..3 {
                force[axis] += nodal[axis];
            }
            moment[0] += y * nodal[2] - z * nodal[1];
            moment[1] += z * nodal[0] - x * nodal[2];
            moment[2] += x * nodal[1] - y * nodal[0];
        }
        for (actual, expected) in force.into_iter().zip([-24.0, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        for (actual, expected) in moment.into_iter().zip([0.0, -12.0, 12.0]) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        let boundary_nodes = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|face| face.node_ids.iter().copied())
            .collect::<std::collections::BTreeSet<_>>();
        model.constraints = boundary_nodes
            .into_iter()
            .map(|node_id| {
                let x = mesh.topology.nodes[node_id as usize - 1]
                    .coordinates_m
                    .canonical_value()[0];
                AnalysisConstraint::Structural(StructuralConstraint {
                    constraint_id: format!("quadratic_boundary_{node_id}"),
                    frame: CoordinateFrame::Global,
                    kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                        target: RegionTarget::point(format!("node:{node_id}")),
                        translations: [
                            (StructuralTranslationDof::Ux, 0.1 * x * x),
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
        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("manufactured quadratic body-force problem should solve");
        let displacement = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(displacement.len(), mesh.topology.nodes.len() * 3);
        let maximum_displacement_error = displacement
            .chunks_exact(3)
            .zip(&mesh.topology.nodes)
            .map(|(value, node)| {
                (value[0] - 0.1 * node.coordinates_m.canonical_value()[0].powi(2))
                    .abs()
                    .max(value[1].abs())
                    .max(value[2].abs())
            })
            .fold(0.0_f64, f64::max);
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_STRESS)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let stress_error_squared = stress
            .chunks_exact(6)
            .zip(&mesh.topology.volume_elements)
            .map(|(value, element)| {
                let x = element.node_ids[..4]
                    .iter()
                    .map(|id| {
                        mesh.topology.nodes[*id as usize - 1]
                            .coordinates_m
                            .canonical_value()[0]
                    })
                    .sum::<f64>()
                    / 4.0;
                value
                    .iter()
                    .zip([24.0 * x, 8.0 * x, 8.0 * x, 0.0, 0.0, 0.0])
                    .map(|(actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let rms_stress_error =
            (stress_error_squared / mesh.topology.volume_elements.len() as f64).sqrt();
        if order == ElementOrder::Tet4 {
            tet4_stress_errors.push(rms_stress_error);
            assert!(maximum_displacement_error < 1.0e-8);
        } else {
            assert!(
                maximum_displacement_error < 1.0e-8,
                "{order:?} {divisions}: {maximum_displacement_error}"
            );
            assert!(
                rms_stress_error < 1.0e-6,
                "{order:?} {divisions}: {rms_stress_error}"
            );
        }
    }
    assert!(
        tet4_stress_errors
            .windows(2)
            .all(|pair| pair[1] < 0.8 * pair[0]),
        "Tet4 manufactured stress errors: {tet4_stress_errors:?}"
    );
}
