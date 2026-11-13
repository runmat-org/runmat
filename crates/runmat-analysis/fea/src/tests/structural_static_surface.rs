use super::*;

use crate::excitation::SpatialProjectionBalance;

fn surface_case(
    order: ElementOrder,
    kind: FixtureExcitationKind,
) -> (FeaRunResult, SolverMeshArtifact) {
    let model = surface_model("face_load", kind);
    let mesh = single_tetrahedron_solver_mesh_with_order_for_model(&model, order);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..baseline_linear_static_options()
        },
    )
    .expect("a supported single-tetrahedron surface-load solve must complete");
    (result, mesh)
}

fn surface_model(id: &str, kind: FixtureExcitationKind) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: id.into(),
            region_id: "tip".into(),
            kind,
        }],
    );
    model
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-8,
        "actual={actual} expected={expected}"
    );
}

fn close_scaled(actual: f64, expected: f64, scale: f64) {
    assert!(
        (actual - expected).abs() <= scale * 1.0e-8 + 1.0e-20,
        "actual={actual} expected={expected} scale={scale}"
    );
}

#[test]
fn solved_tet4_and_tet10_surface_loads_preserve_global_balance_and_work_scaling() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for (load, doubled_load) in [
            (
                FixtureExcitationKind::Force([0.0, -12.0, 0.0]),
                FixtureExcitationKind::Force([0.0, -24.0, 0.0]),
            ),
            (
                FixtureExcitationKind::Pressure(12.0),
                FixtureExcitationKind::Pressure(24.0),
            ),
            (
                FixtureExcitationKind::Traction([5.0, -7.0, 11.0]),
                FixtureExcitationKind::Traction([10.0, -14.0, 22.0]),
            ),
        ] {
            let (base, mesh) = surface_case(order, load);
            let (doubled, _) = surface_case(order, doubled_load);
            let face = &mesh.topology.boundary_faces[1];
            let points = face
                .node_ids
                .iter()
                .take(3)
                .map(|node_id| {
                    mesh.topology
                        .nodes
                        .iter()
                        .find(|node| node.node_id == *node_id)
                        .unwrap()
                        .coordinates_m
                        .canonical_value()
                })
                .collect::<Vec<_>>();
            let u = std::array::from_fn::<_, 3, _>(|axis| points[1][axis] - points[0][axis]);
            let v = std::array::from_fn::<_, 3, _>(|axis| points[2][axis] - points[0][axis]);
            let oriented_area = [
                0.5 * (u[1] * v[2] - u[2] * v[1]),
                0.5 * (u[2] * v[0] - u[0] * v[2]),
                0.5 * (u[0] * v[1] - u[1] * v[0]),
            ];
            let centroid = std::array::from_fn::<_, 3, _>(|axis| {
                (points[0][axis] + points[1][axis] + points[2][axis]) / 3.0
            });
            let expected_force = match load {
                FixtureExcitationKind::Force(components) => components,
                FixtureExcitationKind::Pressure(pascals) => {
                    oriented_area.map(|area| pascals * area)
                }
                FixtureExcitationKind::Traction(components) => {
                    let area = oriented_area
                        .into_iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                        .sqrt();
                    components.map(|component| component * area)
                }
                _ => unreachable!("the surface-load test uses pressure and traction"),
            };
            let expected_moment = [
                centroid[1] * expected_force[2] - centroid[2] * expected_force[1],
                centroid[2] * expected_force[0] - centroid[0] * expected_force[2],
                centroid[0] * expected_force[1] - centroid[1] * expected_force[0],
            ];

            let evidence = &base.spatial_projection_evidence[0];
            evidence.validate().unwrap();
            assert!(evidence.accepted);
            let SpatialProjectionBalance::VectorResultant {
                projected_resultant,
                projected_moment,
                ..
            } = evidence.balance
            else {
                panic!("surface load must have vector-resultant evidence")
            };
            for (actual, expected) in projected_resultant.real.iter().zip(expected_force) {
                close(*actual, expected);
            }
            let projected_about_origin = [
                projected_moment.real[0]
                    + evidence.reference_point_m[1] * projected_resultant.real[2]
                    - evidence.reference_point_m[2] * projected_resultant.real[1],
                projected_moment.real[1]
                    + evidence.reference_point_m[2] * projected_resultant.real[0]
                    - evidence.reference_point_m[0] * projected_resultant.real[2],
                projected_moment.real[2]
                    + evidence.reference_point_m[0] * projected_resultant.real[1]
                    - evidence.reference_point_m[1] * projected_resultant.real[0],
            ];
            for (actual, expected) in projected_about_origin.into_iter().zip(expected_moment) {
                close(actual, expected);
            }

            let reaction = base
                .field(FEA_FIELD_STRUCTURAL_REACTION_FORCE)
                .unwrap()
                .as_host_f64()
                .unwrap();
            let mut reaction_force = [0.0; 3];
            let mut reaction_moment = [0.0; 3];
            for (node, components) in mesh.topology.nodes.iter().zip(reaction.chunks_exact(3)) {
                for axis in 0..3 {
                    reaction_force[axis] += components[axis];
                }
                let [x, y, z] = node.coordinates_m.canonical_value();
                reaction_moment[0] += y * components[2] - z * components[1];
                reaction_moment[1] += z * components[0] - x * components[2];
                reaction_moment[2] += x * components[1] - y * components[0];
            }
            for (actual, expected) in reaction_force.into_iter().zip(expected_force) {
                close(actual, -expected);
            }
            for (actual, expected) in reaction_moment.into_iter().zip(expected_moment) {
                close(actual, -expected);
            }

            let displacement = base
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap();
            let doubled_displacement = doubled
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap();
            assert!(displacement.iter().any(|value| value.abs() > 0.0));
            let displacement_scale = displacement
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max);
            for (base, doubled) in displacement.iter().zip(doubled_displacement) {
                close_scaled(*doubled, 2.0 * base, displacement_scale);
            }
            let energy = base
                .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
                .unwrap()
                .as_host_f64()
                .unwrap()[0];
            let doubled_energy = doubled
                .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
                .unwrap()
                .as_host_f64()
                .unwrap()[0];
            assert!(energy > 0.0);
            close_scaled(doubled_energy, 4.0 * energy, energy);
            if order == ElementOrder::Tet4
                && matches!(
                    load,
                    FixtureExcitationKind::Force(_) | FixtureExcitationKind::Pressure(_)
                )
            {
                // The base triangle fixes nodes 1-3. The opposite node has one
                // shear DOF under the y-directed face force or pressure: K_yy=mu V=mu/6,
                // and its consistent nodal force is one third of the face resultant.
                let shear_modulus = 200.0e9 / (2.0 * (1.0 + 0.3));
                let expected_tip_y = 2.0 * expected_force[1] / shear_modulus;
                let expected_energy = 0.5 * (expected_force[1] / 3.0) * expected_tip_y;
                close_scaled(displacement[10], expected_tip_y, expected_tip_y.abs());
                close_scaled(energy, expected_energy, expected_energy);
            }
            assert!(base.structural_static_balance.as_ref().unwrap().accepted);
            assert!(doubled.structural_static_balance.as_ref().unwrap().accepted);
        }
    }
}

#[test]
fn tet4_and_tet10_surface_loads_obey_reciprocity_and_energy_polarization() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let pressure = FixtureExcitationKind::Pressure(12.0);
        let traction = FixtureExcitationKind::Traction([5.0, -7.0, 11.0]);
        let (pressure_result, mesh) = surface_case(order, pressure);
        let (traction_result, _) = surface_case(order, traction);
        let pressure_model = surface_model("pressure", pressure);
        let traction_model = surface_model("traction", traction);
        let pressure_load = crate::assembly::assemble_structural_system(
            &pressure_model,
            &super::static_excitations(&pressure_model),
            Some(mesh.clone()),
            crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap()
        .physical_load;
        let traction_load = crate::assembly::assemble_structural_system(
            &traction_model,
            &super::static_excitations(&traction_model),
            Some(mesh.clone()),
            crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap()
        .physical_load;
        let pressure_u = pressure_result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let traction_u = traction_result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let pressure_on_traction = pressure_load
            .iter()
            .zip(traction_u)
            .map(|(force, displacement)| force * displacement)
            .sum::<f64>();
        let traction_on_pressure = traction_load
            .iter()
            .zip(pressure_u)
            .map(|(force, displacement)| force * displacement)
            .sum::<f64>();
        assert!(pressure_on_traction.abs() > 0.0);
        let work_scale = pressure_on_traction.abs().max(traction_on_pressure.abs());
        close_scaled(pressure_on_traction, traction_on_pressure, work_scale);

        let mut combined_model = fixture_model(FixtureId::CantileverLinearStatic);
        replace_fixture_excitations(
            &mut combined_model,
            vec![
                FixtureExcitation {
                    id: "pressure".into(),
                    region_id: "tip".into(),
                    kind: pressure,
                },
                FixtureExcitation {
                    id: "traction".into(),
                    region_id: "tip".into(),
                    kind: traction,
                },
            ],
        );
        let combined = crate::run_linear_static_with_options(
            &combined_model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh),
                ..baseline_linear_static_options()
            },
        )
        .unwrap();
        let combined_u = combined
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let displacement_scale = combined_u.iter().copied().map(f64::abs).fold(0.0, f64::max);
        for ((combined, pressure), traction) in combined_u.iter().zip(pressure_u).zip(traction_u) {
            close_scaled(*combined, pressure + traction, displacement_scale);
        }
        let energy = |result: &FeaRunResult| {
            result
                .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
                .unwrap()
                .as_host_f64()
                .unwrap()[0]
        };
        let polarized_energy =
            energy(&combined) - energy(&pressure_result) - energy(&traction_result);
        close_scaled(polarized_energy, pressure_on_traction, work_scale);
        assert!(
            combined
                .structural_static_balance
                .as_ref()
                .unwrap()
                .accepted
        );
    }
}
