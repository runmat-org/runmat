use super::tests::solve_operator_system;
use crate::topology::test_meshes::{entity, slab_mesh};
use runmat_analysis_core::{
    AnalysisConstraint, PrescribedValue, PropertyRegionTarget, RegionTarget, Temperature,
    ThermalConstraint, ThermalConstraintKind,
};
use runmat_meshing_core::{MeshRegion, PersistentEntityKind};

use super::tests::assemble;
use crate::{
    excitation::{ExcitationSamplePoint, RealizedExcitations},
    fixtures::{fixture_model, FixtureId},
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        preconditioner::SpdPreconditionerKind,
    },
};

// witness: FEA-THERMAL-DOMAIN-001/assembly @ thermal_steady/tet4/numerical_kernel;
//   check: series-slab; boundary: Aligned Tet4 two-material series resistance, temperature and flux plus property-order
//   invariance; no arbitrary nonuniform interface claim.
#[test]
fn two_material_slab_matches_series_resistance_and_authored_order_is_invariant() {
    // Independent steady 1-D reference: two half-meter layers in series, k=1 and k=4 W/(m K).
    // T(0)=300 K, T(1)=400 K => interface T=380 K and heat flux magnitude=160 W/m².
    let mut mesh = slab_mesh(2);
    let left_region = entity(PersistentEntityKind::Region, "left_material");
    let right_region = entity(PersistentEntityKind::Region, "right_material");
    let mut left_elements = Vec::new();
    let mut right_elements = Vec::new();
    let topo = &mut *mesh.topology;
    for element in &mut topo.volume_elements {
        let left = element.node_ids.iter().all(|node_id| {
            topo.nodes[*node_id as usize - 1]
                .coordinates_m
                .canonical_value()[0]
                <= 0.5
        });
        if left {
            element.region_id = left_region.clone();
            left_elements.push(element.element_id);
        } else {
            element.region_id = right_region.clone();
            right_elements.push(element.element_id);
        }
    }
    assert_eq!(left_elements.len(), right_elements.len());
    mesh.topology.regions = vec![
        MeshRegion {
            region_id: left_region,
            element_ids: left_elements,
        },
        MeshRegion {
            region_id: right_region,
            element_ids: right_elements,
        },
    ];
    mesh.seal_canonical_digest().expect("two-region slab mesh");

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(1.0);
    model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    let mut right_material = model.materials[0].clone();
    right_material.material_id = "right_conductor".into();
    right_material
        .thermal
        .as_mut()
        .unwrap()
        .conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(4.0);
    model.materials.push(right_material);
    model.material_assignments[0].target = PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_region("left_material"),
    );
    let mut right_assignment = model.material_assignments[0].clone();
    right_assignment.target = PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_region("right_material"),
    );
    right_assignment.assigned_material_id = "right_conductor".into();
    right_assignment.expected_material_id = "right_conductor".into();
    model.material_assignments.push(right_assignment);
    for (id, region, temperature_k) in [
        ("left_temperature", "slab_left", 300.0),
        ("right_temperature", "slab_right", 400.0),
    ] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: id.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(region)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(temperature_k)),
                },
            }));
    }

    let solve = |model: &runmat_analysis_core::AnalysisModel| {
        let assembly = assemble(
            model,
            &mesh,
            crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
                .unwrap()
                .support(),
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .expect("assemble two-material thermal system");
        let reference_temperature_k = 300.0;
        let operator = assembly
            .steady_increment_operator(
                model,
                ExcitationSamplePoint::Static,
                &vec![reference_temperature_k; assembly.projection.nodes.len()],
            )
            .expect("steady increment operator");
        let mut result = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
        assert!(result.converged, "two-material solve did not converge");
        for value in &mut result.solution {
            *value += reference_temperature_k;
        }
        (assembly, result.solution)
    };
    let (assembly, solution) = solve(&model);
    for (node, temperature) in assembly.projection.nodes.iter().zip(&solution) {
        let x = node.coordinates_m.canonical_value()[0];
        let expected = if x <= 0.5 {
            300.0 + 160.0 * x
        } else {
            380.0 + 40.0 * (x - 0.5)
        };
        assert!(
            (temperature - expected).abs() < 1.0e-7,
            "node {} at x={x}: {temperature} K versus {expected} K",
            node.node_id
        );
    }
    assert!(assembly
        .element_conductivity_w_per_mk
        .canonical_values()
        .contains(&1.0));
    assert!(assembly
        .element_conductivity_w_per_mk
        .canonical_values()
        .contains(&4.0));
    for flux in assembly
        .recover_element_heat_flux(&solution)
        .expect("recover steady heat flux")
    {
        assert!(
            (flux[0] + 160.0).abs() < 1.0e-6,
            "incorrect series heat flux: {flux:?}"
        );
        assert!(flux[1].abs() < 1.0e-6 && flux[2].abs() < 1.0e-6);
    }

    model.materials.reverse();
    model.material_assignments.reverse();
    let (reordered, reordered_solution) = solve(&model);
    assert_eq!(assembly.property_identity, reordered.property_identity);
    assert_eq!(
        assembly.element_conductivity_w_per_mk.canonical_values(),
        reordered.element_conductivity_w_per_mk.canonical_values()
    );
    for (original, after_reorder) in solution.iter().zip(reordered_solution) {
        assert!((original - after_reorder).abs() < 1.0e-10);
    }
}

#[test]
fn transient_slab_has_second_order_spatial_convergence() {
    let final_time_s = 0.1;
    let time_step_s = final_time_s / 256.0;
    let mut errors = Vec::new();
    let mut midspan_temperatures = Vec::new();

    let refinements = [2, 3, 4];
    for cells in refinements {
        let mesh = slab_mesh(cells);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints.clear();
        model.boundaries.clear();
        model.excitations.clear();
        let thermal = model.materials[0]
            .thermal
            .as_mut()
            .expect("thermal fixture declares thermal capability");
        thermal.conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(1.0);
        thermal.specific_heat_j_per_kgk = runmat_types::physical::SpecificHeat::new(1.0);
        model.materials[0].mass_density_kg_per_m3 =
            (Some(1.0)).map(runmat_types::physical::MassDensity::new);
        for (id, region) in [
            ("left_temperature", "slab_left"),
            ("right_temperature", "slab_right"),
        ] {
            model
                .constraints
                .push(AnalysisConstraint::Thermal(ThermalConstraint {
                    constraint_id: id.into(),
                    kind: ThermalConstraintKind::Prescribed {
                        target: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_boundary_selection(&(region)),
                        ),
                        value: PrescribedValue::Constant(Temperature::new(300.0)),
                    },
                }));
        }
        let assembly = assemble(
            &model,
            &mesh,
            crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
                .unwrap()
                .support(),
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let mut temperature = assembly
            .projection
            .nodes
            .iter()
            .map(|node| {
                300.0
                    + 20.0 * (std::f64::consts::PI * node.coordinates_m.canonical_value()[0]).sin()
            })
            .collect::<Vec<_>>();
        for _ in 0..256 {
            let operator = assembly
                .transient_operator(
                    &model,
                    ExcitationSamplePoint::Static,
                    time_step_s,
                    &temperature,
                    &vec![0.0; temperature.len()],
                )
                .unwrap();
            let solve = solve_operator_system(
                &operator,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                Some(&temperature),
            );
            assert!(solve.converged);
            temperature = solve.solution;
        }
        let decay = (-std::f64::consts::PI.powi(2) * final_time_s).exp();
        let squared_error = assembly
            .projection
            .nodes
            .iter()
            .zip(&temperature)
            .map(|(node, numerical)| {
                let exact = 300.0
                    + 20.0
                        * (std::f64::consts::PI * node.coordinates_m.canonical_value()[0]).sin()
                        * decay;
                (numerical - exact).powi(2)
            })
            .sum::<f64>();
        errors.push((squared_error / temperature.len() as f64).sqrt());
        midspan_temperatures.push(assembly.projection.nodes.iter().zip(&temperature).find_map(
            |(node, value)| {
                ((node.coordinates_m.canonical_value()[0] - 0.5).abs() <= 1.0e-12).then_some(*value)
            },
        ));
    }

    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "spatial errors: {errors:?}; midspan temperatures: {midspan_temperatures:?}"
    );
    let observed_orders = errors
        .windows(2)
        .zip(refinements.windows(2))
        .map(|(error, cells)| (error[0] / error[1]).ln() / (cells[1] as f64 / cells[0] as f64).ln())
        .collect::<Vec<_>>();
    assert!(
        observed_orders.iter().all(|order| *order > 1.3),
        "spatial errors: {errors:?}; observed orders: {observed_orders:?}"
    );
}
