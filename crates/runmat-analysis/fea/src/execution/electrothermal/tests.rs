use super::*;
use crate::physics::electrical::tests::{mesh, model, prescribe};
use runmat_analysis_core::*;
use runmat_meshing_core::ElementOrder;

mod coupled;
mod source;

fn electrical_workload(voltage: f64, gauge: f64) -> ElectricalStaticWorkload {
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        gauge,
    );
    prescribe(
        &mut model,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        gauge + voltage,
    );
    ElectricalStaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-11,
        element_temperature_k: None,
        operating_time_s: None,
    }
}

fn thermal() -> ThermalWorkload {
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    model.electromagnetic = None;
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(300.0),
    });
    model.materials[0].thermal = Some(MaterialThermalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(300.0),
        modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(1.0),
        specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(1.0),
        expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(0.0),
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_steady("heat")]);
    for face in ["slab_left", "slab_right"] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: face.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(300.0)),
                },
            }));
    }
    ThermalWorkload {
        model,
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::Steady,
        residual_target: 1e-11,
        initial_fields: vec![],
    }
}

// witness: FEA-COUPLING-ELECTROTHERMAL-001/heat @ electrothermal/tet4|tet10/public_study;
//   check: one-way-joule-slab; boundary: Both-order constant-conductivity same-mesh steady one-way conduction to thermal slab;
//   12 W terminal/supplied/boundary power, Tet10 quadratic temperature and cold transfer. No
//   feedback or harmonic Maxwell coupling.
#[test]
fn retained_conduction_drives_independent_quadratic_thermal_patch_and_power() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let electrical = electrical_workload(2.0, 0.0);
        let result = electrical.execute(&mesh).unwrap();
        let source = result.source_result_identity().unwrap();
        let retained_bytes = result.canonical_bytes().unwrap();
        drop(result);
        let cold = ElectricalStaticSolverResult::from_canonical_bytes(&retained_bytes).unwrap();
        let terminal_power = -cold
            .response()
            .terminals
            .iter()
            .map(|terminal| {
                let index = cold
                    .response()
                    .ordered_node_ids
                    .iter()
                    .position(|id| *id == terminal.ordered_node_ids[0])
                    .unwrap();
                cold.response().potential_v.canonical_values()[index]
                    * terminal.outward_flux.canonical_value()
            })
            .sum::<f64>();
        assert!((terminal_power - 12.0).abs() < 1e-8);
        let completion = execute_steady_electrothermal_from_retained(
            &electrical,
            &cold,
            source,
            &thermal(),
            &mesh,
        )
        .unwrap();
        assert!((completion.transfer.power_w() - 12.0).abs() < 1e-8);
        let balance = completion.thermal.energy_balances[0];
        assert!((balance.applied_power_w.canonical_value() - 12.0).abs() < 1e-8);
        assert!((balance.prescribed_boundary_power_w.canonical_value() - 12.0).abs() < 1e-8);
        assert!(balance.relative_residual < 1e-10);
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let values = completion.thermal.temperature_snapshots[0]
            .as_host_f64()
            .unwrap();
        let maximum_error = projection
            .nodes
            .iter()
            .zip(values)
            .map(|(node, value)| {
                let x = node.coordinates_m.canonical_value()[0];
                (value - (300.0 + 6.0 * x * (1.0 - x))).abs()
            })
            .fold(0.0_f64, f64::max);
        if order == ElementOrder::Tet10 {
            assert!(
                maximum_error < 1e-8,
                "quadratic patch error={maximum_error:e}"
            );
        }
        let bytes = completion.transfer.canonical_bytes().unwrap();
        let imported = RetainedJouleHeatTransfer::from_canonical_bytes_for(
            &bytes,
            &electrical,
            &cold,
            &mesh,
            source,
        )
        .unwrap();
        assert_eq!(imported, completion.transfer);
    }
}

// witness: FEA-COUPLING-ELECTROTHERMAL-001/heat @ electrothermal/tet4|tet10/public_study;
//   check: one-way-joule-slab; boundary: Tet4 two-mesh interior temperature refinement and Tet10 voltage sign/gauge invariance
//   at constant conductivity; no two-way thermal iteration.
#[test]
fn tet4_refinement_and_voltage_sign_gauge_preserve_independent_heat_and_power() {
    let mut errors = Vec::new();
    for cells in [2, 4] {
        let mesh = mesh(ElementOrder::Tet4, cells);
        let electrical = electrical_workload(2.0, 0.0);
        let source = electrical.execute(&mesh).unwrap();
        let completion = execute_steady_electrothermal_from_retained(
            &electrical,
            &source,
            source.source_result_identity().unwrap(),
            &thermal(),
            &mesh,
        )
        .unwrap();
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let values = completion.thermal.temperature_snapshots[0]
            .as_host_f64()
            .unwrap();
        // Cell-interior interpolation measures approximation error even when
        // a structured linear patch happens to reproduce nodal values exactly.
        let error = projection
            .elements
            .iter()
            .map(|element| {
                let interpolated = element
                    .node_indices
                    .iter()
                    .map(|index| values[*index])
                    .sum::<f64>()
                    / 4.0;
                let x = element.centroid_m[0];
                element.signed_volume_m3.abs()
                    * (interpolated - (300.0 + 6.0 * x * (1.0 - x))).powi(2)
            })
            .sum::<f64>();
        errors.push(error.sqrt());
    }
    println!("electrothermal Tet4 cell-interior RMS errors={errors:?}");
    assert!(
        errors[1] < 0.35 * errors[0],
        "Tet4 quadratic-temperature interpolation must show second-order refinement: {errors:?}"
    );
    let mesh = mesh(ElementOrder::Tet10, 2);
    let mut temperatures = Vec::new();
    for (voltage, gauge) in [(2.0, 0.0), (-2.0, 7.0), (2.0, 7.0)] {
        let electrical = electrical_workload(voltage, gauge);
        let source = electrical.execute(&mesh).unwrap();
        let completion = execute_steady_electrothermal_from_retained(
            &electrical,
            &source,
            source.source_result_identity().unwrap(),
            &thermal(),
            &mesh,
        )
        .unwrap();
        assert!((completion.transfer.power_w() - 12.0).abs() < 1e-8);
        temperatures.push(
            completion.thermal.temperature_snapshots[0]
                .as_host_f64()
                .unwrap()
                .to_vec(),
        );
    }
    for values in &temperatures[1..] {
        assert!(values
            .iter()
            .zip(&temperatures[0])
            .all(|(a, b)| (a - b).abs() < 1e-8));
    }
}

#[test]
fn retained_transfer_and_unsupported_compositions_fail_closed() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let electrical = electrical_workload(2.0, 0.0);
    let result = electrical.execute(&mesh).unwrap();
    let source = result.source_result_identity().unwrap();
    let (transfer, _) =
        bind_retained_joule_heat(&electrical, &result, source, &thermal(), &mesh).unwrap();
    for mutation in ["density_w_per_m3", "power_w", "ordered_element_ids"] {
        let mut wire = serde_json::to_value(&transfer).unwrap();
        match mutation {
            "density_w_per_m3" => wire[mutation][0] = serde_json::json!(13.0),
            "power_w" => wire[mutation] = serde_json::json!(13.0),
            _ => wire[mutation].as_array_mut().unwrap().swap(0, 1),
        }
        let altered: RetainedJouleHeatTransfer = serde_json::from_value(wire).unwrap();
        assert!(altered
            .validate_for(&electrical, &result, &mesh, source)
            .is_err());
        assert!(RetainedJouleHeatTransfer::from_canonical_bytes_for(
            &altered.canonical_bytes().unwrap(),
            &electrical,
            &result,
            &mesh,
            source
        )
        .is_err());
    }
    let foreign = electrical_workload(1.0, 0.0);
    let foreign_result = foreign.execute(&mesh).unwrap();
    assert!(bind_retained_joule_heat(
        &electrical,
        &result,
        foreign_result.source_result_identity().unwrap(),
        &thermal(),
        &mesh
    )
    .is_err());
    let changed_mesh = crate::physics::electrical::tests::mesh(ElementOrder::Tet4, 3);
    assert!(
        bind_retained_joule_heat(&electrical, &result, source, &thermal(), &changed_mesh).is_err()
    );
    let mut electrostatic = electrical.clone();
    electrostatic.model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::Electrostatics,
    });
    let electrostatic_result = electrostatic.execute(&mesh).unwrap();
    assert!(bind_retained_joule_heat(
        &electrostatic,
        &electrostatic_result,
        electrostatic_result.source_result_identity().unwrap(),
        &thermal(),
        &mesh
    )
    .is_err());
    let mut feedback = electrical.clone();
    feedback.model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .resistive_heating_coefficient = runmat_types::physical::InverseTemperature::new(0.01);
    assert!(bind_retained_joule_heat(&feedback, &result, source, &thermal(), &mesh).is_err());
    let mut transient = thermal();
    transient.model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_transient("heat")]);
    transient.procedure = ThermalProcedureControl::BackwardEuler { snapshot_count: 2 };
    assert!(bind_retained_joule_heat(&electrical, &result, source, &transient, &mesh).is_err());
    let mut external_heat = thermal();
    external_heat.model.boundaries.push(AnalysisBoundary {
        boundary_id: "external_heat".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(1.0)),
        },
    });
    assert!(bind_retained_joule_heat(&electrical, &result, source, &external_heat, &mesh).is_err());
}
