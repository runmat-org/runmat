//! Two-way Joule heating against the closed-form 1D solution. A unit bar carries a fixed
//! current density J; its ends are held at T0 and its sides are insulated. With
//! rho(T) = rho0 (1 + alpha (T - T0)) the temperature rise theta solves
//!   theta'' + beta (1 + alpha theta) = 0,  beta = J^2 rho0 / k,  theta(0) = theta(L) = 0,
//! so theta(x) = (cos(m (x - L/2)) / cos(m L / 2) - 1) / alpha with m^2 = alpha beta.
use super::super::*;
use crate::physics::electrical::tests::{mesh, model};
use runmat_analysis_core::*;
use runmat_meshing_core::ElementOrder;

const SIGMA: f64 = 3.0;
const ALPHA: f64 = 0.004;
const T0: f64 = 300.0;
const K: f64 = 1.0;
const CURRENT: f64 = 34.641016151377546; // J^2 rho0 / k = 400 on the unit cube

fn coupled_model() -> AnalysisModel {
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    let law = model.materials[0].electrical.as_mut().unwrap();
    law.conductivity_s_per_m = runmat_types::physical::Quantity::new(SIGMA);
    law.resistive_heating_coefficient = runmat_types::physical::Quantity::new(ALPHA);
    law.reference_temperature_k = runmat_types::physical::Quantity::new(T0);
    model.materials[0].thermal = Some(MaterialThermalModel {
        reference_temperature_k: runmat_types::physical::Quantity::new(T0),
        modulus_temp_coeff_per_k: runmat_types::physical::Quantity::new(0.0),
        conductivity_w_per_mk: runmat_types::physical::Quantity::new(K),
        specific_heat_j_per_kgk: runmat_types::physical::Quantity::new(1.0),
        expansion_coefficient_per_k: runmat_types::physical::Quantity::new(0.0),
    });
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(T0),
    });
    let mut step = ProcedureStep::thermal_steady("heat");
    step.family = PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal);
    model.sequence = AnalysisSequence::from_steps("joule", vec![step]);
    for face in ["slab_left", "slab_right"] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: format!("{face}_temperature"),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(T0)),
                },
            }));
    }
    crate::physics::electrical::tests::prescribe(
        &mut model,
        "ground".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        0.0,
    );
    model.signals.push(PhysicalSignal::ElectricCurrent(Signal {
        signal_id: SignalId("supply".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(ElectricCurrent::new(
            CURRENT,
        ))),
        precision: SignalPrecision::Real64,
        provenance: SignalProvenance::Authored,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
    }));
    model
        .excitations
        .push(ExcitationAction::ElectricCurrent(AnalysisExcitation {
            excitation_id: ExcitationId("supply".into()),
            target: ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_left"),
            )),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("supply".into()),
            active_step_ids: vec!["heat".into()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-12,
            },
        }));
    // Every excitation belongs to a declared case.
    model
        .excitation_cases
        .push(runmat_analysis_core::ExcitationCase {
            case_id: runmat_analysis_core::ExcitationCaseId("supply".into()),
            action_ids: vec![ExcitationId("supply".into())],
            activation: runmat_analysis_core::ExcitationCaseActivation::Always,
        });
    model
}

fn exact_rise(x: f64) -> f64 {
    let beta = CURRENT * CURRENT / SIGMA / K;
    let m = (ALPHA * beta).sqrt();
    ((m * (x - 0.5)).cos() / (m * 0.5).cos() - 1.0) / ALPHA
}

// witness: FEA-COUPLING-ELECTROTHERMAL-002/two_way_conductivity_feedback @
//   electrothermal/tet10/numerical_kernel; coverage: partial;
//   check: two-way-joule-heating; boundary: Two-way coupled steady Joule heating on a Tet10 bar with temperature-dependent
//   resistivity iterates at least three sweeps; temperature matches the closed-form cosine profile
//   with second-order convergence on 2, 4 and 8 cells, terminal power equals the Joule integral,
//   and power matches I^2 R(T) within 0.5 percent. One bar geometry; the busbar CLI study is not
//   run.
#[test]
fn two_way_joule_heating_matches_the_closed_form_bar() {
    let mut errors = Vec::new();
    for cells in [2, 4, 8] {
        let mesh = mesh(ElementOrder::Tet10, cells);
        let coupled = coupled_model();
        let step = coupled.sequence.steps[0].clone();
        let electrical = electrical_part(&coupled, &step);
        let thermal_model = thermal_part(&coupled, &step);
        let template = ThermalWorkload {
            model: thermal_model,
            step_id: ProcedureStepId("heat".into()),
            backend: ComputeBackend::Cpu,
            procedure: ThermalProcedureControl::Steady,
            residual_target: 1e-9,
            initial_fields: vec![],
        };
        let completion = execute_coupled_electrothermal(
            &electrical,
            &template,
            &mesh,
            1e-9,
            ElectrothermalCouplingControls::default(),
        )
        .unwrap();
        assert!(completion.converged);
        assert!(completion.sweeps.len() >= 3, "two-way coupling iterates");
        let nodal = completion.thermal.temperature_snapshots[0]
            .as_host_f64()
            .unwrap();
        let mut worst = 0.0_f64;
        for (node, temperature) in mesh.topology.nodes.iter().zip(nodal) {
            let exact = T0 + exact_rise(node.coordinates_m.canonical_value()[0]);
            worst = worst.max((temperature - exact).abs());
        }
        let peak = exact_rise(0.5);
        eprintln!(
            "electrothermal bar cells={cells}: sweeps={} max |T - T_exact|={worst:.3e} K of peak rise {peak:.3} K",
            completion.sweeps.len()
        );
        errors.push(worst / peak);
        // Terminal power (sum V I) equals the Joule integral at the solved temperatures.
        let sample = completion.electrical.last().unwrap();
        let joule = CoupledElectrothermalCompletion::joule_power_w(sample);
        let terminal = CoupledElectrothermalCompletion::terminal_power_w(sample);
        assert!(
            (joule - terminal).abs() < 1e-8 * joule,
            "P={joule} VI={terminal}"
        );
        // Closed-form resistance R = int rho0 (1 + alpha theta) dx / A on the unit bar.
        let n = 2000;
        let resistance: f64 = (0..n)
            .map(|i| {
                let x = (i as f64 + 0.5) / n as f64;
                (1.0 + ALPHA * exact_rise(x)) / SIGMA / n as f64
            })
            .sum();
        let exact_power = CURRENT * CURRENT * resistance;
        eprintln!("  power {joule:.6} W vs I^2 R(T) {exact_power:.6} W");
        assert!((joule - exact_power).abs() < 5e-3 * exact_power);
    }
    // Each element's conductivity is taken at its mean temperature: second-order convergence.
    assert!(
        errors[2] < errors[1] / 3.0 && errors[2] < 3e-3,
        "{errors:?}"
    );
}
