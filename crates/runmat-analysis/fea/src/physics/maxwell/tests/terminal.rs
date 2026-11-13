//! Terminal-driven conductors against closed-form impedances (independent of RunMat's
//! element matrices): a conducting strip of thickness t under an air gap g, returned
//! through a perfectly conducting plane, with fields uniform across its width W and along
//! its length L (parallel-plate line). With k^2 = i omega mu sigma, the impedance is
//!   Z = (L / W) [ k coth(k t) / sigma + i omega mu0 g ]
//! which is the 1D skin-effect solution: the internal impedance of the strip plus the
//! inductance of the gap.
use super::super::*;
use super::layered;
use runmat_analysis_core::*;

const MU0: f64 = crate::physics::magnetic::VACUUM_PERMEABILITY_H_PER_M.canonical_value();

struct Line {
    width: f64,
    length: f64,
    thickness: f64,
    gap: f64,
    sigma: f64,
}

#[derive(Clone, Copy)]
struct C(f64, f64);
impl C {
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C(
            (self.0 * o.0 + self.1 * o.1) / d,
            (self.1 * o.0 - self.0 * o.1) / d,
        )
    }
    fn exp(self) -> C {
        let m = self.0.exp();
        C(m * self.1.cos(), m * self.1.sin())
    }
    fn coth(self) -> C {
        let a = C(2.0 * self.0, 2.0 * self.1).exp();
        C(a.0 + 1.0, a.1).div(C(a.0 - 1.0, a.1))
    }
}

impl Line {
    fn impedance(&self, frequency: f64) -> (f64, f64) {
        let omega = std::f64::consts::TAU * frequency;
        // k = sqrt(i omega mu sigma) = (1 + i) / delta
        let delta = (2.0 / (omega * MU0 * self.sigma)).sqrt();
        let k = C(1.0 / delta, 1.0 / delta);
        let internal = k.mul(C(k.0 * self.thickness, k.1 * self.thickness).coth());
        let scale = self.length / self.width;
        (
            scale * internal.0 / self.sigma,
            scale * (internal.1 / self.sigma + omega * MU0 * self.gap),
        )
    }

    fn mesh(
        &self,
        conductor_cells: usize,
        air_cells: usize,
    ) -> runmat_meshing_core::SolverMeshArtifact {
        let mut zs = layered::lines(0.0, self.thickness, conductor_cells);
        zs.extend(
            layered::lines(self.thickness, self.thickness + self.gap, air_cells)
                .into_iter()
                .skip(1),
        );
        let t = self.thickness;
        let (l, top) = (self.length, self.thickness + self.gap);
        layered::mesh(
            &layered::lines(0.0, self.width, 1),
            // Lowest-order edge fields that vary across the thickness carry a companion
            // component that varies along the current, so refine both directions together.
            &layered::lines(0.0, self.length, conductor_cells.max(2)),
            &zs,
            move |center| if center[2] < t { "conductor" } else { "air" },
            move |centroid, _normal, regions| {
                let body = if regions.len() == 2 {
                    return "interface".into();
                } else {
                    regions[0]
                };
                if centroid[1].abs() < 1e-12 {
                    format!("y_min_{body}")
                } else if (centroid[1] - l).abs() < 1e-12 {
                    format!("y_max_{body}")
                } else if (centroid[2] - top).abs() < 1e-12 {
                    "top".into()
                } else {
                    "side".into()
                }
            },
        )
    }

    fn model(&self, frequencies: Vec<f64>, drive: Drive) -> AnalysisModel {
        let mut model =
            crate::physics::electrical::tests::model(StaticElectricalEquation::SteadyConduction);
        model.electromagnetic = None;
        let mut step = ProcedureStep::electromagnetic_harmonic("field");
        step.independent_domain = IndependentDomain::FrequencyHertz(frequencies);
        model.sequence = AnalysisSequence::from_steps("line", vec![step]);
        let conductor = &mut model.materials[0];
        conductor.material_id = "conductor".into();
        conductor.electrical = Some(MaterialElectricalModel {
            reference_temperature_k: runmat_types::physical::Quantity::new(293.15),
            conductivity_s_per_m: runmat_types::physical::Quantity::new(self.sigma),
            relative_permittivity: runmat_types::physical::Quantity::new(1.0),
            relative_permeability: runmat_types::physical::Quantity::new(1.0),
            resistive_heating_coefficient: runmat_types::physical::Quantity::new(0.0),
            conductivity_frequency_response: Vec::new(),
        });
        let mut air = model.materials[0].clone();
        air.material_id = "air".into();
        air.electrical.as_mut().unwrap().conductivity_s_per_m =
            runmat_types::physical::Quantity::new(0.0);
        model.materials.push(air);
        model.material_assignments[0].target =
            PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("conductor"));
        model.material_assignments[0].expected_material_id = "conductor".into();
        model.material_assignments[0].assigned_material_id = "conductor".into();
        let mut assignment = model.material_assignments[0].clone();
        assignment.target =
            PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("air"));
        assignment.expected_material_id = "air".into();
        assignment.assigned_material_id = "air".into();
        model.material_assignments.push(assignment);
        for face in [
            "y_min_conductor",
            "y_max_conductor",
            "y_min_air",
            "y_max_air",
            "top",
        ] {
            model
                .constraints
                .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                    constraint_id: format!("ground_{face}"),
                    kind: MagneticConstraintKind::VectorPotentialGround {
                        target: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                        ),
                    },
                }));
        }
        let potential = |model: &mut AnalysisModel, id: &str, face: &str, volts: f64| {
            model
                .constraints
                .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                    constraint_id: id.into(),
                    kind: ElectricalConstraintKind::PrescribedPotential {
                        target: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                        ),
                        value: PrescribedValue::Constant(ElectricPotential::new(volts)),
                    },
                }));
        };
        potential(&mut model, "return", "y_max_conductor", 0.0);
        match drive {
            Drive::Current(amperes) => {
                model.signals.push(PhysicalSignal::ElectricCurrent(Signal {
                    signal_id: SignalId("drive".into()),
                    shape: SignalShape::Scalar,
                    frame: None,
                    representation: SignalRepresentation::Constant(SignalValue::Scalar(
                        ElectricCurrent::new(amperes),
                    )),
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
                        excitation_id: ExcitationId("drive".into()),
                        target: ExcitationTarget::Region(RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("y_min_conductor"),
                        )),
                        frame: CoordinateFrame::Global,
                        direction: None,
                        distribution: ExcitationDistribution::PreserveResultant,
                        kinematics: ExcitationKinematics::FixedFrame,
                        signal_id: SignalId("drive".into()),
                        active_step_ids: vec!["field".into()],
                        combination_group: None,
                        projection: ProjectionRequirement {
                            preserve_resultant: true,
                            preserve_moment: false,
                            relative_tolerance: 1e-12,
                        },
                    }));
                model.excitation_cases.push(ExcitationCase {
                    case_id: ExcitationCaseId("drive".into()),
                    action_ids: vec![ExcitationId("drive".into())],
                    activation: ExcitationCaseActivation::Always,
                });
            }
            Drive::Voltage(volts) => potential(&mut model, "feed", "y_min_conductor", volts),
        }
        model
    }
}

#[derive(Clone, Copy)]
enum Drive {
    Current(f64),
    Voltage(f64),
}

fn strip() -> Line {
    Line {
        width: 1e-3,
        length: 2e-3,
        thickness: 1e-3,
        gap: 0.5e-3,
        sigma: 1e6,
    }
}

fn solve_line(line: &Line, frequency: f64, drive: Drive, cells: usize) -> MaxwellResponse {
    let mesh = line.mesh(cells, (cells / 2).max(2));
    let model = line.model(vec![frequency], drive);
    solve_canonical_harmonic(&model, &mesh, "field", 1e-9, frequency).unwrap()
}

// witness: FEA-EM-FREQUENCY-001/skin_effect_reference @
//   harmonic_maxwell/edge_elements/numerical_kernel; coverage: partial;
//   boundary: A current-driven conducting strip at t/delta of 0.5, 2 and 4 on 8, 16 and 32 cells:
//   impedance converges to the closed-form skin-effect solution (error below 2e-3 and refinement
//   ratio above 3). Strip impedance only; no round conductor and no current-density profile
//   comparison.
// witness: FEA-EM-FREQUENCY-001/ac_loss_power @ harmonic_maxwell/edge_elements/numerical_kernel;
//   coverage: partial;
//   boundary: On the strip, the Joule loss equals the delivered complex power within 1e-8 and the
//   Poynting and charge-continuity residuals stay below 1e-8; no multi-body gear or inductor loss.
// witness: FEA-EM-QUALIFICATION-001/references @ harmonic_maxwell/edge_elements/numerical_kernel;
//   coverage: partial;
//   boundary: Independent skin-effect closed form for strip impedance with convergence in mesh size
//   at three frequencies.
// witness: FEA-EM-MAXWELL-001/frequency @ harmonic_maxwell/edge_elements/numerical_kernel;
//   coverage: partial;
//   check: skin-effect; boundary: Three frequencies are solved with complex coefficients and each matches its closed
//   form; composition of several frequencies in one study is not asserted.
#[test]
fn current_driven_strip_impedance_converges_to_the_skin_effect_solution() {
    let line = strip();
    for skin_depths in [0.5, 2.0, 4.0] {
        // thickness / skin depth = skin_depths
        let delta = line.thickness / skin_depths;
        let frequency = 2.0 / (MU0 * line.sigma * delta * delta) / std::f64::consts::TAU;
        let (r, x) = line.impedance(frequency);
        let mut errors = Vec::new();
        for cells in [8, 16, 32] {
            let response = solve_line(&line, frequency, Drive::Current(2.0), cells);
            let [drive] = response.drives.as_slice() else {
                panic!("one drive")
            };
            assert!(
                (drive.current_a.canonical_value().real - 2.0).abs() < 1e-9
                    && drive.current_a.canonical_value().imaginary.abs() < 1e-9
            );
            let z = (
                drive.voltage_v.canonical_value().real / 2.0,
                drive.voltage_v.canonical_value().imaginary / 2.0,
            );
            // Joule loss equals (1/2) Re(U conj I) and the Poynting balance closes.
            let delivered = drive.complex_power_va.canonical_value().real;
            assert!(
                (response.ohmic_power_w - runmat_types::physical::Quantity::new(delivered)).abs()
                    <= runmat_types::physical::Quantity::new(1e-8 * delivered),
                "loss {} vs delivered {}",
                response.ohmic_power_w.canonical_value(),
                delivered
            );
            assert!(response.relative_complex_power_error < 1e-8);
            assert!(response.relative_charge_continuity_residual < 1e-8);
            let error = ((z.0 - r).powi(2) + (z.1 - x).powi(2)).sqrt() / (r * r + x * x).sqrt();
            eprintln!(
                "strip t/delta={skin_depths} cells={cells}: Z=({:.6e}, {:.6e}) exact=({r:.6e}, {x:.6e}) error={error:.3e}",
                z.0, z.1
            );
            errors.push(error);
        }
        assert!(errors[2] < errors[0] / 3.0, "refinement {errors:?}");
        assert!(errors[2] < 2e-3, "t/delta={skin_depths}: {errors:?}");
    }
}

// witness: FEA-EM-QUALIFICATION-001/references @ harmonic_maxwell/edge_elements/numerical_kernel;
//   coverage: partial;
//   check: low-frequency-strip; boundary: A voltage-driven strip at 1 Hz matches the DC resistance and the low-frequency
//   inductance.
#[test]
fn voltage_drive_and_low_frequency_limit_match_dc_resistance_and_inductance() {
    let line = strip();
    let frequency = 1.0;
    let response = solve_line(&line, frequency, Drive::Voltage(1e-3), 16);
    let [drive] = response.drives.as_slice() else {
        panic!("one drive")
    };
    assert_eq!(
        drive.voltage_v.canonical_value(),
        ComplexPhasor::new(1e-3, 0.0)
    );
    let current = drive.current_a.canonical_value();
    let denominator = current.norm_squared();
    let z = (
        (drive.voltage_v.canonical_value().real * current.real
            + drive.voltage_v.canonical_value().imaginary * current.imaginary)
            / denominator,
        (drive.voltage_v.canonical_value().imaginary * current.real
            - drive.voltage_v.canonical_value().real * current.imaginary)
            / denominator,
    );
    let dc = line.length / (line.sigma * line.width * line.thickness);
    let omega = std::f64::consts::TAU * frequency;
    let inductance = MU0 * line.length / line.width * (line.gap + line.thickness / 3.0);
    assert!((z.0 - dc).abs() < 1e-6 * dc, "R={} dc={dc}", z.0);
    assert!(
        (z.1 - omega * inductance).abs() < 2e-2 * omega * inductance,
        "X={} omega L={}",
        z.1,
        omega * inductance
    );
}

/// A conductor whose terminals face air inside the model (a gap feed): the terminal current
/// closes through the transparent air, the drive carries exactly the authored current, the
/// loss equals the delivered power and the weak charge continuity holds.
#[test]
fn gap_fed_conductor_closes_its_terminal_current_through_air() {
    let side = 0.02;
    let xs = layered::lines(0.0, side, 6);
    let ys = layered::lines(0.0, side, 6);
    let zs = layered::lines(0.0, side, 6);
    let lo = side / 3.0;
    let hi = 2.0 * side / 3.0;
    let inside = move |c: [f64; 3]| (0..3).all(|a| c[a] > lo && c[a] < hi);
    let mesh = layered::mesh(
        &xs,
        &ys,
        &zs,
        move |center| if inside(center) { "conductor" } else { "air" },
        move |centroid, normal, regions| {
            if regions.len() == 2 {
                // Conductor faces normal to x are the terminals.
                if normal[0].abs() > 0.9 {
                    return if centroid[0] < 0.5 * side {
                        "terminal_in".into()
                    } else {
                        "terminal_out".into()
                    };
                }
                return "interface".into();
            }
            "outer".into()
        },
    );
    let line = strip();
    let mut model = line.model(vec![1.0e3], Drive::Current(5.0));
    // Re-target the strip model's faces onto this box.
    model
        .constraints
        .retain(|c| !matches!(c, AnalysisConstraint::Magnetic(_)));
    model
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "far_field".into(),
            kind: MagneticConstraintKind::VectorPotentialGround {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "outer",
                )),
            },
        }));
    for constraint in &mut model.constraints {
        if let AnalysisConstraint::Electrical(constraint) = constraint {
            if let ElectricalConstraintKind::PrescribedPotential { target, .. } =
                &mut constraint.kind
            {
                *target = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "terminal_out",
                ));
            }
        }
    }
    for action in &mut model.excitations {
        if let ExcitationAction::ElectricCurrent(action) = action {
            action.target = ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("terminal_in"),
            ));
        }
    }
    let response = solve_canonical_harmonic(&model, &mesh, "field", 1e-9, 1.0e3).unwrap();
    let [drive] = response.drives.as_slice() else {
        panic!("one drive")
    };
    assert!(
        (drive.current_a.canonical_value().real - 5.0).abs() < 1e-8
            && drive.current_a.canonical_value().imaginary.abs() < 1e-8
    );
    assert!(response.relative_charge_continuity_residual < 1e-8);
    assert!(response.relative_complex_power_error < 1e-8);
    // DC resistance of the bar (lo..hi cube of side/3) and its loss at 1 kHz (no skin effect).
    let l = side / 3.0;
    let r_dc = l / (line.sigma * l * l);
    let delivered = drive.complex_power_va.canonical_value().real;
    eprintln!(
        "gap feed: U={:?} R_dc={r_dc:e} delivered={delivered:e} loss={:e}",
        drive.voltage_v.canonical_value(),
        response.ohmic_power_w.canonical_value()
    );
    // The closure flow in the air is a modeling device that carries no physical power.
    assert!(
        (response.ohmic_power_w - runmat_types::physical::Quantity::new(delivered)).abs()
            < runmat_types::physical::Quantity::new(1e-4 * delivered)
    );
    assert!((delivered - 0.5 * 25.0 * r_dc).abs() < 1e-2 * delivered);
}

#[test]
fn tabulated_frequency_response_scales_the_conductor_at_the_analysis_frequency() {
    let line = strip();
    let frequency = 2.0e4;
    let mesh = line.mesh(8, 4);
    let mut tabulated = line.model(vec![frequency], Drive::Current(2.0));
    tabulated.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_frequency_response = vec![
        runmat_analysis_core::ConductivityFrequencyPoint {
            frequency_hz: runmat_types::physical::Quantity::new(1.0e4),
            conductivity_scale: 0.25,
            dispersive_loss_scale: None,
            relative_permittivity_scale: None,
            relative_permeability_scale: None,
        },
        runmat_analysis_core::ConductivityFrequencyPoint {
            frequency_hz: runmat_types::physical::Quantity::new(3.0e4),
            conductivity_scale: 0.75,
            dispersive_loss_scale: None,
            relative_permittivity_scale: None,
            relative_permeability_scale: None,
        },
    ];
    let scaled = Line {
        sigma: line.sigma * 0.5,
        ..strip()
    };
    let reference = solve_canonical_harmonic(
        &scaled.model(vec![frequency], Drive::Current(2.0)),
        &mesh,
        "field",
        1e-9,
        frequency,
    )
    .unwrap();
    let actual = solve_canonical_harmonic(&tabulated, &mesh, "field", 1e-9, frequency).unwrap();
    let (a, b) = (
        &actual.drives[0].voltage_v.canonical_value(),
        &reference.drives[0].voltage_v.canonical_value(),
    );
    assert!((a.real - b.real).abs() <= 1e-9 * b.real.abs());
    assert!((a.imaginary - b.imaginary).abs() <= 1e-9 * b.imaginary.abs().max(b.real.abs()));
    // Outside the tabulated band the material is undefined.
    assert!(solve_canonical_harmonic(&tabulated, &mesh, "field", 1e-9, 5.0e4).is_err());
}
