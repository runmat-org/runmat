use super::*;
mod continuum;
mod interface;
mod incident;
mod radiation;
mod volume_velocity;
use crate::fixtures::{fixture_model, replace_fixture_excitations, FixtureId};
use crate::topology::test_meshes::slab_mesh;
use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisBoundary, AnalysisConstraint,
    BoundaryOperator, MaterialAcousticModel, PrescribedValue, Pressure, ProcedureStep,
    RegionTarget, SpecificAcousticImpedance,
};
use runmat_analysis_core::{
    AnalysisExcitation, CoordinateFrame, ExcitationAction, ExcitationCase,
    ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution, ExcitationId,
    ExcitationKinematics, ExcitationTarget, PhysicalSignal, ProjectionRequirement,
    PropertyRegionTarget, Signal, SignalApproximationEvidence, SignalExtrapolation, SignalId,
    SignalInterpolation, SignalKnot, SignalOrdinate, SignalPrecision, SignalProvenance,
    SignalRepresentation, SignalShape, SignalValue, TabulatedSignal, TabulatedSignalDomain,
    Velocity,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn complex_signal<Q: Clone>(real: Q, imaginary: Q) -> Signal<Q> {
    Signal {
        signal_id: SignalId("drive".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::FrequencyHertz,
            knots: [0.01, 1.0]
                .into_iter()
                .map(|f| {
                    SignalKnot::continuous(
                        f,
                        SignalOrdinate::Complex {
                            real: SignalValue::Scalar(real.clone()),
                            imaginary: SignalValue::Scalar(imaginary.clone()),
                        },
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Complex64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    }
}

fn velocity_source(model: &mut AnalysisModel, real: f64, imaginary: f64) {
    model.signals.push(PhysicalSignal::Velocity(complex_signal(
        Velocity::new(real),
        Velocity::new(imaginary),
    )));
    model
        .excitations
        .push(ExcitationAction::NormalVelocity(AnalysisExcitation {
            excitation_id: ExcitationId("speaker".into()),
            target: ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_left"),
            )),
            frame: CoordinateFrame::BoundaryNormalTangent,
            direction: None,
            distribution: ExcitationDistribution::UniformKinematic,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("drive".into()),
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("speaker".into()),
        action_ids: vec![ExcitationId("speaker".into())],
        activation: ExcitationCaseActivation::Always,
    });
}

#[test]
fn matched_duct_pressure_and_velocity_sources_converge_to_independent_traveling_wave() {
    // p=(1+0.4i) exp(-ikx), v_x=p/(rho*c); left outward v_n=-v_x.
    // Unit-area matched termination absorbs |p|²/(2 rho c)=1.16/12 W.
    let frequency = 0.2;
    let k = std::f64::consts::TAU * frequency / 2.0;
    for velocity in [false, true] {
        let mut errors = Vec::new();
        for cells in [2, 4, 8] {
            let mesh = slab_mesh(cells);
            let mut model = model();
            model.material_assignments[0].target =
                PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
            if velocity {
                velocity_source(&mut model, -1.0 / 6.0, -0.4 / 6.0);
            } else {
                model.signals.push(PhysicalSignal::Pressure(complex_signal(
                    Pressure::new(1.0),
                    Pressure::new(0.4),
                )));
                model
                    .constraints
                    .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                        constraint_id: "left_pressure".into(),
                        kind: AcousticConstraintKind::PrescribedPressure {
                            target: RegionTarget::boundary(
                                runmat_meshing_core::fixtures::fixture_face("slab_left"),
                            ),
                            value: PrescribedValue::Signal(SignalId("drive".into())),
                        },
                    }));
            }
            model.boundaries.push(AnalysisBoundary {
                boundary_id: "matched".into(),
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
                operator: BoundaryOperator::AcousticImpedance {
                    impedance: SpecificAcousticImpedance::new(6.0),
                },
            });
            let result = solve_canonical_harmonic(
                &model,
                &mesh,
                "acoustic",
                Frequency::new(frequency),
                AcousticHarmonicPolicy {
                    relative_tolerance: 1e-11,
                    ..Default::default()
                },
                1e-8,
            )
            .unwrap();
            let mut squared = 0.0;
            for (node, pressure) in result
                .projection
                .nodes
                .iter()
                .zip(result.pressure_pa.canonical_values())
            {
                let phase = k * node.coordinates_m.canonical_value()[0];
                let real = phase.cos() + 0.4 * phase.sin();
                let imaginary = 0.4 * phase.cos() - phase.sin();
                squared +=
                    (pressure.real - real).powi(2) + (pressure.imaginary - imaginary).powi(2);
            }
            let error = (squared / result.pressure_pa.canonical_values().len() as f64).sqrt();
            errors.push(error);
            assert!(
                result.power_relative_error < 1e-8,
                "power closure {}",
                result.power_relative_error
            );
            assert!((result.mean_impedance_loss_w.canonical_value() - 1.16 / 12.0).abs() < 0.01);
            if velocity {
                assert_eq!(result.source_evidence.len(), 1);
                assert!(result.source_evidence[0].accepted);
            }
            if cells == 8 {
                assert!(error < 0.004, "velocity={velocity}: error={error}");
                let mut error_velocity = 0.0;
                for (element, value) in result
                    .projection
                    .elements
                    .iter()
                    .zip(result.element_velocity_m_per_s.canonical_values())
                {
                    let phase = k * element.centroid_m[0];
                    error_velocity += (value.real[0] - (phase.cos() + 0.4 * phase.sin()) / 6.0)
                        .powi(2)
                        + (value.imaginary[0] - (0.4 * phase.cos() - phase.sin()) / 6.0).powi(2)
                        + value.real[1].powi(2)
                        + value.real[2].powi(2)
                        + value.imaginary[1].powi(2)
                        + value.imaginary[2].powi(2);
                }
                assert!((error_velocity / result.projection.elements.len() as f64).sqrt() < 0.01);
            }
        }
        assert!(
            errors.windows(2).all(|pair| pair[0] / pair[1] > 3.0),
            "velocity={velocity}: {errors:?}"
        );
    }
}

pub(super) fn model() -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 =
        (Some(3.0)).map(runmat_types::physical::MassDensity::new);
    let step = ProcedureStep::new(
        "acoustic",
        PhysicsFamily::Acoustic,
        AnalysisProcedure::HarmonicResponse,
        runmat_analysis_core::SolutionSpace::Direct,
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![0.1, 0.2]),
        Vec::new(),
    );
    model.sequence.steps = vec![step];
    model
}

#[test]
fn canonical_tet4_operators_match_exact_integrals() {
    let model = model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
    let assembly = AcousticAssembly::assemble(
        projection,
        &properties,
        runmat_types::physical::AngularFrequency::new(1.0),
    )
    .unwrap();
    let gradients = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            let k = gradients[i]
                .iter()
                .zip(gradients[j])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / 18.0;
            let m = if i == j { 1.0 / 60.0 } else { 1.0 / 120.0 } / 12.0;
            assert!(
                (assembly
                    .stiffness
                    .coefficient(i, j)
                    .unwrap()
                    .canonical_value()
                    - k)
                    .abs()
                    < 1e-13
            );
            assert!((assembly.mass.coefficient(i, j).unwrap().canonical_value() - m).abs() < 1e-13);
        }
    }
}

fn prescribe(model: &mut AnalysisModel, target: RegionTarget, value: f64) {
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target,
                value: PrescribedValue::Constant(Pressure::new(value)),
            },
        }));
}

#[test]
fn prescribed_pressure_and_impedance_solve_close_power() {
    let mut model = model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let pressure_face = projection
        .boundary_faces
        .iter()
        .find(|face| !face.node_indices.contains(&0))
        .unwrap();
    let target = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
        &(&pressure_face.provenance[0].source_topology_id),
    ));
    prescribe(&mut model, target, 1.0);
    let impedance_face = projection
        .boundary_faces
        .iter()
        .find(|face| face.node_indices.contains(&0))
        .unwrap();
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "termination".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
            &(&impedance_face.provenance[0].source_topology_id),
        )),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    let result = solve_canonical_harmonic(
        &model,
        &mesh,
        "acoustic",
        Frequency::new(0.1),
        AcousticHarmonicPolicy::default(),
        1e-8,
    )
    .unwrap();
    assert!(result.relative_residual < 1e-9);
    assert!(result.mean_input_power_w.canonical_value() > 0.0);
    assert!(result.power_relative_error < 1e-9);
    assert!(result
        .pressure_pa
        .canonical_values()
        .iter()
        .skip(1)
        .all(|p| p.real == 1.0 && p.imaginary == 0.0));
}

#[test]
fn canonical_acoustics_rejects_invalid_frequency_and_material_attenuation() {
    let mut model = model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    assert!(matches!(
        solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(0.0),
            AcousticHarmonicPolicy::default(),
            1e-8,
        ),
        Err(AcousticHarmonicError::InvalidInput(_))
    ));
    model.materials[0].acoustic.as_mut().unwrap().damping_ratio =
        runmat_types::physical::DampingRatio::new(1.5);
    assert!(matches!(
        solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(0.1),
            AcousticHarmonicPolicy::default(),
            1e-8,
        ),
        Err(AcousticHarmonicError::Material(_))
    ));
}

/// Complex arithmetic for the closed-form damped standing wave.
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
    fn sqrt(self) -> C {
        let r = self.0.hypot(self.1).sqrt();
        let t = self.1.atan2(self.0) / 2.0;
        C(r * t.cos(), r * t.sin())
    }
    fn cos(self) -> C {
        C(self.0.cos() * self.1.cosh(), -self.0.sin() * self.1.sinh())
    }
}

// witness: FEA-ACOUSTIC-BOUNDARY-002/region_damping @ acoustic_harmonic/tet4/numerical_kernel;
//   coverage: partial; check: damped-duct; boundary: a lossy duct (loss factor 2ζ) converges at second order to the
//   complex-wavenumber standing wave and dissipates its input power; resonance width not asserted.
#[test]
fn damped_duct_converges_to_the_complex_wavenumber_standing_wave_and_dissipates_its_input() {
    // Loss factor η = 2ζ: c̃ = c√(1 + iη), k̃ = ω/c̃, p = cos(k̃(1 − x))/cos(k̃).
    let (zeta, c, f) = (0.05, 2.0, 0.2);
    let omega = std::f64::consts::TAU * f;
    let k = C(omega, 0.0).div(C(c, 0.0).mul(C(1.0, 2.0 * zeta).sqrt()));
    let mut errors = Vec::new();
    for cells in [2, 4, 8] {
        let mesh = slab_mesh(cells);
        let mut model = model();
        model.materials[0].acoustic.as_mut().unwrap().damping_ratio =
            runmat_types::physical::Quantity::new(zeta);
        model.material_assignments[0].target =
            PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
        prescribe(
            &mut model,
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
            1.0,
        );
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "rigid".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticRigidWall,
        });
        let result = solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(f),
            AcousticHarmonicPolicy {
                relative_tolerance: 1e-11,
                ..Default::default()
            },
            1e-8,
        )
        .unwrap();
        let denominator = k.cos();
        let error = (result
            .projection
            .nodes
            .iter()
            .zip(result.pressure_pa.canonical_values())
            .map(|(node, p)| {
                let exact = k
                    .mul(C(1.0 - node.coordinates_m.canonical_value()[0], 0.0))
                    .cos()
                    .div(denominator);
                (p.real - exact.0).powi(2) + (p.imaginary - exact.1).powi(2)
            })
            .sum::<f64>()
            / result.pressure_pa.canonical_values().len() as f64)
            .sqrt();
        errors.push(error);
        // The driven end supplies exactly the power the lossy air dissipates.
        assert!(result.mean_impedance_loss_w > runmat_types::physical::Quantity::new(0.0));
        assert!(result.power_relative_error < 1e-9);
    }
    assert!(
        errors.windows(2).all(|pair| pair[0] / pair[1] > 3.0),
        "{errors:?}"
    );
    assert!(errors[2] < 0.004, "{errors:?}");
}

#[test]
fn rigid_duct_converges_to_independent_standing_wave_without_artificial_loss() {
    let k = std::f64::consts::TAU * 0.2 / 2.0;
    let mut errors = Vec::new();
    for cells in [2, 4, 8] {
        let mesh = slab_mesh(cells);
        let mut model = model();
        model.material_assignments[0].target =
            PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
        prescribe(
            &mut model,
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
            1.0,
        );
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "rigid".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticRigidWall,
        });
        let result = solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(0.2),
            AcousticHarmonicPolicy {
                relative_tolerance: 1e-11,
                ..Default::default()
            },
            1e-8,
        )
        .unwrap();
        let error = (result
            .projection
            .nodes
            .iter()
            .zip(result.pressure_pa.canonical_values())
            .map(|(node, p)| {
                (p.real - (k * (1.0 - node.coordinates_m.canonical_value()[0])).cos() / k.cos())
                    .powi(2)
                    + p.imaginary.powi(2)
            })
            .sum::<f64>()
            / result.pressure_pa.canonical_values().len() as f64)
            .sqrt();
        errors.push(error);
        assert!(result.mean_impedance_loss_w.canonical_value().abs() < 1e-14);
        assert!(result.mean_input_power_w.canonical_value().abs() < 1e-12);
        assert!(result.power_relative_error < 1e-9);
    }
    assert!(
        errors.windows(2).all(|pair| pair[0] / pair[1] > 3.0),
        "{errors:?}"
    );
    assert!(errors[2] < 0.004);
}

#[test]
fn straight_and_curved_tet10_preserve_integrals_and_affine_velocity_recovery() {
    for warp in [0.0, 0.1, 0.2] {
        let model = model();
        let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        {
            let quantity_change = warp;
            let quantity_index = 0;
            let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        mesh.seal_canonical_digest().unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
        let mut assembly = AcousticAssembly::assemble(
            projection,
            &properties,
            runmat_types::physical::AngularFrequency::new(1.0),
        )
        .unwrap();
        let size = assembly.projection.nodes.len();
        let total_mass = (0..size)
            .flat_map(|i| (0..size).map(move |j| (i, j)))
            .map(|(i, j)| assembly.mass.coefficient(i, j).unwrap().canonical_value())
            .sum::<f64>();
        assert!((total_mass - 1.0 / 72.0).abs() < 1e-12);
        for i in 0..size {
            assert!(
                (0..size)
                    .map(|j| assembly
                        .stiffness
                        .coefficient(i, j)
                        .unwrap()
                        .canonical_value())
                    .sum::<f64>()
                    .abs()
                    < 1e-11
            );
        }
        for (i, node) in assembly.projection.nodes.iter().enumerate() {
            let [x, y, z] = node.coordinates_m.canonical_value();
            assembly.prescribed = assembly
                .prescribed
                .clone()
                .try_map_canonical_storage(|mut values| {
                    values[i] = Some(crate::excitation::ProjectionScalar {
                        real: 1.0 + 2.0 * x + 3.0 * y - 4.0 * z,
                        imaginary: 0.0,
                    });
                    values
                })
                .unwrap();
        }
        let result = super::response::solve(
            assembly,
            1.0 / std::f64::consts::TAU,
            AcousticHarmonicPolicy::default(),
            1e-8,
        )
        .unwrap();
        let velocity = &result.element_velocity_m_per_s.canonical_values()[0];
        assert!(velocity.real.iter().all(|v| v.abs() < 1e-12));
        for (actual, expected) in velocity.imaginary.iter().zip([2.0 / 3.0, 1.0, -4.0 / 3.0]) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert!(result.element_mean_intensity_w_per_m2.canonical_values()[0]
            .iter()
            .all(|i| i.abs() < 1e-12));
        // Straight unit tetrahedron: E[p²]=3 from barycentric second moments,
        // |grad p|²=29; E=(3/12 + 29/3)/4=119/48 J/m³.
        if warp == 0.0 {
            assert!(
                (result
                    .element_mean_energy_density_j_per_m3
                    .canonical_values()[0]
                    - 119.0 / 48.0)
                    .abs()
                    < 1e-12
            );
        }
    }
}

#[test]
fn quadratic_phasor_recovery_integrates_products_not_products_of_averages() {
    let model = model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
    let mut assembly = AcousticAssembly::assemble(
        projection,
        &properties,
        runmat_types::physical::AngularFrequency::new(1.0),
    )
    .unwrap();
    // p=x²+i*x, omega=1, rho=3, c=2. Every nodal value is prescribed,
    // so this isolates recovery from any numerical field-solve approximation.
    for (i, node) in assembly.projection.nodes.iter().enumerate() {
        let x = node.coordinates_m.canonical_value()[0];
        assembly.prescribed = assembly
            .prescribed
            .clone()
            .try_map_canonical_storage(|mut values| {
                values[i] = Some(crate::excitation::ProjectionScalar {
                    real: x * x,
                    imaginary: x,
                });
                values
            })
            .unwrap();
    }
    let result = super::response::solve(
        assembly,
        1.0 / std::f64::consts::TAU,
        AcousticHarmonicPolicy::default(),
        1e-8,
    )
    .unwrap();
    // v=(-1/3 + 2ix/3,0,0), I=(x²/6,0,0). Unit tetrahedral moments
    // E[x]=1/4, E[x²]=1/10, E[x⁴]=1/35 give I_x=1/60, E=401/3360.
    // Multiplying averaged p and v would incorrectly return I_x=1/240.
    let intensity = result.element_mean_intensity_w_per_m2.canonical_values()[0];
    assert!((intensity[0] - 1.0 / 60.0).abs() < 1e-13);
    assert!(intensity[1].abs() < 1e-13 && intensity[2].abs() < 1e-13);
    assert!(
        (result
            .element_mean_energy_density_j_per_m3
            .canonical_values()[0]
            - 401.0 / 3360.0)
            .abs()
            < 1e-13
    );
}

#[test]
fn boundaries_fail_closed_on_missing_targets_and_overlap() {
    let mesh = slab_mesh(2);
    let mut model = model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "wall".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("not_a_face")),
        operator: BoundaryOperator::AcousticRigidWall,
    });
    let solve = |model: &AnalysisModel| {
        solve_canonical_harmonic(
            model,
            &mesh,
            "acoustic",
            Frequency::new(0.2),
            AcousticHarmonicPolicy::default(),
            1e-8,
        )
    };
    assert!(matches!(
        solve(&model),
        Err(AcousticHarmonicError::Boundary(_))
    ));
    model.boundaries[0].target =
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right"));
    model.boundaries.push(model.boundaries[0].clone());
    assert!(matches!(
        solve(&model),
        Err(AcousticHarmonicError::Boundary(_))
    ));
    model.boundaries.pop();
    model.boundaries[0].operator = BoundaryOperator::AcousticRadiation;
    assert!(solve(&model).is_ok());
}

#[test]
fn pressure_velocity_overlap_is_rejected_in_both_authored_orders() {
    let mesh = slab_mesh(2);
    let mut model = model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    velocity_source(&mut model, -1.0, 0.0);
    let mut signal = complex_signal(Pressure::new(1.0), Pressure::new(0.0));
    signal.signal_id = SignalId("pressure_drive".into());
    model.signals.push(PhysicalSignal::Pressure(signal));
    let mut action = model.excitations[0].excitation().clone();
    action.excitation_id = ExcitationId("pressure".into());
    action.signal_id = SignalId("pressure_drive".into());
    model
        .excitations
        .push(ExcitationAction::AcousticPressure(action));
    model.excitation_cases[0]
        .action_ids
        .push(ExcitationId("pressure".into()));
    for _ in 0..2 {
        assert!(matches!(
            solve_canonical_harmonic(
                &model,
                &mesh,
                "acoustic",
                Frequency::new(0.2),
                AcousticHarmonicPolicy::default(),
                1e-8,
            ),
            Err(AcousticHarmonicError::Boundary(_))
        ));
        model.excitations.reverse();
    }
}
mod periodic;

// witness: FEA-ACOUSTIC-BOUNDARY-002/complex_frequency_dependent_impedance @
//   acoustic_harmonic/tet4/numerical_kernel; coverage: partial; check: impedance-termination; boundary: a tabulated complex
//   termination converges to the two-wave duct solution; Delany-Bazley is checked in core only.
#[test]
fn complex_impedance_termination_converges_to_the_two_wave_duct_solution() {
    // p(0) = 1, p(L) = Z v_n(L), iωρ v = −∂p/∂x: p = A e^{−ikx} + B e^{ikx}.
    let (rho, c, f) = (3.0, 2.0, 0.2);
    let omega = std::f64::consts::TAU * f;
    let k = omega / c;
    let (r, x) = (rho * c * 0.6, -rho * c * 0.8);
    let z = C(r, x);
    let e = |phase: f64| C(phase.cos(), phase.sin());
    // At x = 1: p = A e⁻ + B e⁺, p' = −ik A e⁻ + ik B e⁺, and p + Z p'/(iωρ) = 0.
    // With A + B = 1: A (e⁻ − (Z k/(ωρ)) e⁻) + B (e⁺ + (Z k/(ωρ)) e⁺) = 0.
    let zk = z.mul(C(k / (omega * rho), 0.0));
    let (em, ep) = (e(-k), e(k));
    let a_coefficient = em.mul(C(1.0 - zk.0, -zk.1));
    let b_coefficient = ep.mul(C(1.0 + zk.0, zk.1));
    // A = b/(b − a), B = −a/(b − a)
    let denominator = C(
        b_coefficient.0 - a_coefficient.0,
        b_coefficient.1 - a_coefficient.1,
    );
    let a = b_coefficient.div(denominator);
    let b = C(-a_coefficient.0, -a_coefficient.1).div(denominator);
    let exact = |position: f64| {
        let left = a.mul(e(-k * position));
        let right = b.mul(e(k * position));
        C(left.0 + right.0, left.1 + right.1)
    };
    let mut errors = Vec::new();
    for cells in [2, 4, 8] {
        let mesh = slab_mesh(cells);
        let mut model = model();
        model.material_assignments[0].target =
            PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
        prescribe(
            &mut model,
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
            1.0,
        );
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "lined_end".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticImpedanceLaw {
                law: runmat_analysis_core::AcousticImpedanceLaw::Table {
                    frequencies_hz: vec![0.1, 0.3],
                    resistance_pa_s_per_m: vec![r, r],
                    reactance_pa_s_per_m: vec![x, x],
                },
            },
        });
        let result = solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(f),
            AcousticHarmonicPolicy {
                relative_tolerance: 1e-11,
                ..Default::default()
            },
            1e-8,
        )
        .unwrap();
        let error = (result
            .projection
            .nodes
            .iter()
            .zip(result.pressure_pa.canonical_values())
            .map(|(node, p)| {
                let exact = exact(node.coordinates_m.canonical_value()[0]);
                (p.real - exact.0).powi(2) + (p.imaginary - exact.1).powi(2)
            })
            .sum::<f64>()
            / result.pressure_pa.canonical_values().len() as f64)
            .sqrt();
        errors.push(error);
        assert!(result.mean_impedance_loss_w > runmat_types::physical::Quantity::new(0.0));
        assert!(result.power_relative_error < 1e-9);
    }
    assert!(
        errors.windows(2).all(|pair| pair[0] / pair[1] > 3.0),
        "{errors:?}"
    );
    assert!(errors[2] < 0.004, "{errors:?}");
}
