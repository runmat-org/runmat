use runmat_types::physical::Quantity;
mod rejection;

use super::harness::with_harness_provider;
use super::manifest::default_options;
use super::*;

fn fixed_constraint(
    id: impl Into<String>,
    region: impl Into<String>,
) -> runmat_analysis_core::AnalysisConstraint {
    runmat_analysis_core::AnalysisConstraint::Structural(
        runmat_analysis_core::StructuralConstraint {
            constraint_id: id.into(),
            frame: runmat_analysis_core::CoordinateFrame::Global,
            kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                target: runmat_analysis_core::RegionTarget::authored(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(
                        &Into::<String>::into(region),
                    ),
                ),
            },
        },
    )
}

fn zero_translation_constraint(
    id: impl Into<String>,
    region: impl Into<String>,
) -> runmat_analysis_core::AnalysisConstraint {
    runmat_analysis_core::AnalysisConstraint::Structural(
        runmat_analysis_core::StructuralConstraint {
            constraint_id: id.into(),
            frame: runmat_analysis_core::CoordinateFrame::Global,
            kind: runmat_analysis_core::StructuralConstraintKind::Prescribed(
                runmat_analysis_core::PrescribedStructuralDofs {
                    target: runmat_analysis_core::RegionTarget::authored(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &Into::<String>::into(region),
                        ),
                    ),
                    translations: [
                        runmat_analysis_core::StructuralTranslationDof::Ux,
                        runmat_analysis_core::StructuralTranslationDof::Uy,
                        runmat_analysis_core::StructuralTranslationDof::Uz,
                    ]
                    .into_iter()
                    .map(|dof| runmat_analysis_core::TranslationDofValue {
                        dof,
                        value: runmat_analysis_core::PrescribedValue::Constant(
                            runmat_analysis_core::Displacement::new(0.0),
                        ),
                    })
                    .collect(),
                    rotations: Vec::new(),
                },
            ),
        },
    )
}

fn magnetic_ground(
    id: impl Into<String>,
    region: impl Into<String>,
) -> runmat_analysis_core::AnalysisConstraint {
    runmat_analysis_core::AnalysisConstraint::Magnetic(runmat_analysis_core::MagneticConstraint {
        constraint_id: id.into(),
        kind: runmat_analysis_core::MagneticConstraintKind::VectorPotentialGround {
            target: runmat_analysis_core::RegionTarget::authored(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&Into::<String>::into(
                    region,
                )),
            ),
        },
    })
}

fn magnetic_insulation(
    id: impl Into<String>,
    region: impl Into<String>,
) -> runmat_analysis_core::AnalysisBoundary {
    runmat_analysis_core::AnalysisBoundary {
        boundary_id: id.into(),
        target: runmat_analysis_core::RegionTarget::authored(
            runmat_meshing_core::fixtures::fixture_boundary_selection(&Into::<String>::into(
                region,
            )),
        ),
        operator: runmat_analysis_core::BoundaryOperator::MagneticInsulation,
    }
}

fn acoustic_boundary(
    id: impl Into<String>,
    region: impl Into<String>,
    operator: runmat_analysis_core::BoundaryOperator,
) -> runmat_analysis_core::AnalysisBoundary {
    runmat_analysis_core::AnalysisBoundary {
        boundary_id: id.into(),
        target: runmat_analysis_core::RegionTarget::authored(
            runmat_meshing_core::fixtures::fixture_boundary_selection(&Into::<String>::into(
                region,
            )),
        ),
        operator,
    }
}

fn thermal_temperature(
    id: impl Into<String>,
    region: impl Into<String>,
    kelvin: f64,
) -> runmat_analysis_core::AnalysisConstraint {
    runmat_analysis_core::AnalysisConstraint::Thermal(runmat_analysis_core::ThermalConstraint {
        constraint_id: id.into(),
        kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
            target: runmat_analysis_core::RegionTarget::authored(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&Into::<String>::into(
                    region,
                )),
            ),
            value: runmat_analysis_core::PrescribedValue::Constant(
                runmat_analysis_core::Temperature::new(kelvin),
            ),
        },
    })
}

fn thermal_boundary(
    id: impl Into<String>,
    region: impl Into<String>,
    operator: runmat_analysis_core::BoundaryOperator,
) -> runmat_analysis_core::AnalysisBoundary {
    runmat_analysis_core::AnalysisBoundary {
        boundary_id: id.into(),
        target: runmat_analysis_core::RegionTarget::authored(
            runmat_meshing_core::fixtures::fixture_boundary_selection(&Into::<String>::into(
                region,
            )),
        ),
        operator,
    }
}

enum FluidFixtureConstraint {
    Velocity([f64; 3]),
    PressureGauge(f64),
    NoSlipWall,
}

fn fluid_constraint(
    id: impl Into<String>,
    region: impl Into<String>,
    condition: FluidFixtureConstraint,
) -> runmat_analysis_core::AnalysisConstraint {
    let target = runmat_analysis_core::RegionTarget::authored(
        runmat_meshing_core::fixtures::fixture_boundary_selection(&Into::<String>::into(region)),
    );
    let kind = match condition {
        FluidFixtureConstraint::Velocity(components) => {
            runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
                target,
                value: runmat_analysis_core::PrescribedValue::Constant(
                    components.map(runmat_analysis_core::Velocity::new),
                ),
            }
        }
        FluidFixtureConstraint::PressureGauge(value) => {
            runmat_analysis_core::FluidConstraintKind::PressureReference {
                target,
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Pressure::new(value),
                ),
            }
        }
        FluidFixtureConstraint::NoSlipWall => {
            runmat_analysis_core::FluidConstraintKind::NoSlipWall { target }
        }
    };
    runmat_analysis_core::AnalysisConstraint::Fluid(runmat_analysis_core::FluidConstraint {
        constraint_id: id.into(),
        kind,
    })
}
use runmat_analysis_core::AnalysisField;
use runmat_analysis_fea::{
    fea_buckling_mode_shape_field_id, fea_thermal_temperature_field_id,
    FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE, FEA_FIELD_CFD_VELOCITY,
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
    FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE, FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
    FEA_FIELD_STRUCTURAL_SHELL_VON_MISES, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use runmat_runtime::analysis::{AnalysisRunResult, ContactEnforcementOptions};
use sha2::{Digest, Sha256};

enum FixtureExcitationQuantity {
    Force([f64; 3]),
    Moment([f64; 3]),
    Pressure(f64),
    AcousticPressure(f64),
    HeatGeneration(f64),
    ElectricCurrent {
        amperes: f64,
        phase_radians: runmat_types::physical::Angle,
        amplitude_scale: f64,
    },
    CurrentDensity {
        amperes_per_square_meter: [f64; 3],
        phase_radians: runmat_types::physical::Angle,
        amplitude_scale: f64,
    },
}

struct FixtureExcitation {
    excitation_id: String,
    region_id: String,
    quantity: FixtureExcitationQuantity,
}

fn replace_fixture_excitations(model: &mut AnalysisModel, specs: Vec<FixtureExcitation>) {
    let replaced_signal_ids = model
        .excitations
        .iter()
        .map(|action| action.excitation().signal_id.clone())
        .collect::<std::collections::HashSet<_>>();
    model
        .signals
        .retain(|signal| !replaced_signal_ids.contains(signal.signal_id()));
    model.excitations.clear();
    model.excitation_cases.clear();
    append_fixture_excitations(model, specs);
}

fn append_fixture_excitations(model: &mut AnalysisModel, specs: Vec<FixtureExcitation>) {
    use runmat_analysis_core::{
        AnalysisExcitation, CoordinateFrame, ExcitationAction, ExcitationCase,
        ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution, ExcitationId,
        ExcitationKinematics, ExcitationTarget, PhysicalSignal, ProjectionRequirement, Signal,
        SignalApproximationEvidence, SignalId, SignalPrecision, SignalProvenance,
        SignalRepresentation, SignalShape, SignalValue,
    };

    let case_id = ExcitationCaseId(format!("fixture_case_{}", model.excitation_cases.len()));
    let first_action = model.excitations.len();

    for spec in specs {
        let excitation_id = ExcitationId(spec.excitation_id);
        let signal_id = SignalId(format!("{}_signal", excitation_id.0));
        let (signal, action, distribution, volume_target, preserve_moment): (
            PhysicalSignal,
            fn(AnalysisExcitation) -> ExcitationAction,
            ExcitationDistribution,
            bool,
            bool,
        ) = match spec.quantity {
            FixtureExcitationQuantity::Force(value) => (
                PhysicalSignal::Force(fixture_signal(
                    signal_id.clone(),
                    SignalShape::Vector3,
                    SignalPrecision::Real64,
                    SignalRepresentation::Constant(SignalValue::Vector3(
                        value.map(runmat_analysis_core::Force::new),
                    )),
                )),
                ExcitationAction::Force as fn(AnalysisExcitation) -> ExcitationAction,
                ExcitationDistribution::PreserveResultant,
                false,
                false,
            ),
            FixtureExcitationQuantity::Moment(value) => (
                PhysicalSignal::Moment(fixture_signal(
                    signal_id.clone(),
                    SignalShape::Vector3,
                    SignalPrecision::Real64,
                    SignalRepresentation::Constant(SignalValue::Vector3(
                        value.map(runmat_analysis_core::Moment::new),
                    )),
                )),
                ExcitationAction::Moment,
                ExcitationDistribution::PreserveResultant,
                false,
                true,
            ),
            FixtureExcitationQuantity::Pressure(value) => (
                PhysicalSignal::Pressure(fixture_signal(
                    signal_id.clone(),
                    SignalShape::Scalar,
                    SignalPrecision::Real64,
                    SignalRepresentation::Constant(SignalValue::Scalar(
                        runmat_analysis_core::Pressure::new(value),
                    )),
                )),
                ExcitationAction::Pressure,
                ExcitationDistribution::IntensityPerArea,
                false,
                false,
            ),
            FixtureExcitationQuantity::AcousticPressure(value) => (
                PhysicalSignal::Pressure(fixture_signal(
                    signal_id.clone(),
                    SignalShape::Scalar,
                    SignalPrecision::Real64,
                    SignalRepresentation::Constant(SignalValue::Scalar(
                        runmat_analysis_core::Pressure::new(value),
                    )),
                )),
                ExcitationAction::AcousticPressure,
                ExcitationDistribution::IntensityPerArea,
                false,
                false,
            ),
            FixtureExcitationQuantity::HeatGeneration(value) => (
                PhysicalSignal::HeatGenerationDensity(fixture_signal(
                    signal_id.clone(),
                    SignalShape::Scalar,
                    SignalPrecision::Real64,
                    SignalRepresentation::Constant(SignalValue::Scalar(
                        runmat_analysis_core::HeatGenerationDensity::new(value),
                    )),
                )),
                ExcitationAction::HeatGeneration,
                ExcitationDistribution::IntensityPerVolume,
                true,
                false,
            ),
            FixtureExcitationQuantity::ElectricCurrent {
                amperes,
                phase_radians,
                amplitude_scale,
            } => {
                let amplitude = amperes * amplitude_scale;
                let real = runmat_analysis_core::ElectricCurrent::new(
                    amplitude * phase_radians.canonical_value().cos(),
                );
                let imaginary = runmat_analysis_core::ElectricCurrent::new(
                    amplitude * phase_radians.canonical_value().sin(),
                );
                (
                    PhysicalSignal::ElectricCurrent(fixture_signal(
                        signal_id.clone(),
                        SignalShape::Scalar,
                        SignalPrecision::Complex128,
                        SignalRepresentation::Tabulated(runmat_analysis_core::TabulatedSignal {
                            domain: runmat_analysis_core::TabulatedSignalDomain::FrequencyHertz,
                            knots: vec![runmat_analysis_core::SignalKnot::continuous(
                                1.0,
                                runmat_analysis_core::SignalOrdinate::Complex {
                                    real: SignalValue::Scalar(real),
                                    imaginary: SignalValue::Scalar(imaginary),
                                },
                            )],
                            interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                            extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
                        }),
                    )),
                    ExcitationAction::ElectricCurrent,
                    ExcitationDistribution::Concentrated,
                    false,
                    false,
                )
            }
            FixtureExcitationQuantity::CurrentDensity {
                amperes_per_square_meter,
                phase_radians,
                amplitude_scale,
            } => {
                let real = amperes_per_square_meter.map(|value| {
                    runmat_analysis_core::ElectricCurrentDensity::new(
                        value * amplitude_scale * phase_radians.canonical_value().cos(),
                    )
                });
                let imaginary = amperes_per_square_meter.map(|value| {
                    runmat_analysis_core::ElectricCurrentDensity::new(
                        value * amplitude_scale * phase_radians.canonical_value().sin(),
                    )
                });
                (
                    PhysicalSignal::ElectricCurrentDensity(fixture_signal(
                        signal_id.clone(),
                        SignalShape::Vector3,
                        SignalPrecision::Complex128,
                        SignalRepresentation::Tabulated(runmat_analysis_core::TabulatedSignal {
                            domain: runmat_analysis_core::TabulatedSignalDomain::FrequencyHertz,
                            knots: vec![runmat_analysis_core::SignalKnot::continuous(
                                1.0,
                                runmat_analysis_core::SignalOrdinate::Complex {
                                    real: SignalValue::Vector3(real),
                                    imaginary: SignalValue::Vector3(imaginary),
                                },
                            )],
                            interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                            extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
                        }),
                    )),
                    ExcitationAction::CurrentDensity,
                    ExcitationDistribution::TopologyField,
                    false,
                    false,
                )
            }
        };
        let target = if volume_target {
            runmat_analysis_core::RegionTarget::volume(spec.region_id)
        } else {
            runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&spec.region_id),
            )
        };
        model.signals.push(signal);
        model.excitations.push(action(AnalysisExcitation {
            excitation_id: excitation_id.clone(),
            target: ExcitationTarget::Region(target),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id,
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: distribution == ExcitationDistribution::PreserveResultant,
                preserve_moment,
                relative_tolerance: 1.0e-10,
            },
        }));
    }
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids: model
            .excitations
            .iter()
            .skip(first_action)
            .map(|action| action.excitation().excitation_id.clone())
            .collect(),
        activation: ExcitationCaseActivation::Always,
    });

    fn fixture_signal<Q>(
        signal_id: SignalId,
        shape: SignalShape,
        precision: SignalPrecision,
        representation: SignalRepresentation<Q>,
    ) -> Signal<Q> {
        let frame = if shape == SignalShape::Scalar {
            None
        } else {
            Some(CoordinateFrame::Global)
        };
        Signal {
            signal_id,
            shape,
            frame,
            representation,
            precision,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        }
    }
}

fn set_temperature_history(
    model: &mut AnalysisModel,
    signal_id: &str,
    reference_kelvin: f64,
    delta_kelvin: f64,
    points: &[(f64, f64)],
) {
    set_temperature_excitation(
        model,
        signal_id,
        reference_kelvin,
        delta_kelvin,
        points,
        runmat_analysis_core::ExcitationTarget::WholeModel,
    );
}

fn set_temperature_excitation(
    model: &mut AnalysisModel,
    signal_id: &str,
    reference_kelvin: f64,
    delta_kelvin: f64,
    points: &[(f64, f64)],
    target: runmat_analysis_core::ExcitationTarget,
) {
    let id = runmat_analysis_core::SignalId(signal_id.to_string());
    model.signals.retain(|signal| signal.signal_id() != &id);
    model
        .signals
        .push(runmat_analysis_core::PhysicalSignal::Temperature(
            scalar_signal(id.clone(), points, |scale| {
                runmat_analysis_core::Temperature::new(reference_kelvin + delta_kelvin * scale)
            }),
        ));
    append_signal_excitation(
        model,
        signal_id,
        id,
        target,
        None,
        runmat_analysis_core::ExcitationDistribution::TopologyField,
        runmat_analysis_core::ExcitationAction::TemperatureField,
    );
}

fn set_electric_potential_history(
    model: &mut AnalysisModel,
    signal_id: &str,
    applied_voltage: f64,
    points: &[(f64, f64)],
) {
    let id = runmat_analysis_core::SignalId(signal_id.to_string());
    model.signals.retain(|signal| signal.signal_id() != &id);
    model
        .signals
        .push(runmat_analysis_core::PhysicalSignal::ElectricPotential(
            scalar_signal(id.clone(), points, |scale| {
                runmat_analysis_core::ElectricPotential::new(applied_voltage * scale)
            }),
        ));
    let target = model
        .material_assignments
        .first()
        .map(|assignment| {
            runmat_analysis_core::ExcitationTarget::Region(
                runmat_analysis_core::RegionTarget::boundary(
                    assignment
                        .selected_region_id()
                        .expect("fixture uses selected material region"),
                ),
            )
        })
        .unwrap_or_else(|| {
            runmat_analysis_core::ExcitationTarget::Region(
                runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("terminal"),
                ),
            )
        });
    append_signal_excitation(
        model,
        signal_id,
        id,
        target,
        None,
        runmat_analysis_core::ExcitationDistribution::TopologyField,
        runmat_analysis_core::ExcitationAction::TerminalVoltage,
    );
}

fn set_velocity_history(
    model: &mut AnalysisModel,
    signal_id: &str,
    inlet_velocity: f64,
    points: &[(f64, f64)],
) {
    let id = runmat_analysis_core::SignalId(signal_id.to_string());
    model.signals.retain(|signal| signal.signal_id() != &id);
    model
        .signals
        .push(runmat_analysis_core::PhysicalSignal::Velocity(
            scalar_signal(id.clone(), points, |scale| {
                runmat_analysis_core::Velocity::new(inlet_velocity * scale)
            }),
        ));
    let target = model
        .constraints
        .iter()
        .find_map(|constraint| match constraint {
            runmat_analysis_core::AnalysisConstraint::Fluid(fluid) => match &fluid.kind {
                runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
                    target, ..
                } => Some(target.clone()),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or_else(|| {
            runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("inlet"),
            )
        });
    append_signal_excitation(
        model,
        signal_id,
        id,
        runmat_analysis_core::ExcitationTarget::Region(target),
        Some(runmat_analysis_core::Direction {
            vector: [1.0, 0.0, 0.0],
        }),
        runmat_analysis_core::ExcitationDistribution::TopologyField,
        runmat_analysis_core::ExcitationAction::InletVelocity,
    );
}

fn fixture_cfd_domain(
    model: &mut AnalysisModel,
    signal_id: &str,
    inlet_velocity_m_per_s: f64,
    dynamic_viscosity_pa_s: f64,
    turbulence_intensity: f64,
    points: &[(f64, f64)],
) -> runmat_analysis_core::CfdDomain {
    set_velocity_history(model, signal_id, inlet_velocity_m_per_s, points);
    runmat_analysis_core::CfdDomain {
        density: runmat_analysis_core::MassDensity::new(1.225),
        dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(dynamic_viscosity_pa_s),
        turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(turbulence_intensity),
        rheology: None,
        porous_zones: Vec::new(),
    }
}

fn scalar_signal<Q>(
    signal_id: runmat_analysis_core::SignalId,
    points: &[(f64, f64)],
    value: impl Fn(f64) -> Q,
) -> runmat_analysis_core::Signal<Q> {
    let representation = if points.is_empty() {
        runmat_analysis_core::SignalRepresentation::Constant(
            runmat_analysis_core::SignalValue::Scalar(value(1.0)),
        )
    } else {
        runmat_analysis_core::SignalRepresentation::Tabulated(
            runmat_analysis_core::TabulatedSignal {
                domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
                knots: points
                    .iter()
                    .map(|point| {
                        runmat_analysis_core::SignalKnot::continuous(
                            point.0,
                            runmat_analysis_core::SignalOrdinate::Real(
                                runmat_analysis_core::SignalValue::Scalar(value(point.1)),
                            ),
                        )
                    })
                    .collect(),
                interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
            },
        )
    };
    runmat_analysis_core::Signal {
        signal_id,
        shape: runmat_analysis_core::SignalShape::Scalar,
        frame: None,
        representation,
        precision: runmat_analysis_core::SignalPrecision::Real64,
        evidence: runmat_analysis_core::SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: points.len().max(1),
        },
        provenance: runmat_analysis_core::SignalProvenance::Authored,
    }
}

fn append_signal_excitation(
    model: &mut AnalysisModel,
    id: &str,
    signal_id: runmat_analysis_core::SignalId,
    target: runmat_analysis_core::ExcitationTarget,
    direction: Option<runmat_analysis_core::Direction>,
    distribution: runmat_analysis_core::ExcitationDistribution,
    action: fn(runmat_analysis_core::AnalysisExcitation) -> runmat_analysis_core::ExcitationAction,
) {
    let excitation_id = runmat_analysis_core::ExcitationId(format!("{id}.excitation"));
    let case_id = runmat_analysis_core::ExcitationCaseId(format!("{id}.case"));
    model
        .excitations
        .retain(|existing| existing.excitation().excitation_id != excitation_id);
    model
        .excitation_cases
        .retain(|existing| existing.case_id != case_id);
    model
        .excitations
        .push(action(runmat_analysis_core::AnalysisExcitation {
            excitation_id: excitation_id.clone(),
            target,
            frame: runmat_analysis_core::CoordinateFrame::Global,
            direction,
            distribution,
            kinematics: runmat_analysis_core::ExcitationKinematics::FixedFrame,
            signal_id,
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: runmat_analysis_core::ProjectionRequirement {
                preserve_resultant: false,
                preserve_moment: false,
                relative_tolerance: 1.0e-10,
            },
        }));
    model
        .excitation_cases
        .push(runmat_analysis_core::ExcitationCase {
            case_id,
            action_ids: vec![excitation_id],
            activation: runmat_analysis_core::ExcitationCaseActivation::Always,
        });
}

fn primary_result_field_id(procedure_step: &ProcedureStep) -> String {
    match AnalysisSolverOptions::defaults_for(procedure_step)
        .expect("fixture procedure step must select a supported solver")
    {
        AnalysisSolverOptions::ThermalDiffusion(_) => fea_thermal_temperature_field_id(0),
        AnalysisSolverOptions::AcousticHelmholtz(_) => {
            FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE.to_string()
        }
        AnalysisSolverOptions::AcousticResponse(_) => {
            panic!("acoustic modal/transient fixtures execute through the study runner")
        }
        AnalysisSolverOptions::ElectricalStatic(_)
        | AnalysisSolverOptions::Magnetostatic(_)
        | AnalysisSolverOptions::MaxwellHarmonic(_)
        | AnalysisSolverOptions::LineParameters(_) => {
            panic!(
                "typed electromagnetic completions have no synthetic scalar-harness primary field"
            )
        }
        AnalysisSolverOptions::IncompressibleFlow(_) => FEA_FIELD_CFD_VELOCITY.to_string(),
        AnalysisSolverOptions::StructuralLinear(_)
        | AnalysisSolverOptions::StructuralModes(_)
        | AnalysisSolverOptions::StructuralFrequency(_)
        | AnalysisSolverOptions::StructuralTransient(_)
        | AnalysisSolverOptions::StructuralNonlinear(_) => {
            FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string()
        }
        AnalysisSolverOptions::StructuralResponseSpectrum(_) => {
            panic!("response-spectrum envelopes do not have one ordinary primary field")
        }
        AnalysisSolverOptions::StructuralRandomResponse(_) => {
            panic!("random-response statistics do not have one ordinary primary field")
        }
        AnalysisSolverOptions::StructuralBuckling(_) => fea_buckling_mode_shape_field_id(1),
    }
}

fn analysis_result_field<'a>(
    run_result: &'a AnalysisRunResult,
    field_id: &str,
) -> Option<&'a AnalysisField> {
    run_result
        .run
        .field(field_id)
        .or_else(|| {
            run_result.modal_results.as_ref().and_then(|modal| {
                modal
                    .mode_shapes
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result.thermal_results.as_ref().and_then(|thermal| {
                thermal
                    .temperature_snapshots
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result.transient_results.as_ref().and_then(|transient| {
                transient
                    .displacement_snapshots
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result.nonlinear_results.as_ref().and_then(|nonlinear| {
                nonlinear
                    .displacement_snapshots
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn nonlinear_options_for_spec(
    spec: &FixtureSpec,
    solver_mesh_artifact_path: Option<String>,
) -> AnalysisNonlinearRunOptions {
    let mut options = AnalysisNonlinearRunOptions::solid_recommended();
    options.solver_mesh_artifact_path = solver_mesh_artifact_path;
    options.contact_enforcement = contact_for_fixture(spec.id).map(|contact| contact.enforcement);
    options.increment_count = spec.transient_step_count.unwrap_or(options.increment_count);

    if let Some(value) = env_usize("RUNMAT_NONLINEAR_INCREMENT_COUNT") {
        options.increment_count = value.max(1);
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_NEWTON_ITERS") {
        options.max_newton_iters = value.max(1);
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_TOLERANCE") {
        if value.is_finite() && value > 0.0 {
            options.tolerance = value;
        }
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_LINEAR_TOLERANCE") {
        if value.is_finite() && value > 0.0 {
            options.linear_tolerance = value;
        }
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_MINIMUM_LOAD_INCREMENT") {
        if value.is_finite() && value > 0.0 {
            options.minimum_load_increment = value;
        }
    }
    if let Some(value) = env_bool("RUNMAT_NONLINEAR_LINE_SEARCH") {
        options.line_search = value;
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_BACKTRACKS") {
        options.max_line_search_backtracks = value;
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_LINE_SEARCH_REDUCTION") {
        if value.is_finite() && value > 0.0 && value < 1.0 {
            options.line_search_reduction = value;
        }
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_LOAD_CUTBACKS") {
        options.max_load_cutbacks = value;
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_ACCEPTED_INCREMENTS") {
        options.max_accepted_increments = value.max(1);
    }

    options
}

fn plasticity_for_fixture(spec_id: &str) -> Option<runmat_analysis_core::MaterialPlasticModel> {
    use runmat_analysis_core::MaterialPlasticModel::J2IsotropicLinearHardening;
    match spec_id {
        "nonlinear_plasticity_benchmark_gpu_provider" => Some(J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(250e6),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(1e9),
        }),
        "nonlinear_invalid_plasticity_options" => Some(J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(-1.0),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(1e9),
        }),
        "nonlinear_plastic_hardening_reference_gpu_provider" => Some(J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(300e6),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(2e9),
        }),
        "nonlinear_plastic_hardening_reference_complex_gpu_provider" => {
            Some(J2IsotropicLinearHardening {
                yield_stress_pa: runmat_types::physical::YieldStress::new(350e6),
                isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(3e9),
            })
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct FixtureContactOptions {
    enforcement: ContactEnforcementOptions,
    friction_coefficient: f64,
}

fn contact_for_fixture(spec_id: &str) -> Option<FixtureContactOptions> {
    match spec_id {
        "nonlinear_contact_benchmark_gpu_provider" => Some(FixtureContactOptions {
            enforcement: ContactEnforcementOptions {
                penalty_stiffness_scale: 0.15,
                max_penetration_ratio: 0.035,
            },
            friction_coefficient: 0.9,
        }),
        "nonlinear_invalid_contact_options" => Some(FixtureContactOptions {
            enforcement: ContactEnforcementOptions {
                penalty_stiffness_scale: 0.0,
                max_penetration_ratio: 0.01,
            },
            friction_coefficient: 0.0,
        }),
        "nonlinear_contact_frictionless_reference_gpu_provider" => Some(FixtureContactOptions {
            enforcement: ContactEnforcementOptions {
                penalty_stiffness_scale: 2.0,
                max_penetration_ratio: 0.01,
            },
            friction_coefficient: 0.0,
        }),
        "nonlinear_contact_frictionless_reference_complex_gpu_provider" => {
            Some(FixtureContactOptions {
                enforcement: ContactEnforcementOptions {
                    penalty_stiffness_scale: 1.2,
                    max_penetration_ratio: 0.014,
                },
                friction_coefficient: 0.0,
            })
        }
        _ => None,
    }
}

struct FixtureThermoRegionTemperatureDelta {
    region_id: String,
    temperature_delta_k: f64,
}

struct FixtureThermoMechanicalCoupling {
    reference_temperature_k: runmat_types::physical::Temperature,
    applied_temperature_delta_k: runmat_types::physical::TemperatureDifference,
    thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion,
    region_temperature_deltas: Vec<FixtureThermoRegionTemperatureDelta>,
}

struct FixtureElectroThermalCoupling {
    reference_temperature_k: runmat_types::physical::Temperature,
    applied_voltage_v: f64,
    base_electrical_conductivity_s_per_m: f64,
    resistive_heating_coefficient: runmat_types::physical::InverseTemperature,
}

fn thermo_coupling_for_fixture(spec_id: &str) -> Option<FixtureThermoMechanicalCoupling> {
    match spec_id {
        "thermo_mech_kickoff_rejected" => Some(FixtureThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(65.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: vec![
                FixtureThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 75.0,
                },
                FixtureThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 55.0,
                },
            ],
        }),
        "thermo_gradient_benign_rejected" => Some(FixtureThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(55.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.0e-5),
            region_temperature_deltas: vec![
                FixtureThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 60.0,
                },
                FixtureThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 50.0,
                },
            ],
        }),
        "thermo_gradient_pathological_rejected" => Some(FixtureThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(220.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(2.5e-5),
            region_temperature_deltas: vec![
                FixtureThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 260.0,
                },
                FixtureThermoRegionTemperatureDelta {
                    region_id: "polymer_segment".to_string(),
                    temperature_delta_k: 120.0,
                },
            ],
        }),
        "thermo_ramp_smooth_rejected" => Some(FixtureThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(70.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.1e-5),
            region_temperature_deltas: vec![
                FixtureThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 72.0,
                },
                FixtureThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 68.0,
                },
            ],
        }),
        "thermo_shock_oscillatory_rejected" => Some(FixtureThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(140.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(2.0e-5),
            region_temperature_deltas: vec![
                FixtureThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 210.0,
                },
                FixtureThermoRegionTemperatureDelta {
                    region_id: "polymer_segment".to_string(),
                    temperature_delta_k: 90.0,
                },
            ],
        }),
        _ => None,
    }
}

fn thermo_history_for_fixture(spec_id: &str) -> &'static [(f64, f64)] {
    match spec_id {
        "thermo_mech_kickoff_rejected" => &[(0.0, 0.6), (1.0, 1.0)],
        "thermo_gradient_benign_rejected" => &[(0.0, 0.7), (1.0, 1.0)],
        "thermo_gradient_pathological_rejected" => &[(0.0, 0.2), (0.4, 1.3), (1.0, 0.9)],
        value
            if value == "thermo_ramp_smooth_rejected"
                || value.starts_with("thermal_standalone_ramp_") =>
        {
            &[(0.0, 0.3), (0.5, 0.7), (1.0, 1.0)]
        }
        "thermo_shock_oscillatory_rejected" => {
            &[(0.0, 0.4), (0.25, 1.4), (0.5, 0.5), (0.75, 1.3), (1.0, 0.6)]
        }
        _ => &[],
    }
}

fn electro_coupling_for_fixture(spec_id: &str) -> Option<FixtureElectroThermalCoupling> {
    match spec_id {
        "electro_thermal_joule_benign_rejected" => Some(FixtureElectroThermalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(3.5e-4),
        }),
        "electro_thermal_joule_pathological_rejected" => Some(FixtureElectroThermalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_voltage_v: 180.0,
            base_electrical_conductivity_s_per_m: 2.0e7,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(9.5e-4),
        }),
        "electro_thermal_invalid_voltage" => Some(FixtureElectroThermalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_voltage_v: f64::NAN,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(3.5e-4),
        }),
        "electro_thermal_invalid_conductivity" => Some(FixtureElectroThermalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 0.0,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(3.5e-4),
        }),
        _ => None,
    }
}

fn electro_history_for_fixture(spec_id: &str) -> &'static [(f64, f64)] {
    match spec_id {
        "electro_thermal_joule_benign_rejected" => &[(0.0, 0.7), (1.0, 1.0)],
        "electro_thermal_joule_pathological_rejected" => {
            &[(0.0, 0.45), (0.3, 1.45), (0.6, 0.6), (1.0, 1.25)]
        }
        _ => &[],
    }
}

fn thermal_options_for_spec(
    spec: &FixtureSpec,
    solver_mesh_artifact_path: Option<String>,
) -> AnalysisThermalRunOptions {
    AnalysisThermalRunOptions {
        snapshot_count: spec
            .transient_step_count
            .unwrap_or(AnalysisThermalRunOptions::default().snapshot_count),
        solver_mesh_artifact_path,
        ..AnalysisThermalRunOptions::default()
    }
}

fn transient_options_for_spec(
    spec: &FixtureSpec,
    solver_mesh_artifact_path: Option<String>,
) -> AnalysisTransientRunOptions {
    AnalysisTransientRunOptions {
        time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed {
            interval_count: spec.transient_step_count.unwrap_or(
                AnalysisTransientRunOptions::default()
                    .time_step_policy
                    .interval_count(),
            ),
        },
        solver_mesh_artifact_path,
        ..AnalysisTransientRunOptions::solid_recommended()
    }
}

#[derive(Debug, Clone, Copy)]
enum ElectromagneticFixtureKind {
    Homogeneous,
    Heterogeneous,
    SparseAssignments,
    MultiRegionAssignments,
    OverlapInterference,
    BoundaryKernel,
    BoundaryPenaltyStress,
    MultiRegionPhasedSource,
}

#[derive(Debug, Clone, Copy)]
struct ElectromagneticFixtureProfile {
    reference_frequency_hz: f64,
    applied_current_a: f64,
    kind: ElectromagneticFixtureKind,
}

fn electromagnetic_profile_for_fixture(spec_id: &str) -> Option<ElectromagneticFixtureProfile> {
    match spec_id {
        "electromagnetic_reference_homogeneous_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 60.0,
                applied_current_a: 120.0,
                kind: ElectromagneticFixtureKind::Homogeneous,
            })
        }
        "electromagnetic_missing_material"
        | "electromagnetic_missing_source"
        | "electromagnetic_missing_boundary" => Some(ElectromagneticFixtureProfile {
            reference_frequency_hz: 60.0,
            applied_current_a: 120.0,
            kind: ElectromagneticFixtureKind::Homogeneous,
        }),
        "electromagnetic_reference_heterogeneous_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 400.0,
                applied_current_a: 250.0,
                kind: ElectromagneticFixtureKind::Heterogeneous,
            })
        }
        "electromagnetic_reference_sparse_assignments_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 180.0,
                applied_current_a: 170.0,
                kind: ElectromagneticFixtureKind::SparseAssignments,
            })
        }
        "electromagnetic_reference_multiregion_assignments_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 260.0,
                applied_current_a: 210.0,
                kind: ElectromagneticFixtureKind::MultiRegionAssignments,
            })
        }
        "electromagnetic_reference_overlap_interference_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 320.0,
                applied_current_a: 240.0,
                kind: ElectromagneticFixtureKind::OverlapInterference,
            })
        }
        "electromagnetic_reference_boundary_kernel_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 520.0,
                applied_current_a: 180.0,
                kind: ElectromagneticFixtureKind::BoundaryKernel,
            })
        }
        "electromagnetic_reference_boundary_penalty_stress_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 650.0,
                applied_current_a: 210.0,
                kind: ElectromagneticFixtureKind::BoundaryPenaltyStress,
            })
        }
        "electromagnetic_reference_multi_region_phased_source_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 460.0,
                applied_current_a: 260.0,
                kind: ElectromagneticFixtureKind::MultiRegionPhasedSource,
            })
        }
        _ => None,
    }
}

fn authored_cfd_boundaries_for_fixture(
    spec_id: &str,
    inlet_velocity_m_per_s: [f64; 3],
) -> Vec<runmat_analysis_core::AnalysisConstraint> {
    vec![
        fluid_constraint(
            format!("bc_cfd_inlet_{spec_id}"),
            "fluid_inlet",
            FluidFixtureConstraint::Velocity(inlet_velocity_m_per_s),
        ),
        fluid_constraint(
            format!("bc_cfd_outlet_{spec_id}"),
            "fluid_outlet",
            FluidFixtureConstraint::PressureGauge(0.0),
        ),
        fluid_constraint(
            format!("bc_cfd_wall_upper_{spec_id}"),
            "fluid_wall_upper",
            FluidFixtureConstraint::NoSlipWall,
        ),
        fluid_constraint(
            format!("bc_cfd_wall_lower_{spec_id}"),
            "fluid_wall_lower",
            FluidFixtureConstraint::NoSlipWall,
        ),
    ]
}

fn configure_model_for_fixture(spec_id: &str, model: &mut AnalysisModel) {
    if spec_id.starts_with("acoustic_harmonic_") {
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::acoustic_harmonic(
            format!("step_acoustic_{}", spec_id),
        )];
        for material in &mut model.materials {
            material.mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(1.225));
            material.acoustic = Some(runmat_analysis_core::MaterialAcousticModel {
                speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(343.0),
                damping_ratio: runmat_types::physical::DampingRatio::new(0.024),
            });
        }
        if model.materials.is_empty() {
            model.materials.push(runmat_analysis_core::MaterialModel {
                fluid: None,
                material_id: "mat_acoustic_air".to_string(),
                name: "Acoustic Air".to_string(),
                provenance: runmat_analysis_core::MaterialProvenance::Authored,
                mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(1.225)),
                elasticity: None,
                damping: None,
                thermal: None,
                acoustic: Some(runmat_analysis_core::MaterialAcousticModel {
                    speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(343.0),
                    damping_ratio: runmat_types::physical::DampingRatio::new(0.024),
                }),
                electrical: None,
                plastic: None,
            });
        }
        model.constraints.clear();
        model.boundaries = vec![
            acoustic_boundary(
                format!("bc_acoustic_rigid_{spec_id}"),
                "acoustic_wall",
                runmat_analysis_core::BoundaryOperator::AcousticRigidWall,
            ),
            acoustic_boundary(
                format!("bc_acoustic_radiation_{spec_id}"),
                "acoustic_open",
                runmat_analysis_core::BoundaryOperator::AcousticRadiation,
            ),
            acoustic_boundary(
                format!("bc_acoustic_impedance_{spec_id}"),
                "acoustic_liner",
                runmat_analysis_core::BoundaryOperator::AcousticImpedance {
                    impedance: runmat_analysis_core::SpecificAcousticImpedance::new(420.0),
                },
            ),
        ];
        replace_fixture_excitations(
            model,
            vec![FixtureExcitation {
                excitation_id: format!("load_acoustic_pressure_{}", spec_id),
                region_id: "acoustic_source".to_string(),
                quantity: FixtureExcitationQuantity::AcousticPressure(1.0),
            }],
        );
        match spec_id {
            "acoustic_harmonic_missing_source" => {
                replace_fixture_excitations(
                    model,
                    vec![FixtureExcitation {
                        excitation_id: format!("load_acoustic_invalid_force_{}", spec_id),
                        region_id: "acoustic_source".to_string(),
                        quantity: FixtureExcitationQuantity::Force([1.0, 0.0, 0.0]),
                    }],
                );
            }
            "acoustic_harmonic_missing_boundary" => {
                model.boundaries.clear();
                model.constraints = vec![fixed_constraint(
                    format!("bc_acoustic_invalid_structural_{spec_id}"),
                    "acoustic_wall",
                )];
            }
            _ => {}
        }
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.cfd = None;
    }
    if spec_id.starts_with("thermal_standalone_ramp_") {
        model.thermal = Some(runmat_analysis_core::ThermalDomain {
            reference_temperature: runmat_analysis_core::Temperature::new(293.15),
        });
        model.material_assignments.truncate(1);
        if let Some(assignment) = model.material_assignments.first_mut() {
            *assignment
                .target
                .selected_region_id_mut()
                .expect("fixture uses selected material region") =
                runmat_meshing_core::fixtures::fixture_region("thermal_core");
        }
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::thermal_transient(
            format!("step_thermal_{}", spec_id),
        )];
        model.constraints = vec![thermal_temperature(
            format!("bc_thermal_temperature_{spec_id}"),
            "thermal_hot_wall",
            343.15,
        )];
        model.boundaries = vec![
            thermal_boundary(
                format!("bc_thermal_flux_{spec_id}"),
                "thermal_flux_wall",
                runmat_analysis_core::BoundaryOperator::ThermalHeatFlux {
                    value: runmat_analysis_core::PrescribedValue::Constant(
                        runmat_analysis_core::HeatFlux::new(4500.0),
                    ),
                },
            ),
            thermal_boundary(
                format!("bc_thermal_convection_{spec_id}"),
                "thermal_open_wall",
                runmat_analysis_core::BoundaryOperator::ThermalConvection {
                    ambient_temperature: runmat_analysis_core::PrescribedValue::Constant(
                        runmat_analysis_core::Temperature::new(308.15),
                    ),
                    coefficient: runmat_analysis_core::HeatTransferCoefficient::new(35.0),
                },
            ),
        ];
        replace_fixture_excitations(
            model,
            vec![FixtureExcitation {
                excitation_id: format!("load_thermal_heat_source_{}", spec_id),
                region_id: runmat_meshing_core::fixtures::fixture_region("thermal_core"),
                quantity: FixtureExcitationQuantity::HeatGeneration(1.2e6),
            }],
        );
        match spec_id {
            "thermal_standalone_ramp_invalid_material" => {
                if let Some(material) = model.materials.first_mut() {
                    material
                        .thermal
                        .as_mut()
                        .expect("thermal fixture material")
                        .conductivity_w_per_mk =
                        runmat_types::physical::ThermalConductivity::new(0.0);
                }
            }
            "thermal_standalone_ramp_invalid_source" => {
                replace_fixture_excitations(
                    model,
                    vec![FixtureExcitation {
                        excitation_id: format!("load_thermal_invalid_heat_source_{}", spec_id),
                        region_id: runmat_meshing_core::fixtures::fixture_region("thermal_core"),
                        quantity: FixtureExcitationQuantity::Moment([1.0, 0.0, 0.0]),
                    }],
                );
            }
            "thermal_standalone_ramp_invalid_boundary" => {
                model.constraints.clear();
                model.boundaries = vec![thermal_boundary(
                    format!("bc_thermal_invalid_convection_{spec_id}"),
                    "thermal_open_wall",
                    runmat_analysis_core::BoundaryOperator::ThermalConvection {
                        ambient_temperature: runmat_analysis_core::PrescribedValue::Constant(
                            runmat_analysis_core::Temperature::new(293.15),
                        ),
                        coefficient: runmat_analysis_core::HeatTransferCoefficient::new(f64::NAN),
                    },
                )];
            }
            _ => {}
        }
        model.electro_thermal = None;
        model.electromagnetic = None;
        model.interfaces.clear();
        model.cfd = None;
    }
    if spec_id.starts_with("cfd_steady_") {
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::fluid_static(format!(
            "step_cfd_{}",
            spec_id
        ))];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.boundaries.clear();
        model.constraints = authored_cfd_boundaries_for_fixture(spec_id, [5.0, 0.0, 0.0]);
        model.cfd = Some(fixture_cfd_domain(
            model,
            &format!("cfd_velocity_{spec_id}"),
            5.0,
            1.81e-5,
            0.06,
            &[],
        ));
    }
    if spec_id.starts_with("cfd_transient_") {
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::fluid_transient(
            format!("step_cfd_{}", spec_id),
        )];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.boundaries.clear();
        model.constraints = authored_cfd_boundaries_for_fixture(spec_id, [5.0, 0.0, 0.0]);
        model.cfd = Some(fixture_cfd_domain(
            model,
            &format!("cfd_velocity_{spec_id}"),
            5.0,
            1.81e-5,
            0.06,
            &[(0.0, 0.65), (1.0, 1.0)],
        ));
    }
    if spec_id == "cfd_invalid_domain_options" || spec_id == "cfd_invalid_boundary_conditions" {
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::fluid_static(format!(
            "step_cfd_{}",
            spec_id
        ))];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.boundaries.clear();
        model.constraints = if spec_id == "cfd_invalid_boundary_conditions" {
            vec![fluid_constraint(
                format!("bc_cfd_inlet_only_{spec_id}"),
                "inlet",
                FluidFixtureConstraint::Velocity([5.0, -1.0, 0.0]),
            )]
        } else {
            authored_cfd_boundaries_for_fixture(spec_id, [5.0, 0.0, 0.0])
        };
        model.cfd = Some(fixture_cfd_domain(
            model,
            &format!("cfd_velocity_{spec_id}"),
            5.0,
            if spec_id == "cfd_invalid_domain_options" {
                0.0
            } else {
                1.81e-5
            },
            0.06,
            &[],
        ));
    }
    if spec_id.starts_with("cht_coupled_") {
        model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
            format!("sequence_{spec_id}"),
            vec![runmat_analysis_core::ProcedureStep::coupled(
                format!("step_cht_{spec_id}"),
                runmat_analysis_core::CoupledPhysicsFamily::ConjugateHeatTransfer,
                runmat_analysis_core::AnalysisProcedure::TransientResponse,
            )],
        );
        model.electro_thermal = None;
        model.thermal = Some(runmat_analysis_core::ThermalDomain {
            reference_temperature: runmat_analysis_core::Temperature::new(293.15),
        });
        model.interfaces.clear();
        model.cfd = Some(fixture_cfd_domain(
            model,
            &format!("cht_velocity_{spec_id}"),
            5.0,
            1.81e-5,
            0.06,
            &[(0.0, 0.8), (1.0, 1.0)],
        ));
        if spec_id == "cht_coupled_channel_slab_cpu" {
            model.cfd = Some(fixture_cfd_domain(
                model,
                &format!("cht_velocity_{spec_id}"),
                4.1,
                1.81e-5,
                0.052,
                &[(0.0, 0.7), (0.5, 0.9), (1.0, 1.0)],
            ));
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("cht_channel_slab_interface_{spec_id}"),
                primary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_channel"),
                ),
                secondary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("solid_slab"),
                ),
                kind: runmat_analysis_core::AnalysisInterfaceKind::ConjugateHeatTransfer(
                    runmat_analysis_core::ConjugateHeatTransferInterfaceModel {
                        transfer: runmat_analysis_core::ThermalInterfaceTransfer::Conductance(
                            runmat_analysis_core::HeatTransferCoefficient::new(2000.0),
                        ),
                    },
                ),
            }];
        }
        if spec_id == "cht_coupled_invalid_cfd_domain" {
            if let Some(cfd_domain) = model.cfd.as_mut() {
                cfd_domain.dynamic_viscosity = runmat_types::physical::Quantity::new(0.0);
            }
        }
        set_temperature_history(
            model,
            &format!("cht_temperature_{spec_id}"),
            293.15,
            60.0,
            &[(0.0, 0.5), (1.0, 1.0)],
        );
        model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
            reference_temperature: runmat_analysis_core::Temperature::new(293.15),
        });
        if spec_id == "cht_coupled_invalid_interface_mapping" {
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("cht_invalid_contact_{spec_id}"),
                primary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_channel"),
                ),
                secondary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("solid_wall"),
                ),
                kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                    runmat_analysis_core::ContactInterfaceModel {
                        friction: runmat_analysis_core::ContactFriction::Frictionless,
                    },
                ),
            }];
        }
    }
    if spec_id.starts_with("fsi_coupled_") {
        model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
            format!("sequence_{spec_id}"),
            vec![runmat_analysis_core::ProcedureStep::coupled(
                format!("step_fsi_{spec_id}"),
                runmat_analysis_core::CoupledPhysicsFamily::FluidStructureInteraction,
                runmat_analysis_core::AnalysisProcedure::TransientResponse,
            )],
        );
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.cfd = Some(fixture_cfd_domain(
            model,
            &format!("fsi_velocity_{spec_id}"),
            4.0,
            1.81e-5,
            0.06,
            &[(0.0, 0.6), (1.0, 1.0)],
        ));
        if spec_id == "fsi_coupled_pipe_plate_cpu" {
            model.cfd = Some(fixture_cfd_domain(
                model,
                &format!("fsi_velocity_{spec_id}"),
                4.2,
                1.81e-5,
                0.055,
                &[(0.0, 0.5), (0.5, 0.85), (1.0, 1.0)],
            ));
            append_fixture_excitations(
                model,
                vec![FixtureExcitation {
                    excitation_id: format!("pipe_plate_pressure_preload_{spec_id}"),
                    region_id: "pipe_wall".to_string(),
                    quantity: FixtureExcitationQuantity::Pressure(2.5e4),
                }],
            );
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("fsi_pipe_plate_interface_{spec_id}"),
                primary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_pipe"),
                ),
                secondary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("plate_wall"),
                ),
                kind: runmat_analysis_core::AnalysisInterfaceKind::FluidStructure(
                    runmat_analysis_core::FluidStructureInterfaceModel {
                        kinematics: runmat_analysis_core::FluidStructureKinematics::NoSlip,
                    },
                ),
            }];
        }
        if spec_id == "fsi_coupled_invalid_cfd_domain" {
            if let Some(cfd_domain) = model.cfd.as_mut() {
                cfd_domain.dynamic_viscosity = runmat_types::physical::Quantity::new(0.0);
            }
        }
        if spec_id == "fsi_coupled_invalid_interface_mapping" {
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("fsi_invalid_contact_{spec_id}"),
                primary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_channel"),
                ),
                secondary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("structure_wall"),
                ),
                kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                    runmat_analysis_core::ContactInterfaceModel {
                        friction: runmat_analysis_core::ContactFriction::Frictionless,
                    },
                ),
            }];
        }
    }

    if let Some(profile) = electromagnetic_profile_for_fixture(spec_id) {
        model.sequence.steps = vec![
            runmat_analysis_core::ProcedureStep::electromagnetic_harmonic(format!(
                "step_em_{}",
                spec_id
            )),
        ];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();

        if model.materials.len() < 3 {
            let base = model.materials.first().cloned().unwrap_or_else(|| {
                runmat_analysis_core::MaterialModel {
                    fluid: None,
                    material_id: "mat_default".to_string(),
                    name: "Default".to_string(),
                    provenance: runmat_analysis_core::MaterialProvenance::Authored,
                    mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(7850.0)),
                    elasticity: Some(
                        runmat_analysis_core::MaterialElasticityModel::IsotropicLinear(
                            runmat_analysis_core::MaterialMechanicalModel {
                                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(
                                    110e9,
                                ),
                                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.31),
                            },
                        ),
                    ),
                    damping: None,
                    thermal: Some(runmat_analysis_core::MaterialThermalModel {
                        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                        modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(
                            -2.5e-4,
                        ),
                        conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(
                            45.0,
                        ),
                        specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(500.0),
                        expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(
                            1.2e-5,
                        ),
                    }),
                    acoustic: None,
                    electrical: None,
                    plastic: None,
                }
            });
            model.materials = vec![
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_em_copper".to_string(),
                    name: "EM Copper".to_string(),
                    provenance: runmat_analysis_core::MaterialProvenance::Authored,
                    electrical: Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: Quantity::new(293.15),
                        conductivity_s_per_m: Quantity::new(5.8e7),
                        resistive_heating_coefficient: Quantity::new(0.0039),
                        relative_permittivity: Quantity::new(2.0),
                        relative_permeability: Quantity::new(1.0),
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(50.0),
                                conductivity_scale: 1.05,
                                dispersive_loss_scale: Some(0.02),
                                relative_permittivity_scale: Some(1.01),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(200.0),
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.025),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(1_000.0),
                                conductivity_scale: 0.94,
                                dispersive_loss_scale: Some(0.03),
                                relative_permittivity_scale: Some(0.99),
                                relative_permeability_scale: Some(1.0),
                            },
                        ],
                    }),
                    ..base.clone()
                },
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_em_ferrite".to_string(),
                    name: "EM Ferrite".to_string(),
                    provenance: runmat_analysis_core::MaterialProvenance::Authored,
                    electrical: Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: Quantity::new(293.15),
                        conductivity_s_per_m: Quantity::new(8.0e4),
                        resistive_heating_coefficient: Quantity::new(0.0020),
                        relative_permittivity: Quantity::new(14.0),
                        relative_permeability: Quantity::new(90.0),
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(50.0),
                                conductivity_scale: 1.32,
                                dispersive_loss_scale: Some(0.08),
                                relative_permittivity_scale: Some(1.15),
                                relative_permeability_scale: Some(1.35),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(200.0),
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.11),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(1_000.0),
                                conductivity_scale: 0.72,
                                dispersive_loss_scale: Some(0.16),
                                relative_permittivity_scale: Some(0.85),
                                relative_permeability_scale: Some(0.70),
                            },
                        ],
                    }),
                    ..base.clone()
                },
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_em_polymer".to_string(),
                    name: "EM Polymer".to_string(),
                    provenance: runmat_analysis_core::MaterialProvenance::Authored,
                    electrical: Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: Quantity::new(293.15),
                        conductivity_s_per_m: Quantity::new(0.2),
                        resistive_heating_coefficient: Quantity::new(0.0010),
                        relative_permittivity: Quantity::new(3.5),
                        relative_permeability: Quantity::new(1.05),
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(50.0),
                                conductivity_scale: 0.88,
                                dispersive_loss_scale: Some(0.24),
                                relative_permittivity_scale: Some(0.95),
                                relative_permeability_scale: Some(0.98),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(200.0),
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.26),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(1_000.0),
                                conductivity_scale: 1.24,
                                dispersive_loss_scale: Some(0.29),
                                relative_permittivity_scale: Some(1.10),
                                relative_permeability_scale: Some(1.04),
                            },
                        ],
                    }),
                    ..base
                },
            ];
        } else {
            for (idx, material) in model.materials.iter_mut().enumerate() {
                let (sigma, eps_r, mu_r) = match idx % 3 {
                    0 => (5.8e7, 2.0, 1.0),
                    1 => (8.0e4, 14.0, 90.0),
                    _ => (0.2, 3.5, 1.05),
                };
                let conductivity_frequency_response = match idx % 3 {
                    0 => vec![
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(50.0),
                            conductivity_scale: 1.05,
                            dispersive_loss_scale: Some(0.02),
                            relative_permittivity_scale: Some(1.01),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(200.0),
                            conductivity_scale: 1.0,
                            dispersive_loss_scale: Some(0.025),
                            relative_permittivity_scale: Some(1.0),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(1_000.0),
                            conductivity_scale: 0.94,
                            dispersive_loss_scale: Some(0.03),
                            relative_permittivity_scale: Some(0.99),
                            relative_permeability_scale: Some(1.0),
                        },
                    ],
                    1 => vec![
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(50.0),
                            conductivity_scale: 1.32,
                            dispersive_loss_scale: Some(0.08),
                            relative_permittivity_scale: Some(1.15),
                            relative_permeability_scale: Some(1.35),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(200.0),
                            conductivity_scale: 1.0,
                            dispersive_loss_scale: Some(0.11),
                            relative_permittivity_scale: Some(1.0),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(1_000.0),
                            conductivity_scale: 0.72,
                            dispersive_loss_scale: Some(0.16),
                            relative_permittivity_scale: Some(0.85),
                            relative_permeability_scale: Some(0.70),
                        },
                    ],
                    _ => vec![
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(50.0),
                            conductivity_scale: 0.88,
                            dispersive_loss_scale: Some(0.24),
                            relative_permittivity_scale: Some(0.95),
                            relative_permeability_scale: Some(0.98),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(200.0),
                            conductivity_scale: 1.0,
                            dispersive_loss_scale: Some(0.26),
                            relative_permittivity_scale: Some(1.0),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: Quantity::new(1_000.0),
                            conductivity_scale: 1.24,
                            dispersive_loss_scale: Some(0.29),
                            relative_permittivity_scale: Some(1.10),
                            relative_permeability_scale: Some(1.04),
                        },
                    ],
                };
                material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
                    reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                    conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(
                        sigma,
                    ),
                    resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(
                        0.0025,
                    ),
                    relative_permittivity: runmat_types::physical::RelativePermittivity::new(eps_r),
                    relative_permeability: runmat_types::physical::RelativePermeability::new(mu_r),
                    conductivity_frequency_response,
                });
            }
        }

        match profile.kind {
            ElectromagneticFixtureKind::Homogeneous => {
                for material in &mut model.materials {
                    material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(
                            5.8e7,
                        ),
                        resistive_heating_coefficient:
                            runmat_types::physical::InverseTemperature::new(0.0039),
                        relative_permittivity: runmat_types::physical::RelativePermittivity::new(
                            2.1,
                        ),
                        relative_permeability: runmat_types::physical::RelativePermeability::new(
                            1.0,
                        ),
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(50.0),
                                conductivity_scale: 1.02,
                                dispersive_loss_scale: Some(0.015),
                                relative_permittivity_scale: Some(1.01),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(60.0),
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.017),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: Quantity::new(1_000.0),
                                conductivity_scale: 0.95,
                                dispersive_loss_scale: Some(0.03),
                                relative_permittivity_scale: Some(0.99),
                                relative_permeability_scale: Some(1.0),
                            },
                        ],
                    });
                }
                let material_id = model
                    .materials
                    .first()
                    .map(|material| material.material_id.clone())
                    .unwrap_or_else(|| "mat_default".to_string());
                if model.material_assignments.is_empty() {
                    model
                        .material_assignments
                        .push(runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(
                                "em_region_homogeneous",
                            ),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
                        });
                } else {
                    for assignment in &mut model.material_assignments {
                        *assignment
                            .target
                            .selected_region_id_mut()
                            .expect("fixture uses selected material region") =
                            "em_region_homogeneous".to_string();
                        assignment.expected_material_id = material_id.clone();
                        assignment.assigned_material_id = material_id.clone();
                        assignment.confidence = runmat_analysis_core::EvidenceConfidence::Verified;
                    }
                }
            }
            ElectromagneticFixtureKind::Heterogeneous => {
                if model.material_assignments.is_empty() {
                    model.material_assignments = model
                        .materials
                        .iter()
                        .enumerate()
                        .map(|(idx, material)| runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                "em_region_{idx}"
                            )),
                            expected_material_id: material.material_id.clone(),
                            assigned_material_id: material.material_id.clone(),
                            confidence: match idx % 3 {
                                0 => runmat_analysis_core::EvidenceConfidence::Verified,
                                1 => runmat_analysis_core::EvidenceConfidence::Probable,
                                _ => runmat_analysis_core::EvidenceConfidence::Inferred,
                            },
                        })
                        .collect();
                } else {
                    for (idx, assignment) in model.material_assignments.iter_mut().enumerate() {
                        *assignment
                            .target
                            .selected_region_id_mut()
                            .expect("fixture uses selected material region") =
                            format!("em_region_{idx}");
                        let material_id = &model.materials[idx % model.materials.len()].material_id;
                        assignment.expected_material_id = material_id.clone();
                        assignment.assigned_material_id = material_id.clone();
                        assignment.confidence = match idx % 3 {
                            0 => runmat_analysis_core::EvidenceConfidence::Verified,
                            1 => runmat_analysis_core::EvidenceConfidence::Probable,
                            _ => runmat_analysis_core::EvidenceConfidence::Inferred,
                        };
                    }
                }
            }
            ElectromagneticFixtureKind::SparseAssignments => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..10)
                    .map(|idx| {
                        if idx < 4 {
                            let material_id = ids[idx % ids.len()].clone();
                            runmat_analysis_core::MaterialAssignment {
                                target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                    "em_sparse_region_{idx}"
                                )),
                                expected_material_id: material_id.clone(),
                                assigned_material_id: material_id,
                                confidence: if idx % 2 == 0 {
                                    runmat_analysis_core::EvidenceConfidence::Verified
                                } else {
                                    runmat_analysis_core::EvidenceConfidence::Probable
                                },
                            }
                        } else {
                            let material_id = ids[idx % ids.len()].clone();
                            runmat_analysis_core::MaterialAssignment {
                                target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                    "em_sparse_region_{idx}"
                                )),
                                expected_material_id: material_id.clone(),
                                assigned_material_id: material_id,
                                confidence: runmat_analysis_core::EvidenceConfidence::Inferred,
                            }
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::MultiRegionAssignments => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..9)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                "em_multiregion_region_{idx}"
                            )),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 2 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Probable
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Inferred
                            },
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::OverlapInterference => {
                if model.material_assignments.is_empty() {
                    model.material_assignments = model
                        .materials
                        .iter()
                        .enumerate()
                        .take(3)
                        .map(|(idx, material)| runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                "em_overlap_region_{idx}"
                            )),
                            expected_material_id: material.material_id.clone(),
                            assigned_material_id: material.material_id.clone(),
                            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
                        })
                        .collect();
                } else {
                    for (idx, assignment) in model.material_assignments.iter_mut().enumerate() {
                        *assignment
                            .target
                            .selected_region_id_mut()
                            .expect("fixture uses selected material region") =
                            format!("em_overlap_region_{}", idx % 3);
                        let material_id = &model.materials[idx % model.materials.len()].material_id;
                        assignment.expected_material_id = material_id.clone();
                        assignment.assigned_material_id = material_id.clone();
                        assignment.confidence = runmat_analysis_core::EvidenceConfidence::Verified;
                    }
                }
            }
            ElectromagneticFixtureKind::BoundaryKernel => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..32)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                "em_boundary_region_{idx}"
                            )),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 2 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Verified
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Probable
                            },
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::BoundaryPenaltyStress => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..48)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                "em_penalty_region_{idx}"
                            )),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 4 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Probable
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Verified
                            },
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::MultiRegionPhasedSource => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..16)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            target: runmat_analysis_core::PropertyRegionTarget::solid(format!(
                                "em_phase_region_{idx}"
                            )),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 3 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Inferred
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Verified
                            },
                        }
                    })
                    .collect();
            }
        }

        match profile.kind {
            ElectromagneticFixtureKind::Homogeneous => {
                model.constraints = vec![
                    magnetic_ground("em_bc_ground_0", "em_region_homogeneous"),
                    magnetic_ground("em_bc_ground_1", "em_region_homogeneous"),
                ];
                model.boundaries = vec![
                    magnetic_insulation("em_bc_insulation_0", "em_region_homogeneous"),
                    magnetic_insulation("em_bc_insulation_1", "em_region_homogeneous"),
                ];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_coil_0".to_string(),
                            region_id: "em_region_homogeneous".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_current_density_0".to_string(),
                            region_id: "em_region_homogeneous".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [1.0, 0.0, 0.0],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_current_density_1".to_string(),
                            region_id: "em_region_homogeneous".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.0, 1.0, 0.0],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::Heterogeneous => {
                model.constraints = vec![
                    magnetic_ground("em_bc_ground_hetero", "em_region_0"),
                    fixed_constraint("em_bc_structural_fallback", "em_region_2"),
                ];
                model.boundaries = vec![
                    magnetic_insulation("em_bc_insulation_hetero_0", "em_region_1"),
                    magnetic_insulation("em_bc_insulation_hetero_1", "em_region_2"),
                ];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_coil_hetero".to_string(),
                            region_id: "em_region_0".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a * 0.85,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_current_density_hetero".to_string(),
                            region_id: "em_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.8, 0.3, 0.1],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_structural_mixed".to_string(),
                            region_id: "em_region_2".to_string(),
                            quantity: FixtureExcitationQuantity::Force([0.0, -50.0, 0.0]),
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::SparseAssignments => {
                model.boundaries.clear();
                model.constraints = vec![
                    magnetic_ground("em_bc_sparse_ground", "em_sparse_region_0"),
                    fixed_constraint("em_bc_sparse_struct_0", "em_sparse_region_1"),
                    fixed_constraint("em_bc_sparse_struct_1", "em_sparse_region_2"),
                    zero_translation_constraint("em_bc_sparse_struct_2", "em_sparse_region_3"),
                ];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_sparse_coil".to_string(),
                            region_id: "em_sparse_region_0".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a * 0.4,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_sparse_current_density_0".to_string(),
                            region_id: "em_sparse_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.0, 0.6, 0.2],
                                phase_radians: Quantity::new(0.2),
                                amplitude_scale: 0.8,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_sparse_current_density_1".to_string(),
                            region_id: "em_sparse_region_2".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.3, 0.1, 0.4],
                                phase_radians: Quantity::new(0.4),
                                amplitude_scale: 0.7,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_sparse_current_density_2".to_string(),
                            region_id: "em_sparse_region_3".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.1, 0.0, 0.7],
                                phase_radians: Quantity::new(0.6),
                                amplitude_scale: 0.6,
                            },
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::MultiRegionAssignments => {
                model.constraints = vec![
                    magnetic_ground("em_bc_multiregion_ground", "em_multiregion_region_0"),
                    magnetic_ground("em_bc_multiregion_ground_1", "em_multiregion_region_3"),
                ];
                model.boundaries = vec![
                    magnetic_insulation(
                        "em_bc_multiregion_insulation_0",
                        "em_multiregion_region_1",
                    ),
                    magnetic_insulation(
                        "em_bc_multiregion_insulation_1",
                        "em_multiregion_region_2",
                    ),
                    magnetic_insulation(
                        "em_bc_multiregion_insulation_2",
                        "em_multiregion_region_4",
                    ),
                ];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_multiregion_coil".to_string(),
                            region_id: "em_multiregion_region_0".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a * 0.25,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_multiregion_current_density_0".to_string(),
                            region_id: "em_multiregion_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.4, 0.2, 0.1],
                                phase_radians: Quantity::new(0.15),
                                amplitude_scale: 0.9,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_multiregion_current_density_1".to_string(),
                            region_id: "em_multiregion_region_2".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.1, 0.5, 0.2],
                                phase_radians: Quantity::new(0.35),
                                amplitude_scale: 0.8,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_multiregion_current_density_2".to_string(),
                            region_id: "em_multiregion_region_3".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.2, 0.1, 0.6],
                                phase_radians: Quantity::new(0.55),
                                amplitude_scale: 0.7,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_multiregion_current_density_3".to_string(),
                            region_id: "em_multiregion_region_4".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.3, 0.3, 0.3],
                                phase_radians: Quantity::new(0.75),
                                amplitude_scale: 0.65,
                            },
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::OverlapInterference => {
                model.constraints = vec![magnetic_ground(
                    "em_bc_overlap_ground",
                    "em_overlap_region_0",
                )];
                model.boundaries = vec![
                    magnetic_insulation("em_bc_overlap_insulation_0", "em_overlap_region_1"),
                    magnetic_insulation("em_bc_overlap_insulation_1", "em_overlap_region_2"),
                ];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_overlap_coil_pos".to_string(),
                            region_id: "em_overlap_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_overlap_coil_neg".to_string(),
                            region_id: "em_overlap_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: -profile.applied_current_a * 0.92,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_overlap_cd_pos".to_string(),
                            region_id: "em_overlap_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [1.0, 0.4, 0.0],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_overlap_cd_neg".to_string(),
                            region_id: "em_overlap_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [-0.8, -0.35, 0.0],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::BoundaryKernel => {
                model.constraints = vec![
                    magnetic_ground("em_bc_boundary_ground_mapped", "em_boundary_region_24"),
                    magnetic_ground(
                        "em_bc_boundary_ground_unmapped_0",
                        "em_boundary_unmapped_ground_0",
                    ),
                    magnetic_ground(
                        "em_bc_boundary_ground_unmapped_1",
                        "em_boundary_unmapped_ground_1",
                    ),
                    fixed_constraint("em_bc_boundary_structural_noise", "em_boundary_region_3"),
                ];
                model.boundaries = vec![
                    magnetic_insulation(
                        "em_bc_boundary_insulation_unmapped_0",
                        "em_boundary_unmapped_insulation_0",
                    ),
                    magnetic_insulation(
                        "em_bc_boundary_insulation_unmapped_1",
                        "em_boundary_unmapped_insulation_1",
                    ),
                ];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_boundary_coil_primary".to_string(),
                            region_id: "em_boundary_region_24".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a * 0.9,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_boundary_current_density".to_string(),
                            region_id: "em_boundary_region_25".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.9, -0.5, 0.2],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::BoundaryPenaltyStress => {
                model.constraints = (0..18)
                    .map(|idx| {
                        magnetic_ground(
                            format!("em_bc_penalty_ground_{idx}"),
                            format!("em_penalty_region_{}", idx + 8),
                        )
                    })
                    .collect();
                model.boundaries = (0..18)
                    .filter(|idx| idx % 2 == 0)
                    .map(|idx| {
                        magnetic_insulation(
                            format!("em_bc_penalty_insulation_{idx}"),
                            format!("em_penalty_region_{}", idx + 20),
                        )
                    })
                    .collect();
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_penalty_coil_0".to_string(),
                            region_id: "em_penalty_region_10".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_penalty_coil_1".to_string(),
                            region_id: "em_penalty_region_13".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a * -0.65,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_penalty_density".to_string(),
                            region_id: "em_penalty_region_29".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [1.2, -0.7, 0.15],
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                    ],
                );
            }
            ElectromagneticFixtureKind::MultiRegionPhasedSource => {
                model.constraints = vec![
                    magnetic_ground("em_bc_phase_ground_0", "em_phase_region_0"),
                    magnetic_ground("em_bc_phase_ground_1", "em_phase_region_8"),
                ];
                model.boundaries = vec![magnetic_insulation(
                    "em_bc_phase_insulation_0",
                    "em_phase_region_5",
                )];
                replace_fixture_excitations(
                    model,
                    vec![
                        FixtureExcitation {
                            excitation_id: "em_load_phase_coil_0".to_string(),
                            region_id: "em_phase_region_1".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a,
                                phase_radians: Quantity::new(0.0),
                                amplitude_scale: 1.0,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_phase_coil_1".to_string(),
                            region_id: "em_phase_region_3".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: -profile.applied_current_a * 0.8,
                                phase_radians: Quantity::new(std::f64::consts::PI * 0.85),
                                amplitude_scale: 0.9,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_phase_coil_2".to_string(),
                            region_id: "em_phase_region_7".to_string(),
                            quantity: FixtureExcitationQuantity::ElectricCurrent {
                                amperes: profile.applied_current_a * 0.55,
                                phase_radians: Quantity::new(std::f64::consts::FRAC_PI_2),
                                amplitude_scale: 0.8,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_phase_density_0".to_string(),
                            region_id: "em_phase_region_10".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [0.9, -0.45, 0.2],
                                phase_radians: Quantity::new(std::f64::consts::PI / 3.0),
                                amplitude_scale: 0.9,
                            },
                        },
                        FixtureExcitation {
                            excitation_id: "em_load_phase_density_1".to_string(),
                            region_id: "em_phase_region_12".to_string(),
                            quantity: FixtureExcitationQuantity::CurrentDensity {
                                amperes_per_square_meter: [-0.85, 0.35, -0.1],
                                phase_radians: Quantity::new(-std::f64::consts::FRAC_PI_2),
                                amplitude_scale: 0.85,
                            },
                        },
                    ],
                );
            }
        }

        model.electromagnetic = Some(runmat_analysis_core::ElectromagneticDomain::Harmonic {
            reference_frequency: runmat_analysis_core::Frequency::new(
                profile.reference_frequency_hz,
            ),
        });
        match spec_id {
            "electromagnetic_missing_material" => {
                for material in &mut model.materials {
                    material.electrical = None;
                }
            }
            "electromagnetic_missing_source" => {
                replace_fixture_excitations(
                    model,
                    vec![FixtureExcitation {
                        excitation_id: "em_invalid_structural_force".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        quantity: FixtureExcitationQuantity::Force([1.0, 0.0, 0.0]),
                    }],
                );
            }
            "electromagnetic_missing_boundary" => {
                model.boundaries.clear();
                model.constraints = vec![fixed_constraint(
                    "em_invalid_structural_fixed",
                    "em_region_homogeneous",
                )];
            }
            _ => {}
        }
    }

    let mut thermo = thermo_coupling_for_fixture(spec_id);
    if thermo.is_none() && spec_id == "nonlinear_load_path_mix_gpu_provider" {
        thermo = Some(FixtureThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(75.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        });
    }
    if let Some(value) = thermo {
        set_temperature_history(
            model,
            &format!("thermo_temperature_{spec_id}"),
            value.reference_temperature_k.canonical_value(),
            value.applied_temperature_delta_k.canonical_value(),
            thermo_history_for_fixture(spec_id),
        );
        for (index, delta) in value.region_temperature_deltas.iter().enumerate() {
            set_temperature_excitation(
                model,
                &format!("thermo_region_temperature_{spec_id}_{index}"),
                value.reference_temperature_k.canonical_value(),
                delta.temperature_delta_k,
                &[],
                runmat_analysis_core::ExcitationTarget::Region(
                    runmat_analysis_core::RegionTarget::volume(&delta.region_id),
                ),
            );
        }
        for material in &mut model.materials {
            let thermal = material.thermal.as_mut().expect("thermal fixture material");
            thermal.reference_temperature_k = runmat_types::physical::Temperature::new(
                value.reference_temperature_k.canonical_value(),
            );
            thermal.expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(
                value.thermal_expansion_coefficient.canonical_value(),
            );
        }
        model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
            reference_temperature: runmat_analysis_core::Temperature::new(
                value.reference_temperature_k.canonical_value(),
            ),
        });
        for step in &mut model.sequence.steps {
            if step.family == runmat_analysis_core::PhysicsFamily::Structural
                && matches!(
                    step.procedure,
                    runmat_analysis_core::AnalysisProcedure::Static
                        | runmat_analysis_core::AnalysisProcedure::TransientResponse
                        | runmat_analysis_core::AnalysisProcedure::Nonlinear
                )
            {
                step.family = runmat_analysis_core::PhysicsFamily::Coupled(
                    runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
                );
            }
        }
    }

    if let Some(electro) = electro_coupling_for_fixture(spec_id) {
        for material in &mut model.materials {
            material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
                reference_temperature_k: runmat_types::physical::Temperature::new(
                    electro.reference_temperature_k.canonical_value(),
                ),
                conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(
                    electro.base_electrical_conductivity_s_per_m,
                ),
                resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(
                    electro.resistive_heating_coefficient.canonical_value(),
                ),
                relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
                relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
                conductivity_frequency_response: Vec::new(),
            });
        }
        set_electric_potential_history(
            model,
            &format!("electro_potential_{spec_id}"),
            electro.applied_voltage_v,
            electro_history_for_fixture(spec_id),
        );
        model.electro_thermal = Some(runmat_analysis_core::ElectroThermalDomain {
            reference_temperature: runmat_analysis_core::Temperature::new(
                electro.reference_temperature_k.canonical_value(),
            ),
        });
    }

    if let Some(plastic) = plasticity_for_fixture(spec_id) {
        for material in &mut model.materials {
            material.plastic = Some(plastic.clone());
        }
        replace_fixture_excitations(model, plastic_excitations_for_fixture(spec_id));
    }

    if let Some(contact) = contact_for_fixture(spec_id) {
        let mut regions = model
            .material_assignments
            .iter()
            .map(|assignment| {
                assignment
                    .selected_region_id()
                    .expect("fixture uses selected material region")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        if regions.is_empty() {
            regions.push("root".to_string());
            regions.push("tip".to_string());
        }
        let primary = regions
            .first()
            .cloned()
            .unwrap_or_else(|| "root".to_string());
        let secondary = regions
            .iter()
            .find(|region| **region != primary)
            .cloned()
            .unwrap_or_else(|| "tip".to_string());
        model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
            interface_id: format!("contact_{spec_id}"),
            primary: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&primary),
            ),
            secondary: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&secondary),
            ),
            kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                runmat_analysis_core::ContactInterfaceModel {
                    friction: if contact.friction_coefficient == 0.0 {
                        runmat_analysis_core::ContactFriction::Frictionless
                    } else {
                        runmat_analysis_core::ContactFriction::Coulomb {
                            coefficient: runmat_analysis_core::FrictionCoefficient::new(
                                contact.friction_coefficient,
                            ),
                        }
                    },
                },
            ),
        }];
    }
}

fn plastic_excitations_for_fixture(spec_id: &str) -> Vec<FixtureExcitation> {
    let peak_force = match spec_id {
        "nonlinear_plasticity_benchmark_gpu_provider" => 1.2e7,
        "nonlinear_plastic_hardening_reference_gpu_provider" => 3.0e7,
        "nonlinear_plastic_hardening_reference_complex_gpu_provider" => 4.2e7,
        _ => 1.0e7,
    };
    vec![
        FixtureExcitation {
            excitation_id: format!("plastic_axial_drive_{spec_id}"),
            region_id: "plastic_drive_tip".to_string(),
            quantity: FixtureExcitationQuantity::Force([peak_force, 0.0, 0.0]),
        },
        FixtureExcitation {
            excitation_id: format!("plastic_shear_drive_{spec_id}"),
            region_id: "plastic_drive_web".to_string(),
            quantity: FixtureExcitationQuantity::Force([
                0.0,
                -0.35 * peak_force,
                0.12 * peak_force,
            ]),
        },
        FixtureExcitation {
            excitation_id: format!("plastic_pressure_drive_{spec_id}"),
            region_id: "plastic_drive_face".to_string(),
            quantity: FixtureExcitationQuantity::Pressure(0.08 * peak_force),
        },
    ]
}

type FixtureRunEnvelope =
    runmat_runtime::operations::OperationEnvelope<runmat_runtime::analysis::AnalysisRunResult>;
type FixtureRunResult =
    Result<FixtureRunEnvelope, Box<runmat_runtime::operations::OperationErrorEnvelope>>;

fn boxed_fixture_run_result(
    result: Result<FixtureRunEnvelope, runmat_runtime::operations::OperationErrorEnvelope>,
) -> FixtureRunResult {
    result.map_err(Box::new)
}

fn run_fixture_cpu(
    spec: &FixtureSpec,
    model: &AnalysisModel,
    solver_mesh_artifact_path: Option<&str>,
) -> FixtureRunResult {
    let solver_mesh_artifact_path = solver_mesh_artifact_path.map(str::to_owned);
    let procedure_step = fixture_procedure_step(model);
    let solver = match AnalysisSolverOptions::defaults_for(procedure_step) {
        Ok(solver) => solver,
        Err(_)
            if procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::TransientResponse
                && matches!(
                    procedure_step.family,
                    runmat_analysis_core::PhysicsFamily::Coupled(
                        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical
                            | runmat_analysis_core::CoupledPhysicsFamily::Electrothermal
                    )
                ) =>
        {
            return boxed_fixture_run_result(analysis_run_transient_with_options_op(
                model,
                ComputeBackend::Cpu,
                transient_options_for_spec(spec, solver_mesh_artifact_path),
                OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
            ));
        }
        Err(error) => panic!(
            "fixture procedure step must select a supported solver or an explicitly governed negative path: {error}"
        ),
    };
    boxed_fixture_run_result(match solver {
        AnalysisSolverOptions::StructuralLinear(_) => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Cpu,
            default_options(solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralModes(_) => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                solver_mesh_artifact_path: solver_mesh_artifact_path.clone(),
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(
                    spec.modal_mode_count.unwrap_or_else(|| {
                        AnalysisModalRunOptions::default()
                            .truncation
                            .requested_mode_count()
                    }),
                ),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralFrequency(_) => panic!(
            "structural-frequency fixtures must execute through the study runner, not the legacy single-run CPU fixture harness"
        ),
        AnalysisSolverOptions::StructuralResponseSpectrum(_) => panic!(
            "response-spectrum fixtures must execute through the study runner, not the legacy single-run CPU fixture harness"
        ),
        AnalysisSolverOptions::StructuralRandomResponse(_) => panic!(
            "random-response fixtures must execute through the study runner, not the legacy single-run CPU fixture harness"
        ),
        AnalysisSolverOptions::StructuralBuckling(_) => panic!(
            "structural-buckling fixtures must execute through the study runner, not the legacy single-run CPU fixture harness"
        ),
        AnalysisSolverOptions::AcousticResponse(_) => panic!(
            "acoustic modal/transient fixtures must execute through the study runner, not the legacy single-run CPU fixture harness"
        ),
        AnalysisSolverOptions::AcousticHelmholtz(_) => analysis_run_acoustic_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisAcousticRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisAcousticRunOptions::default().mode_count),
                ..AnalysisAcousticRunOptions::default()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralTransient(_) => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Cpu,
            transient_options_for_spec(spec, solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::ThermalDiffusion(_) => analysis_run_thermal_with_options_op(
            model,
            ComputeBackend::Cpu,
            thermal_options_for_spec(spec, solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::IncompressibleFlow(_) => analysis_run_cfd_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisCfdRunOptions::default(),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralNonlinear(_) => analysis_run_nonlinear_with_options_op(
            model,
            ComputeBackend::Cpu,
            nonlinear_options_for_spec(spec, solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::ElectricalStatic(_)
        | AnalysisSolverOptions::Magnetostatic(_)
        | AnalysisSolverOptions::MaxwellHarmonic(_)
        | AnalysisSolverOptions::LineParameters(_) => panic!(
            "electromagnetic fixtures must execute through the governed typed study runner, not the legacy single-run fixture harness"
        ),
    })
}

fn run_fixture_gpu(
    spec: &FixtureSpec,
    model: &AnalysisModel,
    mode: GpuMode,
    solver_mesh_artifact_path: Option<&str>,
) -> FixtureRunResult {
    let solver_mesh_artifact_path = solver_mesh_artifact_path.map(str::to_owned);
    let run = || {
        match AnalysisSolverOptions::defaults_for(fixture_procedure_step(model))
        .expect("fixture procedure step must select a supported solver")
    {
        AnalysisSolverOptions::StructuralLinear(_) => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Gpu,
            default_options(solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralModes(_) => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisModalRunOptions {
                solver_mesh_artifact_path: solver_mesh_artifact_path.clone(),
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(
                    spec.modal_mode_count.unwrap_or_else(|| {
                        AnalysisModalRunOptions::default()
                            .truncation
                            .requested_mode_count()
                    }),
                ),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralFrequency(_) => panic!(
            "structural-frequency fixtures must execute through the study runner, not the legacy single-run GPU fixture harness"
        ),
        AnalysisSolverOptions::StructuralResponseSpectrum(_) => panic!(
            "response-spectrum fixtures must execute through the study runner, not the legacy single-run GPU fixture harness"
        ),
        AnalysisSolverOptions::StructuralRandomResponse(_) => panic!(
            "random-response fixtures must execute through the study runner, not the legacy single-run GPU fixture harness"
        ),
        AnalysisSolverOptions::StructuralBuckling(_) => panic!(
            "structural-buckling fixtures must execute through the study runner, not the legacy single-run GPU fixture harness"
        ),
        AnalysisSolverOptions::AcousticResponse(_) => panic!(
            "acoustic modal/transient fixtures must execute through the study runner, not the legacy single-run GPU fixture harness"
        ),
        AnalysisSolverOptions::AcousticHelmholtz(_) => analysis_run_acoustic_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisAcousticRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisAcousticRunOptions::default().mode_count),
                ..AnalysisAcousticRunOptions::default()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralTransient(_) => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Gpu,
            transient_options_for_spec(spec, solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::ThermalDiffusion(_) => analysis_run_thermal_with_options_op(
            model,
            ComputeBackend::Gpu,
            thermal_options_for_spec(spec, solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::IncompressibleFlow(_) => analysis_run_cfd_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisCfdRunOptions::default(),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::StructuralNonlinear(_) => analysis_run_nonlinear_with_options_op(
            model,
            ComputeBackend::Gpu,
            nonlinear_options_for_spec(spec, solver_mesh_artifact_path.clone()),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisSolverOptions::ElectricalStatic(_)
        | AnalysisSolverOptions::Magnetostatic(_)
        | AnalysisSolverOptions::MaxwellHarmonic(_)
        | AnalysisSolverOptions::LineParameters(_) => panic!(
            "electromagnetic fixtures must execute through the governed typed study runner, not the legacy single-run fixture harness"
        ),
    }
    };
    boxed_fixture_run_result(match mode {
        GpuMode::WithProvider => with_harness_provider(run),
        GpuMode::WithoutProvider => {
            let _guard = ThreadProviderGuard::set(None);
            run()
        }
    })
}

fn parse_metric_value(message: &str, key: &str) -> Option<f64> {
    message
        .split_whitespace()
        .find_map(|token| token.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse::<f64>().ok())
}

fn diagnostic_metric(
    run: &runmat_runtime::analysis::AnalysisRunResult,
    code: &str,
    key: &str,
) -> Option<f64> {
    run.run
        .diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| parse_metric_value(&diag.message, key))
}

fn analysis_result_field_max_abs(run: &AnalysisRunResult, field_id: &str) -> Option<f64> {
    analysis_result_field(run, field_id)
        .and_then(AnalysisField::as_host_f64)
        .map(|values| {
            values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
        })
}

struct ThresholdAssertion<'a> {
    fixture_id: &'a str,
    name: &'a str,
    source_diagnostic: &'a str,
    observed: Option<f64>,
    min_allowed: Option<f64>,
    max_allowed: Option<f64>,
}

fn record_threshold_assertion(
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    assertion: ThresholdAssertion<'_>,
) {
    let ThresholdAssertion {
        fixture_id,
        name,
        source_diagnostic,
        observed,
        min_allowed,
        max_allowed,
    } = assertion;
    let passed = observed
        .map(|value| {
            min_allowed.map(|min| value >= min).unwrap_or(true)
                && max_allowed.map(|max| value <= max).unwrap_or(true)
        })
        .unwrap_or(false);
    assertions.push(ThresholdAssertionRecord {
        name: name.into(),
        source_diagnostic: source_diagnostic.to_string(),
        observed,
        min_allowed,
        max_allowed,
        passed,
    });
    if !passed {
        failures.push(format!(
            "threshold assertion failed for fixture {}: {} observed={:?} min={:?} max={:?}",
            fixture_id, name, observed, min_allowed, max_allowed
        ));
    }
}

macro_rules! push_threshold_assertion {
    (
        $fixture_id:expr,
        $assertions:expr,
        $failures:expr,
        $name:expr,
        $source_diagnostic:expr,
        $observed:expr,
        $min_allowed:expr,
        $max_allowed:expr $(,)?
    ) => {
        record_threshold_assertion(
            $assertions,
            $failures,
            ThresholdAssertion {
                fixture_id: $fixture_id,
                name: $name,
                source_diagnostic: $source_diagnostic,
                observed: $observed,
                min_allowed: $min_allowed,
                max_allowed: $max_allowed,
            },
        )
    };
}

fn push_fixed_step_transient_assertions(
    fixture_id: &str,
    execution: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    expected_interval_count: Option<usize>,
    maximum_residual: Option<f64>,
) {
    let method = run
        .run
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "FEA_TRANSIENT_METHOD")
        .map(|diagnostic| diagnostic.message.as_str());
    if !method.is_some_and(|message| message.contains("solver=newmark_average_acceleration_pcg")) {
        failures.push(format!(
            "fixture {fixture_id} {execution} run did not report the fixed-step average-acceleration Newmark method"
        ));
    }
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        &format!("{execution}_transient_newmark_beta"),
        "FEA_TRANSIENT_METHOD",
        method.and_then(|message| parse_metric_value(message, "beta")),
        Some(0.25),
        Some(0.25),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        &format!("{execution}_transient_newmark_gamma"),
        "FEA_TRANSIENT_METHOD",
        method.and_then(|message| parse_metric_value(message, "gamma")),
        Some(0.5),
        Some(0.5),
    );
    if let Some(expected) = expected_interval_count.map(|value| value as f64) {
        push_threshold_assertion!(
            fixture_id,
            assertions,
            failures,
            &format!("{execution}_transient_nominal_interval_count"),
            "FEA_TRANSIENT_CONVERGENCE",
            diagnostic_metric(run, "FEA_TRANSIENT_CONVERGENCE", "nominal_interval_count"),
            Some(expected),
            Some(expected),
        );
        push_threshold_assertion!(
            fixture_id,
            assertions,
            failures,
            &format!("{execution}_transient_accepted_step_count"),
            "FEA_TRANSIENT_CONVERGENCE",
            diagnostic_metric(run, "FEA_TRANSIENT_CONVERGENCE", "accepted_step_count"),
            Some(expected),
            None,
        );
    }
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        &format!("{execution}_transient_rejected_step_count"),
        "FEA_TRANSIENT_CONVERGENCE",
        diagnostic_metric(run, "FEA_TRANSIENT_CONVERGENCE", "rejected_step_count"),
        Some(0.0),
        Some(0.0),
    );
    let dt_min = diagnostic_metric(run, "FEA_TRANSIENT_STABILITY", "dt_min");
    let dt_max = diagnostic_metric(run, "FEA_TRANSIENT_STABILITY", "dt_max");
    for (name, observed) in [("dt_min", dt_min), ("dt_max", dt_max)] {
        push_threshold_assertion!(
            fixture_id,
            assertions,
            failures,
            &format!("{execution}_transient_{name}"),
            "FEA_TRANSIENT_STABILITY",
            observed,
            Some(f64::MIN_POSITIVE),
            Some(f64::MAX),
        );
    }
    if matches!((dt_min, dt_max), (Some(minimum), Some(maximum)) if minimum > maximum) {
        failures.push(format!(
            "fixture {fixture_id} {execution} run reported transient dt_min greater than dt_max"
        ));
    }
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        &format!("{execution}_transient_max_projected_dynamic_residual"),
        "FEA_TRANSIENT_STABILITY",
        diagnostic_metric(
            run,
            "FEA_TRANSIENT_STABILITY",
            "max_projected_dynamic_residual",
        ),
        Some(0.0),
        maximum_residual,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        &format!("{execution}_transient_maximum_physical_energy"),
        "FEA_TRANSIENT_ENERGY",
        diagnostic_metric(
            run,
            "FEA_TRANSIENT_ENERGY",
            "maximum_physical_strain_plus_kinetic_energy_joules",
        ),
        Some(0.0),
        Some(f64::MAX),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        &format!("{execution}_transient_maximum_relative_energy_balance_error"),
        "FEA_TRANSIENT_ENERGY",
        diagnostic_metric(
            run,
            "FEA_TRANSIENT_ENERGY",
            "maximum_relative_energy_balance_error",
        ),
        Some(0.0),
        Some(1.0e-6),
    );
    if let Err(error) = validate_transient_result(run) {
        failures.push(format!(
            "fixture {fixture_id} {execution} run returned invalid transient results: {error}"
        ));
    }
}

fn validate_transient_result(run: &AnalysisRunResult) -> Result<(), String> {
    let transient = run
        .transient_results
        .as_ref()
        .ok_or_else(|| "transient payload is missing".to_string())?;
    let snapshot_count = transient.time_points_s.len();
    if snapshot_count == 0
        || transient
            .time_points_s
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("time grid is empty or nonfinite".to_string());
    }
    if transient
        .time_points_s
        .windows(2)
        .any(|times| times[1] <= times[0])
    {
        return Err("time grid is not strictly increasing".to_string());
    }
    let required_fields = [
        &transient.displacement_snapshots,
        &transient.velocity_snapshots,
        &transient.acceleration_snapshots,
        &transient.kinetic_energy_snapshots,
        &transient.strain_energy_snapshots,
        &transient.residual_norm_snapshots,
        &transient.reaction_force_snapshots,
        &transient.external_work_snapshots,
        &transient.support_work_snapshots,
        &transient.damping_dissipation_snapshots,
        &transient.energy_balance_residual_snapshots,
        &transient.relative_energy_balance_error_snapshots,
    ];
    if required_fields
        .iter()
        .any(|history| history.len() != snapshot_count)
    {
        return Err("required transient histories do not match the time grid".to_string());
    }
    if transient
        .residual_norms
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err("physical residual history contains a nonfinite value".to_string());
    }
    if transient.residual_norms.len() != snapshot_count {
        return Err("physical residual history does not match the time grid".to_string());
    }
    if required_fields
        .iter()
        .flat_map(|history| history.iter())
        .any(|field| {
            field.as_host_f64().is_none_or(|values| {
                values.is_empty() || values.iter().any(|value| !value.is_finite())
            })
        })
    {
        return Err(
            "required transient history contains unavailable or nonfinite values".to_string(),
        );
    }
    let optional_fields = [
        &transient.rotation_snapshots,
        &transient.angular_velocity_snapshots,
        &transient.angular_acceleration_snapshots,
        &transient.strain_snapshots,
        &transient.stress_snapshots,
        &transient.strain_energy_density_snapshots,
        &transient.von_mises_snapshots,
        &transient.reaction_moment_snapshots,
        &transient.electro_thermal_temperature_snapshots,
        &transient.electro_thermal_thermal_residual_snapshots,
    ];
    if optional_fields.iter().any(|history| {
        !history.is_empty()
            && (history.len() != snapshot_count
                || history.iter().any(|field| {
                    field.as_host_f64().is_none_or(|values| {
                        values.is_empty() || values.iter().any(|value| !value.is_finite())
                    })
                }))
    }) {
        return Err("optional formulation-owned history is incomplete or nonfinite".to_string());
    }
    if transient.dynamic_balance_snapshots.len() != snapshot_count {
        return Err("dynamic-balance history does not match the time grid".to_string());
    }
    for balance in &transient.dynamic_balance_snapshots {
        balance
            .validate()
            .map_err(|error| format!("dynamic-balance history is invalid: {error}"))?;
    }
    Ok(())
}

fn push_thermal_standalone_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_max_relative_residual_norm",
        "FEA_THERMAL_STABILITY",
        diagnostic_metric(run, "FEA_THERMAL_STABILITY", "max_relative_residual_norm",),
        Some(0.0),
        Some(1.0e-4),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_min_temperature_k",
        "FEA_THERMAL_STABILITY",
        diagnostic_metric(run, "FEA_THERMAL_STABILITY", "minimum_temperature_k",),
        Some(290.0),
        Some(360.0),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_max_temperature_k",
        "FEA_THERMAL_STABILITY",
        diagnostic_metric(run, "FEA_THERMAL_STABILITY", "maximum_temperature_k",),
        Some(300.0),
        Some(370.0),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_energy_balance_residual_ratio",
        "FEA_THERMAL_ENERGY_BALANCE",
        diagnostic_metric(run, "FEA_THERMAL_ENERGY_BALANCE", "max_relative_residual",),
        Some(0.0),
        Some(1.0e-4),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_conductivity_spread_ratio",
        "FEA_THERMAL_CONSTITUTIVE",
        diagnostic_metric(run, "FEA_THERMAL_CONSTITUTIVE", "conductivity_spread_ratio"),
        Some(1.0),
        Some(1.5),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_heat_capacity_spread_ratio",
        "FEA_THERMAL_CONSTITUTIVE",
        diagnostic_metric(
            run,
            "FEA_THERMAL_CONSTITUTIVE",
            "heat_capacity_spread_ratio",
        ),
        Some(1.0),
        Some(1.5),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_element_count",
        "FEA_THERMAL_FIELD_RECOVERY",
        diagnostic_metric(run, "FEA_THERMAL_FIELD_RECOVERY", "element_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_boundary_face_count",
        "FEA_THERMAL_FIELD_RECOVERY",
        diagnostic_metric(run, "FEA_THERMAL_FIELD_RECOVERY", "boundary_face_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_initial_prescribed_node_count",
        "FEA_THERMAL_INITIAL_STATE",
        diagnostic_metric(run, "FEA_THERMAL_INITIAL_STATE", "prescribed_node_count",),
        Some(1.0),
        None,
    );
}

fn push_em_reference_topology_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_edge_dof_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(run, "FEA_EM_MAXWELL_EDGE_TOPOLOGY", "edge_dof_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_element_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(run, "FEA_EM_MAXWELL_EDGE_TOPOLOGY", "element_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_oriented_edge_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(run, "FEA_EM_MAXWELL_EDGE_TOPOLOGY", "oriented_edge_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_full_topology_edge_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(
            run,
            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
            "full_topology_edge_count",
        ),
        Some(3.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_full_topology_element_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(
            run,
            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
            "full_topology_element_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_incidence_element_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(
            run,
            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
            "incidence_element_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_incidence_orientation_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(
            run,
            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
            "incidence_orientation_count",
        ),
        Some(3.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_incidence_pair_count",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(run, "FEA_EM_MAXWELL_EDGE_TOPOLOGY", "incidence_pair_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "electromagnetic_incidence_operator_pair_coverage_ratio",
        "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
        diagnostic_metric(
            run,
            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
            "incidence_operator_pair_coverage_ratio",
        ),
        Some(0.5),
        Some(1.0),
    );
}

fn push_linear_structural_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_normalized_residual_norm",
        "FEA_STRUCTURAL_RESIDUAL",
        diagnostic_metric(run, "FEA_STRUCTURAL_RESIDUAL", "normalized_residual_norm"),
        Some(0.0),
        Some(1.0e-6),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_total_strain_energy",
        "FEA_STRUCTURAL_ENERGY",
        diagnostic_metric(run, "FEA_STRUCTURAL_ENERGY", "total_strain_energy").or_else(|| {
            analysis_result_field_max_abs(run, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        }),
        Some(0.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_work_energy_ratio",
        "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
            "work_energy_ratio",
        ),
        Some(0.999_999),
        Some(1.000_001),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_work_energy_residual_ratio",
        "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
            "work_energy_residual_ratio",
        ),
        Some(0.0),
        Some(1.0e-8),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_known_answer_coverage_ratio",
        "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
            "known_answer_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_active_stiffness_edge_count",
        "FEA_STRUCTURAL_FIELD_RECOVERY",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_FIELD_RECOVERY",
            "active_stiffness_edge_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_recovery_element_count",
        "FEA_STRUCTURAL_FIELD_RECOVERY",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_FIELD_RECOVERY",
            "recovery_element_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_max_edge_displacement_jump",
        "FEA_STRUCTURAL_FIELD_RECOVERY",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_FIELD_RECOVERY",
            "max_edge_displacement_jump",
        ),
        Some(1.0e-12),
        Some(1.0e-2),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_von_mises_peak_pa",
        FEA_FIELD_STRUCTURAL_VON_MISES,
        analysis_result_field_max_abs(run, FEA_FIELD_STRUCTURAL_VON_MISES),
        Some(1.0e5),
        Some(1.0e9),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_stress_tensor_peak_pa",
        FEA_FIELD_STRUCTURAL_STRESS,
        analysis_result_field_max_abs(run, FEA_FIELD_STRUCTURAL_STRESS),
        Some(1.0e5),
        Some(1.0e9),
    );
}

fn push_structural_reference_kinematics_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_reference_transverse_displacement_leakage_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "transverse_displacement_leakage_ratio",
        ),
        Some(0.0),
        Some(0.1),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_reference_primary_stress_component_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "primary_stress_component_ratio",
        ),
        Some(0.5),
        Some(1.0),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_reference_directional_coverage_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "directional_reference_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_reference_closed_form_displacement_error_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "closed_form_displacement_error_ratio",
        ),
        Some(0.0),
        Some(1.0e-7),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_reference_closed_form_stress_error_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "closed_form_stress_error_ratio",
        ),
        Some(0.0),
        Some(1.0e-7),
    );
    push_threshold_assertion!(
        fixture_id,
        assertions,
        failures,
        "structural_reference_closed_form_coverage_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "closed_form_reference_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
}

fn push_structural_beam_moment_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    for (name, diagnostic, metric, min, max) in [
        (
            "structural_rotational_dof_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_rotational_dof_count",
            Some(6.0),
            Some(6.0),
        ),
        (
            "structural_rotation_node_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_rotation_node_count",
            Some(2.0),
            Some(2.0),
        ),
        (
            "structural_moment_load_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_moment_load_count",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_direct_rotational_moment_load_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_direct_rotational_moment_load_count",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_beam_element_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_beam_element_count",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_beam_local_frame_coverage_ratio",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_beam_local_frame_coverage_ratio",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_beam_stiffness_matrix_symmetry_residual",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_beam_stiffness_matrix_symmetry_residual",
            Some(0.0),
            Some(1.0e-10),
        ),
        (
            "structural_reaction_moment_norm_n_m",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_reaction_moment_norm_n_m",
            Some(1.0e-12),
            None,
        ),
        (
            "structural_moment_requested_norm_n_m",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_requested_norm_n_m",
            Some(1.0e-12),
            None,
        ),
        (
            "structural_moment_realized_norm_n_m",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_realized_norm_n_m",
            Some(1.0e-12),
            None,
        ),
        (
            "structural_moment_realization_ratio",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_realization_ratio",
            Some(0.999_999),
            Some(1.000_001),
        ),
        (
            "structural_moment_balance_residual_ratio",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_balance_residual_ratio",
            Some(0.0),
            Some(1.0e-6),
        ),
        (
            "structural_beam_closed_form_rotation_error_ratio",
            "FEA_STRUCTURAL_BEAM_CLOSED_FORM",
            "structural_beam_closed_form_rotation_error_ratio",
            Some(0.0),
            Some(1.0e-6),
        ),
        (
            "structural_beam_closed_form_displacement_error_ratio",
            "FEA_STRUCTURAL_BEAM_CLOSED_FORM",
            "structural_beam_closed_form_displacement_error_ratio",
            Some(0.0),
            Some(1.0e-6),
        ),
        (
            "structural_beam_closed_form_torsion_error_ratio",
            "FEA_STRUCTURAL_BEAM_CLOSED_FORM",
            "structural_beam_closed_form_torsion_error_ratio",
            Some(0.0),
            Some(1.0e-6),
        ),
        (
            "structural_beam_closed_form_coverage_ratio",
            "FEA_STRUCTURAL_BEAM_CLOSED_FORM",
            "structural_beam_closed_form_coverage_ratio",
            Some(1.0),
            Some(1.0),
        ),
    ] {
        push_threshold_assertion!(
            fixture_id,
            assertions,
            failures,
            name,
            diagnostic,
            diagnostic_metric(run, diagnostic, metric),
            min,
            max,
        );
    }
}

fn push_structural_shell_moment_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    for (name, diagnostic, metric, min, max) in [
        (
            "structural_rotational_dof_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_rotational_dof_count",
            Some(9.0),
            Some(9.0),
        ),
        (
            "structural_rotation_node_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_rotation_node_count",
            Some(3.0),
            Some(3.0),
        ),
        (
            "structural_moment_load_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_moment_load_count",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_direct_rotational_moment_load_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_direct_rotational_moment_load_count",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_shell_element_count",
            "FEA_STRUCTURAL_ROTATIONAL_DOF",
            "structural_shell_element_count",
            Some(1.0),
            Some(1.0),
        ),
        (
            "structural_reaction_moment_norm_n_m",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_reaction_moment_norm_n_m",
            Some(1.0e-12),
            None,
        ),
        (
            "structural_moment_requested_norm_n_m",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_requested_norm_n_m",
            Some(1.0e-12),
            None,
        ),
        (
            "structural_moment_realized_norm_n_m",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_realized_norm_n_m",
            Some(1.0e-12),
            None,
        ),
        (
            "structural_moment_realization_ratio",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_realization_ratio",
            Some(0.999_99),
            Some(1.000_01),
        ),
        (
            "structural_moment_balance_residual_ratio",
            "FEA_STRUCTURAL_MOMENT_BALANCE",
            "structural_moment_balance_residual_ratio",
            Some(0.0),
            Some(1.0e-5),
        ),
    ] {
        push_threshold_assertion!(
            fixture_id,
            assertions,
            failures,
            name,
            diagnostic,
            diagnostic_metric(run, diagnostic, metric),
            min,
            max,
        );
    }

    // A single shell triangle loaded only by a nodal moment carries no transverse load, so
    // equilibrium of the transverse-displacement rows forces the integral of the transverse
    // shear resultant over the element to vanish. The MITC3 assumed-shear field is linear, so
    // its centroid value (the published one) is zero up to round-off: Q = div(M) = 0 for this
    // pure-bending patch. The original lower bound of 1e-12 predates the MITC3 tying
    // (22468691f4, which removed parasitic shear locking) and asserted the locking artefact.
    for (name, field_id, min, max) in [
        (
            "structural_shell_membrane_force_max_abs",
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            Some(0.0),
            None,
        ),
        (
            "structural_shell_bending_moment_max_abs",
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            Some(1.0e-12),
            None,
        ),
        (
            "structural_shell_transverse_shear_max_abs",
            FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
            Some(0.0),
            Some(1.0e-9),
        ),
        (
            "structural_shell_von_mises_max_abs",
            FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
            Some(1.0e-12),
            None,
        ),
    ] {
        push_threshold_assertion!(
            fixture_id,
            assertions,
            failures,
            name,
            field_id,
            analysis_result_field_max_abs(run, field_id),
            min,
            max,
        );
    }
}

fn validate_fallback_event_schema(event: &FallbackEvent) -> bool {
    match (event.stage, event.reason) {
        (FallbackStage::Solver, FallbackReason::RequestedBackendUnavailable) => {
            event.actual_preconditioner.is_none()
        }
        (FallbackStage::Preconditioner, FallbackReason::RequestedPreconditionerUnavailable) => {
            event.actual_preconditioner.is_some()
        }
        _ => false,
    }
}

fn compute_parity(left: &[f64], right: &[f64]) -> ParitySummary {
    let mut max_abs = 0.0;
    let mut max_rel = 0.0;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        let abs = (lhs - rhs).abs();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        let rel = abs / scale;
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    ParitySummary {
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
    }
}

fn subdivide_fixture_tetrahedron(mesh: &mut SolverMeshArtifact, model: &AnalysisModel) {
    let mut identity_hasher = Sha256::new();
    identity_hasher.update(b"runmat-analysis-fixture-interior-node\0");
    identity_hasher.update(model.geometry_id.as_bytes());
    let interior_identity = StableDigest::from_bytes(identity_hasher.finalize().into());
    let interior_node_id = 5;
    let topo = &mut *mesh.topology;
    topo.nodes.push(SolverMeshNode {
        node_id: interior_node_id,
        stable_identity: interior_identity,
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new([
                0.25, 0.25, 0.25,
            ]),
        provenance: topo.nodes[0].provenance.clone(),
        exact_parameters: Vec::new(),
    });

    let original_element = mesh.topology.volume_elements[0].clone();
    let tetrahedra = [
        [1, 2, 3, interior_node_id],
        [1, 4, 2, interior_node_id],
        [1, 3, 4, interior_node_id],
        [2, 4, 3, interior_node_id],
    ];
    let node_identity = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.stable_identity))
        .collect::<std::collections::BTreeMap<_, _>>();
    mesh.topology.volume_elements = tetrahedra
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| SolverVolumeElement {
            element_id: index as u64 + 1,
            stable_identity: solver_volume_element_identity(
                node_ids.map(|node_id| node_identity[&node_id]),
            ),
            order: ElementOrder::Tet4,
            node_ids: node_ids.to_vec(),
            region_id: original_element.region_id.clone(),
            provenance: original_element.provenance.clone(),
        })
        .collect();
    mesh.topology.neighbors = (1_u64..=4)
        .flat_map(|element_id| {
            let adjacent = (1_u64..=4)
                .filter(|candidate| *candidate != element_id)
                .collect::<Vec<_>>();
            (0_u8..4).map(move |local_face_index| MeshNeighbor {
                element_id,
                local_face_index,
                adjacent_element_id: if local_face_index == 0 {
                    None
                } else {
                    Some(adjacent[usize::from(local_face_index - 1)])
                },
            })
        })
        .collect();
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.adjacent_volume_element_ids = vec![index as u64 + 1];
    }
    mesh.topology.regions[0].element_ids = (1_u64..=4).collect();
    for field in &mut mesh.topology.field_topologies {
        match field.location {
            FieldTopologyLocation::Node => field.ordered_entity_ids = (1_u64..=5).collect(),
            FieldTopologyLocation::VolumeElement => {
                field.ordered_entity_ids = (1_u64..=4).collect()
            }
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => {}
        }
    }
}

fn write_fixture_solver_mesh_artifact(
    root: &Path,
    model: &AnalysisModel,
) -> Result<PathBuf, String> {
    fs::create_dir_all(root).map_err(|error| error.to_string())?;
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let is_physical_transient_reference = matches!(
        model.model_id.0.as_str(),
        "structural_transient_tet4_reference"
            | "structural_transient_tet4_support_motion_reference"
    );
    if !is_physical_transient_reference {
        subdivide_fixture_tetrahedron(&mut mesh, model);
    }
    let mut source_hasher = Sha256::new();
    source_hasher.update(model.geometry_id.as_bytes());
    mesh.geometry.source_digest = StableDigest::from_bytes(source_hasher.finalize().into());
    mesh.geometry.geometry_revision = u64::from(model.geometry_revision);

    if let Some(assignment) = model.material_assignments.first() {
        let assigned_region = assignment
            .selected_region_id()
            .expect("fixture uses selected material region")
            .to_owned();
        let region_identity =
            runmat_meshing_core::fixtures::fixture_entity_for_selection(&assigned_region)
                .unwrap_or_else(|| {
                    PersistentEntityId::new(
                        PersistentEntityKind::Region,
                        assigned_region.clone(),
                        vec!["root".to_owned()],
                    )
                });
        for element in &mut mesh.topology.volume_elements {
            element.region_id = region_identity.clone();
        }
        mesh.topology.regions[0].region_id = region_identity;
    }

    let mut boundary_regions = std::collections::BTreeSet::new();
    for constraint in &model.constraints {
        constraint.visit_targets(|target| {
            boundary_regions.insert(target.region_id.clone());
        });
    }
    boundary_regions.extend(
        model
            .boundaries
            .iter()
            .map(|boundary| boundary.target.region_id.clone()),
    );
    for excitation in &model.excitations {
        if let runmat_analysis_core::ExcitationTarget::Region(target) =
            &excitation.excitation().target
        {
            boundary_regions.insert(target.region_id.clone());
        }
    }
    for interface in &model.interfaces {
        boundary_regions.insert(interface.primary.region_id.clone());
        boundary_regions.insert(interface.secondary.region_id.clone());
    }
    let boundary_face_count = mesh.topology.boundary_faces.len();
    for (index, region_id) in boundary_regions.into_iter().enumerate() {
        // Canonical fixture selectors name a fixture entity: place that exact entity, matched by
        // its source name. Other targets (point ids) keep a region identity.
        let fixture_entity =
            runmat_meshing_core::fixtures::fixture_entity_for_selection(&region_id);
        let source_name = fixture_entity
            .as_ref()
            .map_or(region_id.as_str(), |entity| {
                entity.source_topology_id.as_str()
            });
        let face_index = if is_physical_transient_reference {
            match source_name {
                "root" => 0,
                "free_vertex_y_support" => 1,
                "free_vertex_z_support" => 2,
                "dynamic_force" => 3,
                _ => index % boundary_face_count,
            }
        } else {
            index % boundary_face_count
        };
        let face = &mut mesh.topology.boundary_faces[face_index];
        face.provenance.push(fixture_entity.unwrap_or_else(|| {
            PersistentEntityId::new(
                PersistentEntityKind::Region,
                region_id.clone(),
                vec!["root".to_owned()],
            )
        }));
        face.provenance.sort();
        face.provenance.dedup();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .map_err(|error| format!("{}: {error}", model.model_id.0))?;
    let path = root.join("solver_mesh.cbor");
    fs::write(
        &path,
        mesh.canonical_encode().map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(path)
}

fn fixture_requires_solver_mesh(spec: &FixtureSpec, model: &AnalysisModel) -> bool {
    spec.expect_validate_error.is_none()
        && spec.expect_run_error.is_none()
        && ((spec.id == "transient_physical_tet4_reference_cpu"
            || spec.id == "transient_physical_tet4_support_motion_reference_cpu")
            || matches!(
                AnalysisSolverOptions::defaults_for(fixture_procedure_step(model)),
                Ok(AnalysisSolverOptions::ThermalDiffusion(_))
                    | Ok(AnalysisSolverOptions::MaxwellHarmonic(_))
            )
            || spec.id.starts_with("electro_thermal_joule_"))
}

fn configure_mesh_backed_electro_thermal_model(spec: &FixtureSpec, model: &mut AnalysisModel) {
    if !spec.id.starts_with("electro_thermal_joule_") {
        return;
    }
    replace_fixture_excitations(
        model,
        vec![FixtureExcitation {
            excitation_id: format!("{}_force", spec.id),
            region_id: "fixture_force_face".to_owned(),
            quantity: FixtureExcitationQuantity::Force([0.0, -100.0, 0.0]),
        }],
    );
    model.boundaries.clear();
    model.constraints = vec![fixed_constraint(
        format!("{}_fixed", spec.id),
        "fixture_fixed_face",
    )];
}

pub(super) fn run_fixture(
    spec: &FixtureSpec,
    filesystem_root: Option<&PathBuf>,
) -> FixtureRunRecord {
    // Fixture runs are numerical contract tests: they run under the explicit finite test
    // admission the runtime's own contract tests use, never a signed-in lease.
    let _fea_admission = crate::fea_admission_support::guard();
    let mut model = (spec.model)();
    configure_model_for_fixture(spec.id, &mut model);
    configure_mesh_backed_electro_thermal_model(spec, &mut model);
    let fixture_store_root = filesystem_root.map(|root| root.join(spec.id));
    let filesystem_root = fixture_store_root.as_ref();
    if let Some(root) = filesystem_root {
        runmat_runtime::analysis::storage::configure_artifact_store(
            runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                root: root.clone(),
            },
        )
        .expect("configure isolated fixture artifact store");
    }
    let solver_mesh_artifact_path = if fixture_requires_solver_mesh(spec, &model) {
        filesystem_root.map(|root| {
            write_fixture_solver_mesh_artifact(root, &model)
                .expect("write canonical fixture solver mesh artifact")
        })
    } else {
        None
    };
    let solver_mesh_artifact_path = solver_mesh_artifact_path.as_deref().and_then(Path::to_str);
    let mut failures = Vec::new();

    let validate_start = Instant::now();
    let validate_result = analysis_validate(
        &model,
        &runmat_analysis_core::CoordinateFrame::Global,
        OperationContext::new(Some(format!("trace-validate-{}", spec.id)), None),
    );
    let _validate_ms = validate_start.elapsed().as_secs_f64() * 1_000.0;

    let mut validate_ok = false;
    let mut validate_error_code = None;

    match validate_result {
        Ok(_) => {
            validate_ok = true;
            if let Some(expected) = spec.expect_validate_error {
                failures.push(format!(
                    "expected validate error code {expected}, but validate succeeded"
                ));
            }
        }
        Err(err) => {
            validate_error_code = Some(err.error_code.clone());
            if let Some(expected) = spec.expect_validate_error {
                if err.error_code != expected {
                    failures.push(format!(
                        "validate error mismatch: expected {expected}, got {}",
                        err.error_code
                    ));
                }
            } else {
                failures.push(format!(
                    "unexpected validate error code {} for fixture {}",
                    err.error_code, spec.id
                ));
            }
        }
    }

    let mut run_ok = false;
    let mut run_error_code = None;
    let mut cpu_run_ms = None;
    let mut gpu_run_ms = None;
    let mut gpu_fallback_events = Vec::new();
    let mut published_primary_field_storage = None;
    let mut gpu_solver_host_sync_count = None;
    let mut gpu_solver_device_apply_k_ratio = None;
    let mut gpu_speedup_ratio = None;
    let mut gpu_solver_backend = None;
    let mut gpu_solver_prepared_build_ms = None;
    let mut gpu_solver_solve_ms = None;
    let mut gpu_solver_fallback_apply_count = None;
    let mut thermo_coupling_enabled = None;
    let mut thermo_coupling_fingerprint = None;
    let mut thermo_constitutive_temperature_factor = None;
    let mut thermo_effective_modulus_scale = None;
    let mut thermo_constitutive_material_spread_ratio = None;
    let mut thermo_assignment_heterogeneity_index = None;
    let mut thermo_region_delta_count = None;
    let mut thermo_spatial_coverage_ratio = None;
    let mut thermo_nonlinear_severity = None;
    let mut electro_thermal_coupling_enabled = None;
    let mut electro_thermal_coupling_fingerprint = None;
    let mut electro_joule_heating_scale = None;
    let mut electro_conductivity_spread_ratio = None;
    let mut electro_transient_severity = None;
    let mut electro_nonlinear_severity = None;
    let mut plastic_nonlinear_severity = None;
    let mut plastic_nonlinear_severity_mean = None;
    let mut plastic_load_realization_ratio = None;
    let mut plastic_load_amplification_ratio = None;
    let mut contact_nonlinear_severity = None;
    let mut contact_nonlinear_severity_mean = None;
    let mut contact_load_realization_ratio = None;
    let mut contact_load_amplification_ratio = None;
    let mut thermal_max_relative_residual_norm = None;
    let mut thermal_min_temperature_k = None;
    let mut thermal_max_temperature_k = None;
    let mut thermal_conductivity_spread_ratio = None;
    let mut thermal_heat_capacity_spread_ratio = None;
    let mut electromagnetic_enabled = None;
    let mut electromagnetic_formulation_coverage_ratio = None;
    let mut electromagnetic_magnetostatic_curl_curl_coverage_ratio = None;
    let mut electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio = None;
    let mut electromagnetic_full_wave_displacement_current_coverage_ratio = None;
    let mut electromagnetic_displacement_to_conduction_ratio = None;
    let mut electromagnetic_material_frequency_response_coverage_ratio = None;
    let mut electromagnetic_reference_frequency_hz = None;
    let mut electromagnetic_applied_current_a = None;
    let mut electromagnetic_solve_quality = None;
    let mut electromagnetic_conductivity_spread_ratio = None;
    let mut electromagnetic_relative_permittivity_spread_ratio = None;
    let mut electromagnetic_relative_permeability_spread_ratio = None;
    let mut electromagnetic_material_heterogeneity_index = None;
    let mut electromagnetic_assignment_coverage_ratio = None;
    let mut electromagnetic_assigned_coefficient_coverage_ratio = None;
    let mut electromagnetic_region_coefficient_contrast_index = None;
    let mut electromagnetic_condition_number_estimate = None;
    let mut electromagnetic_source_realization_ratio = None;
    let mut electromagnetic_source_region_coverage_ratio = None;
    let mut electromagnetic_source_material_alignment_ratio = None;
    let mut electromagnetic_source_localization_ratio = None;
    let mut electromagnetic_source_overlap_ratio = None;
    let mut electromagnetic_source_interference_index = None;
    let mut electromagnetic_boundary_anchor_ratio = None;
    let mut electromagnetic_boundary_condition_localization_ratio = None;
    let mut electromagnetic_ground_anchor_effectiveness_ratio = None;
    let mut electromagnetic_insulation_leakage_ratio = None;
    let mut electromagnetic_flux_divergence_ratio = None;
    let mut electromagnetic_energy_imbalance_ratio = None;
    let mut electromagnetic_boundary_energy_ratio = None;
    let mut electromagnetic_boundary_penalty_conditioning_contribution = None;
    let mut electromagnetic_source_region_energy_consistency_ratio = None;
    let mut electromagnetic_real_residual_norm = None;
    let mut electromagnetic_imag_residual_norm = None;
    let mut electromagnetic_sweep_count = None;
    let mut electromagnetic_resonance_peak_frequency_hz = None;
    let mut electromagnetic_resonance_peak_flux_density = None;
    let mut electromagnetic_resonance_bandwidth_hz = None;
    let mut electromagnetic_resonance_quality_factor = None;
    let mut electromagnetic_resonance_flux_gain = None;
    let mut numerically_acceptable = None;
    let mut parity = None;
    let mut threshold_assertions = Vec::new();

    if spec.expect_validate_error.is_none() && spec.expect_run_error.is_none() {
        let cpu_start = Instant::now();
        let cpu_result = run_fixture_cpu(spec, &model, solver_mesh_artifact_path);
        cpu_run_ms = Some(cpu_start.elapsed().as_secs_f64() * 1_000.0);

        let cpu_envelope = match cpu_result {
            Ok(value) => value,
            Err(err) => {
                failures.push(format!(
                    "unexpected CPU run failure for fixture {}: {}: {}",
                    spec.id, err.error_code, err.message
                ));
                return FixtureRunRecord {
                    fixture_id: spec.id.to_string(),
                    validate_ok,
                    validate_error_code,
                    run_ok,
                    run_error_code,
                    cpu_run_ms,
                    gpu_run_ms,
                    gpu_fallback_events,
                    published_primary_field_storage,
                    gpu_solver_host_sync_count,
                    gpu_solver_device_apply_k_ratio,
                    gpu_speedup_ratio,
                    gpu_solver_backend,
                    gpu_solver_prepared_build_ms,
                    gpu_solver_solve_ms,
                    gpu_solver_fallback_apply_count,
                    thermo_coupling_enabled,
                    thermo_coupling_fingerprint,
                    thermo_constitutive_temperature_factor,
                    thermo_effective_modulus_scale,
                    thermo_constitutive_material_spread_ratio,
                    thermo_assignment_heterogeneity_index,
                    thermo_region_delta_count,
                    thermo_spatial_coverage_ratio,
                    thermo_nonlinear_severity,
                    electro_thermal_coupling_enabled,
                    electro_thermal_coupling_fingerprint,
                    electro_joule_heating_scale,
                    electro_conductivity_spread_ratio,
                    electro_transient_severity,
                    electro_nonlinear_severity,
                    plastic_nonlinear_severity,
                    plastic_nonlinear_severity_mean,
                    plastic_load_realization_ratio,
                    plastic_load_amplification_ratio,
                    contact_nonlinear_severity,
                    contact_nonlinear_severity_mean,
                    contact_load_realization_ratio,
                    contact_load_amplification_ratio,
                    thermal_max_relative_residual_norm,
                    thermal_min_temperature_k,
                    thermal_max_temperature_k,
                    thermal_conductivity_spread_ratio,
                    thermal_heat_capacity_spread_ratio,
                    electromagnetic_enabled,
                    electromagnetic_formulation_coverage_ratio,
                    electromagnetic_magnetostatic_curl_curl_coverage_ratio,
                    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
                    electromagnetic_full_wave_displacement_current_coverage_ratio,
                    electromagnetic_displacement_to_conduction_ratio,
                    electromagnetic_material_frequency_response_coverage_ratio,
                    electromagnetic_reference_frequency_hz,
                    electromagnetic_applied_current_a,
                    electromagnetic_solve_quality,
                    electromagnetic_conductivity_spread_ratio,
                    electromagnetic_relative_permittivity_spread_ratio,
                    electromagnetic_relative_permeability_spread_ratio,
                    electromagnetic_material_heterogeneity_index,
                    electromagnetic_assignment_coverage_ratio,
                    electromagnetic_assigned_coefficient_coverage_ratio,
                    electromagnetic_region_coefficient_contrast_index,
                    electromagnetic_condition_number_estimate,
                    electromagnetic_source_realization_ratio,
                    electromagnetic_source_region_coverage_ratio,
                    electromagnetic_source_material_alignment_ratio,
                    electromagnetic_source_localization_ratio,
                    electromagnetic_source_overlap_ratio,
                    electromagnetic_source_interference_index,
                    electromagnetic_boundary_anchor_ratio,
                    electromagnetic_boundary_condition_localization_ratio,
                    electromagnetic_ground_anchor_effectiveness_ratio,
                    electromagnetic_insulation_leakage_ratio,
                    electromagnetic_flux_divergence_ratio,
                    electromagnetic_energy_imbalance_ratio,
                    electromagnetic_boundary_energy_ratio,
                    electromagnetic_boundary_penalty_conditioning_contribution,
                    electromagnetic_source_region_energy_consistency_ratio,
                    electromagnetic_real_residual_norm,
                    electromagnetic_imag_residual_norm,
                    electromagnetic_sweep_count,
                    electromagnetic_resonance_peak_frequency_hz,
                    electromagnetic_resonance_peak_flux_density,
                    electromagnetic_resonance_bandwidth_hz,
                    electromagnetic_resonance_quality_factor,
                    electromagnetic_resonance_flux_gain,
                    numerically_acceptable,
                    parity,
                    threshold_assertions,
                    failures,
                };
            }
        };
        run_ok = true;
        numerically_acceptable = Some(cpu_envelope.data.numerically_acceptable);
        if let Some(expected_numerically_acceptable) = spec.expected_numerically_acceptable {
            if cpu_envelope.data.numerically_acceptable != expected_numerically_acceptable {
                failures.push(format!(
                    "cpu numerical acceptance mismatch: expected {expected_numerically_acceptable}, got {}",
                    cpu_envelope.data.numerically_acceptable
                ));
            }
        }

        if matches!(
            AnalysisSolverOptions::defaults_for(fixture_procedure_step(&model)),
            Ok(AnalysisSolverOptions::StructuralModes(_))
        ) {
            let observed = cpu_envelope
                .data
                .modal_results
                .as_ref()
                .and_then(|modal| modal.residual_norms().into_iter().reduce(f64::max));
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_max_residual_norm",
                "FEA_MODAL_CONVERGENCE",
                observed,
                None,
                Some(1.0e-1),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_adjacent_mode_pair_count",
                "FEA_MODAL_CLUSTER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_MODAL_CLUSTER",
                    "adjacent_mode_pair_count",
                ),
                spec.modal_mode_count
                    .and_then(|mode_count| mode_count.checked_sub(1))
                    .map(|count| count as f64),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_cluster_coverage_ratio",
                "FEA_MODAL_CLUSTER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_MODAL_CLUSTER",
                    "cluster_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
        }
        if let Some(max_offdiag) = spec.max_modal_orthogonality_offdiag {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_ORTHOGONALITY",
                "max_m_orthogonality_offdiag",
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_max_m_orthogonality_offdiag",
                "FEA_MODAL_ORTHOGONALITY",
                observed,
                None,
                Some(max_offdiag),
            );
        }
        if let Some(min_separation) = spec.min_modal_relative_frequency_separation {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_SEPARATION",
                "min_relative_frequency_separation",
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_min_relative_frequency_separation",
                "FEA_MODAL_SEPARATION",
                observed,
                Some(min_separation),
                None,
            );
        }
        if spec.id.starts_with("acoustic_harmonic_") {
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_normalized_residual_norm",
                "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL",
                    "normalized_residual_norm",
                ),
                None,
                Some(1.0e-3),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_drive_frequency_hz",
                "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                    "drive_frequency_hz",
                ),
                Some(50.0),
                Some(20_000.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_peak_pressure_pa",
                "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                    "peak_pressure_pa",
                ),
                Some(1.0e-12),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_domain_node_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "domain_node_count",
                ),
                Some(3.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_domain_edge_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "domain_edge_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_domain_active_dimension_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "domain_active_dimension_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_boundary_node_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "boundary_node_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_material_coverage_ratio",
                "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                    "acoustic_material_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_boundary_coverage_ratio",
                "FEA_ACOUSTIC_BOUNDARY_MODEL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_BOUNDARY_MODEL",
                    "acoustic_boundary_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_radiation_boundary_count",
                "FEA_ACOUSTIC_BOUNDARY_MODEL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_BOUNDARY_MODEL",
                    "radiation_boundary_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_impedance_boundary_count",
                "FEA_ACOUSTIC_BOUNDARY_MODEL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_BOUNDARY_MODEL",
                    "impedance_boundary_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_frequency_response_sweep_count",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_count",
                ),
                Some(3.0),
                Some(3.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_frequency_response_coverage_ratio",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "response_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_sweep_bandwidth_hz",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_bandwidth_hz",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_sweep_peak_pressure_pa",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_peak_pressure_pa",
                ),
                Some(1.0e-12),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_sweep_max_residual_norm",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_max_residual_norm",
                ),
                None,
                Some(1.0e-3),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_tube_mode_alignment_error_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "tube_mode_alignment_error_ratio",
                ),
                Some(0.0),
                Some(0.5),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_tube_pressure_variation_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "tube_pressure_variation_ratio",
                ),
                Some(1.0e-12),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_cavity_mode_spacing_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "cavity_mode_spacing_ratio",
                ),
                Some(1.0e-9),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_cavity_reference_mode_count",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "cavity_reference_mode_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_known_answer_coverage_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
        }
        if matches!(
            spec.id,
            "cantilever_gpu_provider"
                | "cantilever_gpu_fallback"
                | "cantilever_load_sweep_gpu_provider"
                | "cantilever_large_load_sweep_gpu_provider"
                | "structural_axial_bar_reference_gpu_provider"
                | "structural_beam_bending_reference_gpu_provider"
        ) {
            push_linear_structural_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }
        if matches!(
            spec.id,
            "structural_axial_bar_reference_gpu_provider"
                | "structural_beam_bending_reference_gpu_provider"
        ) {
            push_structural_reference_kinematics_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }
        if matches!(
            spec.id,
            "structural_beam_cantilever_end_moment_reference_cpu"
                | "structural_beam_torsion_reference_cpu"
                | "structural_beam_force_and_moment_reference_cpu"
        ) {
            push_structural_beam_moment_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }
        if spec.id == "structural_shell_plate_moment_reference_cpu" {
            push_structural_shell_moment_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }
        if spec.id.starts_with("transient_") {
            push_fixed_step_transient_assertions(
                spec.id,
                "cpu",
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
                spec.transient_step_count,
                spec.max_transient_residual_norm,
            );
        }
        if spec.id.starts_with("cht_coupled_") {
            let expected_cht_sample_count = 12.0;
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_reference_density_kg_per_m3",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "density"),
                Some(1.20),
                Some(1.25),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_dynamic_viscosity_pa_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "viscosity"),
                Some(1.0e-5),
                Some(3.0e-5),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_inlet_velocity_m_per_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "inlet_velocity"),
                Some(4.0),
                Some(6.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_turbulence_intensity",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "turbulence_intensity"),
                Some(0.04),
                Some(0.08),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_reynolds_number",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "reynolds_number"),
                Some(2.0e5),
                Some(5.0e5),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_executed_sample_count",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "executed_sample_count"),
                Some(expected_cht_sample_count),
                Some(expected_cht_sample_count),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_applied_temperature_delta_k",
                "FEA_CHT_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_COUPLING",
                    "applied_temperature_delta_k",
                ),
                Some(60.0),
                Some(60.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_sample_count",
                "FEA_CHT_COUPLING",
                diagnostic_metric(&cpu_envelope.data, "FEA_CHT_COUPLING", "sample_count"),
                Some(12.0),
                Some(12.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_nominal_time_step_s",
                "FEA_CHT_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_COUPLING",
                    "nominal_time_step_s",
                ),
                Some(1.0 / 11.0),
                Some(1.0 / 11.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_momentum_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_momentum_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_continuity_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_continuity_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_thermal_residual",
                "FEA_THERMAL_STABILITY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_THERMAL_STABILITY",
                    "max_relative_residual_norm",
                ),
                None,
                Some(2.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_face_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_face_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_temperature_jump_k",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "max_temperature_jump_k",
                ),
                Some(0.0),
                Some(0.1),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_energy_residual",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "max_energy_residual",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_heat_flux_balance_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "heat_flux_balance_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_transport_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_transport_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_temperature_continuity_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_temperature_continuity_ratio",
                ),
                Some(0.999),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_advection_temperature_shift_k",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "max_advection_temperature_shift_k",
                ),
                Some(0.0),
                Some(0.1),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_conductance_w_per_m2k",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_conductance_w_per_m2k",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_flux_temperature_law_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "flux_temperature_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_heat_flux_realization_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "heat_flux_realization_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_coupled_interface_iteration_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "coupled_interface_iteration_count",
                ),
                Some(1.0),
                Some(64.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_coupled_interface_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "coupled_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_network_node_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_network_node_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_network_edge_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_network_edge_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_connectivity_coverage_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_connectivity_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            let cht_mesh_backed_min = if matches!(
                spec.id,
                "cht_coupled_gpu_provider" | "cht_coupled_gpu_fallback"
            ) {
                1.0
            } else {
                0.0
            };
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_mesh_backed_interface_connectivity_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "mesh_backed_interface_connectivity_ratio",
                ),
                Some(cht_mesh_backed_min),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_full_topology_edge_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "full_topology_edge_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_full_topology_element_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "full_topology_element_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_network_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_network_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_heated_channel_energy_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "heated_channel_energy_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_conjugate_slab_flux_law_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "conjugate_slab_flux_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_coupled_interface_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "coupled_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_heat_flux_realization_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "heat_flux_realization_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_thermal_network_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "thermal_network_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_interface_connectivity_coverage_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "interface_connectivity_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_interface_temperature_continuity_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "interface_temperature_continuity_ratio",
                ),
                Some(0.999),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_advection_shift_coverage_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "advection_shift_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_coverage_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            if spec.id == "cht_coupled_channel_slab_cpu" {
                push_threshold_assertion!(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "cht_authored_interface_count",
                    "FEA_CHT_COUPLING",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_CHT_COUPLING",
                        "authored_interface_count",
                    ),
                    Some(1.0),
                    Some(1.0),
                );
            }
        }
        if spec.id.starts_with("fsi_coupled_") {
            let expected_fsi_sample_count = 12.0;
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_momentum_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_momentum_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_continuity_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_continuity_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_interface_residual",
                "FEA_FSI_INTERFACE_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_RESIDUAL",
                    "max_interface_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_node_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_node_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_force_balance_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "force_balance_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_displacement_transfer_residual_m",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "max_displacement_transfer_residual_m",
                ),
                Some(0.0),
                Some(1.0e-12),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_coupling_iteration_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "max_coupling_iteration_count",
                ),
                Some(1.0),
                Some(128.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_pressure_feedback_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "pressure_feedback_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_two_way_interface_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "two_way_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_traction_update_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_traction_update_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_pressure_displacement_law_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "pressure_displacement_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_solve_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_solve_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_work_j_per_m2",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_work_j_per_m2",
                ),
                Some(1.0e-18),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_strain_energy_j_per_m2",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_strain_energy_j_per_m2",
                ),
                Some(1.0e-18),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_work_energy_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_work_energy_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_coupling_edge_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_coupling_edge_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_connectivity_coverage_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_connectivity_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            let fsi_mesh_backed_min = if matches!(
                spec.id,
                "fsi_coupled_gpu_provider" | "fsi_coupled_gpu_fallback"
            ) {
                1.0
            } else {
                0.0
            };
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_mesh_backed_interface_connectivity_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "mesh_backed_interface_connectivity_ratio",
                ),
                Some(fsi_mesh_backed_min),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_full_topology_edge_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "full_topology_edge_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_full_topology_element_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "full_topology_element_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_stiffness_pa_per_m",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_stiffness_pa_per_m",
                ),
                Some(1.0e6),
                None,
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_pressure_loaded_wall_displacement_law_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "pressure_loaded_wall_displacement_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_traction_balance_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_traction_balance_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_displacement_transfer_residual_m",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_displacement_transfer_residual_m",
                ),
                Some(0.0),
                Some(1.0e-12),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_partitioned_pressure_feedback_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "partitioned_pressure_feedback_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_two_way_interface_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "two_way_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_structural_traction_update_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "structural_traction_update_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_structural_solve_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "structural_solve_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_interface_connectivity_coverage_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_connectivity_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_interface_work_energy_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_work_energy_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_coverage_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_reference_density_kg_per_m3",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "density"),
                Some(1.20),
                Some(1.25),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_dynamic_viscosity_pa_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "viscosity"),
                Some(1.0e-5),
                Some(3.0e-5),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_inlet_velocity_m_per_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "inlet_velocity"),
                Some(3.5),
                Some(4.5),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_turbulence_intensity",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "turbulence_intensity"),
                Some(0.04),
                Some(0.08),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_reynolds_number",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "reynolds_number"),
                Some(2.0e5),
                Some(3.0e5),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_executed_sample_count",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "executed_sample_count"),
                Some(expected_fsi_sample_count),
                Some(expected_fsi_sample_count),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_sample_count",
                "FEA_FSI_COUPLING",
                diagnostic_metric(&cpu_envelope.data, "FEA_FSI_COUPLING", "sample_count"),
                Some(12.0),
                Some(12.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_nominal_time_step_s",
                "FEA_FSI_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_COUPLING",
                    "nominal_time_step_s",
                ),
                Some(1.0 / 11.0),
                Some(1.0 / 11.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_procedure_count",
                "FEA_FSI_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_COUPLING",
                    "fsi_procedure_count",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion!(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_cfd_executed_sample_count",
                "FEA_FSI_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_COUPLING",
                    "cfd_executed_sample_count",
                ),
                Some(expected_fsi_sample_count),
                Some(expected_fsi_sample_count),
            );
            if spec.id == "fsi_coupled_pipe_plate_cpu" {
                push_threshold_assertion!(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "fsi_authored_interface_count",
                    "FEA_FSI_COUPLING",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_FSI_COUPLING",
                        "authored_interface_count",
                    ),
                    Some(1.0),
                    Some(1.0),
                );
            }
        }
        if spec.id == "thermal_standalone_ramp_cpu" {
            push_thermal_standalone_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }

        if let Some(gpu_mode) = spec.gpu_mode {
            let gpu_start = Instant::now();
            let gpu_result = run_fixture_gpu(spec, &model, gpu_mode, solver_mesh_artifact_path);
            gpu_run_ms = Some(gpu_start.elapsed().as_secs_f64() * 1_000.0);

            match gpu_result {
                Ok(gpu_envelope) => {
                    run_ok = true;
                    numerically_acceptable = Some(gpu_envelope.data.numerically_acceptable);
                    gpu_fallback_events = gpu_envelope.data.provenance.fallback_events.clone();
                    gpu_solver_host_sync_count =
                        Some(gpu_envelope.data.provenance.solver_host_sync_count);
                    gpu_solver_device_apply_k_ratio =
                        Some(gpu_envelope.data.provenance.solver_device_apply_k_ratio);
                    gpu_solver_backend = Some(
                        gpu_envelope
                            .data
                            .provenance
                            .solver_backend
                            .as_str()
                            .to_string(),
                    );
                    gpu_speedup_ratio = match (cpu_run_ms, gpu_run_ms) {
                        (Some(cpu_ms), Some(gpu_ms)) if gpu_ms > 0.0 => Some(cpu_ms / gpu_ms),
                        _ => None,
                    };
                    gpu_solver_prepared_build_ms = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "prepared_build_ms",
                    )
                    .or_else(|| {
                        diagnostic_metric(&gpu_envelope.data, "FEA_EM_COST", "prepared_build_ms")
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "prepared_build_ms",
                        )
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_NONLINEAR_COST",
                            "prepared_build_ms",
                        )
                    });
                    gpu_solver_solve_ms =
                        diagnostic_metric(&gpu_envelope.data, "FEA_MODAL_COST", "solve_ms")
                            .or_else(|| {
                                diagnostic_metric(&gpu_envelope.data, "FEA_EM_COST", "solve_ms")
                            })
                            .or_else(|| {
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_TRANSIENT_COST",
                                    "solve_ms",
                                )
                            })
                            .or_else(|| {
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_NONLINEAR_COST",
                                    "solve_ms",
                                )
                            })
                            .or(gpu_run_ms);
                    gpu_solver_fallback_apply_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "fallback_apply_count",
                    )
                    .or_else(|| {
                        diagnostic_metric(&gpu_envelope.data, "FEA_EM_COST", "fallback_apply_count")
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "fallback_apply_count",
                        )
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_NONLINEAR_COST",
                            "fallback_apply_count",
                        )
                    });
                    for event in &gpu_fallback_events {
                        if !validate_fallback_event_schema(event) {
                            failures.push(format!("invalid fallback event schema: {event:?}"));
                        }
                    }

                    let gpu_primary_field_id =
                        primary_result_field_id(fixture_procedure_step(&model));
                    if let Some(primary_field) =
                        analysis_result_field(&gpu_envelope.data, &gpu_primary_field_id)
                    {
                        published_primary_field_storage = Some(match &primary_field.values {
                            AnalysisFieldValues::DeviceRef(_) => "device_ref".to_string(),
                            AnalysisFieldValues::HostF64(_) => "host_f64".to_string(),
                        });
                    } else {
                        failures.push(format!(
                            "primary field {gpu_primary_field_id} should be present for gpu fixture {}",
                            spec.id
                        ));
                    }

                    let gpu_expected_numerically_acceptable =
                        if spec.id.starts_with("acoustic_harmonic_") {
                            Some(false)
                        } else {
                            spec.expected_numerically_acceptable
                        };
                    if let Some(expected_numerically_acceptable) =
                        gpu_expected_numerically_acceptable
                    {
                        if gpu_envelope.data.numerically_acceptable
                            != expected_numerically_acceptable
                        {
                            failures.push(format!(
                                "gpu numerical acceptance mismatch: expected {expected_numerically_acceptable}, got {}",
                                gpu_envelope.data.numerically_acceptable
                            ));
                        }
                    }
                    if spec.id.starts_with("transient_") {
                        push_fixed_step_transient_assertions(
                            spec.id,
                            "gpu",
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            spec.transient_step_count,
                            spec.max_transient_residual_norm,
                        );
                    }

                    if let Some(expected_solver_backend) = spec.expected_solver_backend {
                        if gpu_envelope.data.provenance.solver_backend.as_str()
                            != expected_solver_backend
                        {
                            failures.push(format!(
                                "solver_backend mismatch for fixture {}: expected={} got={}",
                                spec.id,
                                expected_solver_backend,
                                gpu_envelope.data.provenance.solver_backend
                            ));
                        }
                    }
                    // Shared accelerator admission: a GPU placement that ran on cpu_reference
                    // publishes the solver-backend fallback event and exactly one fallback
                    // diagnostic, whatever the physics family.
                    if gpu_envelope.data.provenance.solver_backend.as_str() == "cpu_reference" {
                        let fallback_diagnostics = gpu_envelope
                            .data
                            .run
                            .diagnostics
                            .iter()
                            .filter(|diagnostic| {
                                diagnostic.code == "FEA_ACCELERATOR_ADMISSION_FALLBACK"
                            })
                            .count();
                        if fallback_diagnostics != 1 {
                            failures.push(format!(
                                "fixture {} must report FEA_ACCELERATOR_ADMISSION_FALLBACK once for a GPU placement that ran on cpu_reference; found {fallback_diagnostics}",
                                spec.id
                            ));
                        }
                        if !gpu_fallback_events
                            .iter()
                            .any(|event| event.stage == FallbackStage::Solver)
                        {
                            failures.push(format!(
                                "fixture {} must publish a solver-backend fallback event for a GPU placement that ran on cpu_reference",
                                spec.id
                            ));
                        }
                    }
                    if env_bool("RUNMAT_FEA_ENFORCE_SPEEDUP_GATES").unwrap_or(false) {
                        if let Some(min_speedup_ratio) = spec.min_gpu_speedup_ratio {
                            let observed = gpu_speedup_ratio.unwrap_or(0.0);
                            if observed < min_speedup_ratio {
                                failures.push(format!(
                                    "gpu speedup ratio below target for fixture {}: observed={} min={}",
                                    spec.id, observed, min_speedup_ratio
                                ));
                            }
                        }
                    }
                    if spec.id.starts_with("thermal_standalone_ramp_") {
                        push_thermal_standalone_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                        );
                    }
                    if spec.id.starts_with("electromagnetic_reference_") {
                        push_em_reference_topology_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_formulation_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "formulation_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_magnetostatic_curl_curl_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "magnetostatic_curl_curl_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "magnetoquasistatic_eddy_current_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_full_wave_displacement_current_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "full_wave_displacement_current_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_material_frequency_response_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "material_frequency_response_coverage_ratio",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_displacement_to_conduction_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "displacement_to_conduction_ratio",
                            ),
                            Some(0.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_source_energy_diagnostic_coverage_ratio",
                            "FEA_EM_SOURCE_ENERGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SOURCE_ENERGY",
                                "field_energy_integral",
                            )
                            .filter(|value| value.is_finite())
                            .map(|_| 1.0),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_source_energy_consistency_ratio",
                            "FEA_EM_SOURCE_ENERGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SOURCE_ENERGY",
                                "source_region_energy_consistency_ratio",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_source_energy_imbalance_ratio",
                            "FEA_EM_SOURCE_ENERGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SOURCE_ENERGY",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_reference_coverage_ratio",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "reference_frequency_in_sweep_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_peak_frequency_error_ratio",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "normalized_peak_frequency_error_ratio",
                            ),
                            Some(0.0),
                            Some(0.25),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_quality_factor",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "resonance_quality_factor",
                            ),
                            Some(1.5),
                            Some(1.0e9),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_answer_coverage_ratio",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "sweep_known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_homogeneous_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_sigma_omega_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_mean",
                            ),
                            Some(0.95),
                            Some(1.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_sigma_omega_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.15),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_sigma_omega_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_mean",
                            ),
                            Some(0.95),
                            Some(1.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_mean",
                            ),
                            Some(0.95),
                            Some(1.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_loss_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_mean",
                            ),
                            Some(0.0),
                            Some(0.08),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_loss_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_phase_attenuation_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_mean",
                            ),
                            Some(0.99),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_phase_attenuation_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_coupling_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_conductivity_coupling_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_phase_conductivity_attenuation_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_conductivity_attenuation_ratio",
                            ),
                            Some(0.99),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_conductivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_material_heterogeneity_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "electromagnetic_material_heterogeneity_index",
                            ),
                            Some(0.0),
                            Some(0.02),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_assigned_coefficient_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assigned_coefficient_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_flux_phasor_coherence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_phasor_coherence_ratio",
                            ),
                            Some(0.5),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_flux_divergence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_divergence_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.0),
                            Some(0.40),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_boundary_energy_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_energy_ratio",
                            ),
                            Some(0.12),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_edge_dof_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "edge_dof_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_element_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "element_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_oriented_edge_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "oriented_edge_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_incidence_element_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "incidence_element_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_incidence_orientation_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "incidence_orientation_count",
                            ),
                            Some(3.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_incidence_pair_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "incidence_pair_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_incidence_operator_pair_coverage_ratio",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "incidence_operator_pair_coverage_ratio",
                            ),
                            Some(0.5),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_gauge_anchor_count",
                            "FEA_EM_GAUGE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_GAUGE",
                                "gauge_anchor_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_gauge_anchor_residual_ratio",
                            "FEA_EM_GAUGE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_GAUGE",
                                "gauge_anchor_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-9),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_material_residual_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "homogeneous_material_residual_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_source_energy_consistency_residual_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "source_energy_consistency_residual_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_gauge_anchor_residual_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "gauge_anchor_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-9),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_flux_divergence_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "flux_divergence_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_answer_coverage_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_heterogeneous_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_sigma_omega_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_sigma_omega_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.2),
                            Some(2.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_sigma_omega_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.1),
                            Some(5.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.5),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_spread_ratio",
                            ),
                            Some(1.1),
                            Some(8.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_loss_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_mean",
                            ),
                            Some(0.06),
                            Some(0.22),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_loss_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(20.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_phase_attenuation_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_mean",
                            ),
                            Some(0.9),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_phase_attenuation_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.2),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_coupling_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_conductivity_coupling_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_phase_conductivity_attenuation_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_conductivity_attenuation_ratio",
                            ),
                            Some(0.9),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_conductivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_spread_ratio",
                            ),
                            Some(10.0),
                            Some(1.0e9),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_spread_ratio",
                            ),
                            Some(2.0),
                            Some(1.0e9),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_spread_ratio",
                            ),
                            Some(5.0),
                            Some(1.0e12),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_material_heterogeneity_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "electromagnetic_material_heterogeneity_index",
                            ),
                            Some(0.25),
                            Some(2.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_region_contrast_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "region_coefficient_contrast_index",
                            ),
                            Some(1.0),
                            Some(10.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(0.6),
                            Some(0.8),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.7),
                            Some(0.85),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_flux_phasor_coherence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_phasor_coherence_ratio",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_flux_divergence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_divergence_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.1),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_sparse_assignments_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_assigned_coefficient_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assigned_coefficient_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.2),
                            Some(0.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                    }
                    if spec.id == "electromagnetic_reference_multiregion_assignments_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_assigned_coefficient_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assigned_coefficient_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                    }
                    if spec.id == "electromagnetic_reference_overlap_interference_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_overlap_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_overlap_ratio",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_interference_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_interference_index",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_boundary_kernel_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_known_answer_coverage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_boundary_localization_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_condition_localization_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_ground_selector_coverage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "ground_anchor_effectiveness_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_insulation_leakage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "insulation_leakage_ratio",
                            ),
                            Some(0.8),
                            Some(2.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_boundary_penalty_stress_gpu_provider" {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_known_answer_coverage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_strong_constraint_penalty_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_penalty_conditioning_contribution",
                            ),
                            Some(0.0),
                            Some(0.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_anchor_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.75),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_real_residual_norm",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "real_residual_norm",
                            ),
                            Some(0.0),
                            Some(0.95),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_imag_residual_norm",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "imag_residual_norm",
                            ),
                            Some(0.0),
                            Some(0.95),
                        );
                    }
                    if spec.id
                        == "electromagnetic_reference_multi_region_phased_source_gpu_provider"
                    {
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_energy_consistency_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_energy_consistency_ratio",
                            ),
                            Some(0.2),
                            Some(0.95),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_overlap_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_overlap_ratio",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion!(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_interference_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_interference_index",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                    }

                    let gpu_primary_field_id =
                        primary_result_field_id(fixture_procedure_step(&model));
                    let gpu_results = analysis_results_op(
                        &gpu_envelope.data,
                        AnalysisResultsQuery {
                            include_fields: vec![gpu_primary_field_id.clone()],
                            include_field_values: true,

                            include_diagnostics: false,
                            diagnostic_codes: Vec::new(),
                            include_modal_results: false,
                            include_buckling_results: false,
                            mode_indices: Vec::new(),
                            include_transient_results: false,
                            transient_snapshot_indices: Vec::new(),
                            include_nonlinear_results: false,
                        },
                        OperationContext::new(Some(format!("trace-results-gpu-{}", spec.id)), None),
                    );
                    let gpu_results = match gpu_results {
                        Ok(value) => value,
                        Err(err) => {
                            failures.push(format!(
                                "fea.results failed for gpu fixture {}: {}",
                                spec.id, err.error_code
                            ));
                            return FixtureRunRecord {
                                fixture_id: spec.id.to_string(),
                                validate_ok,
                                validate_error_code,
                                run_ok,
                                run_error_code,
                                cpu_run_ms,
                                gpu_run_ms,
                                gpu_fallback_events,
                                published_primary_field_storage,
                                gpu_solver_host_sync_count,
                                gpu_solver_device_apply_k_ratio,
                                gpu_speedup_ratio,
                                gpu_solver_backend,
                                gpu_solver_prepared_build_ms,
                                gpu_solver_solve_ms,
                                gpu_solver_fallback_apply_count,
                                thermo_coupling_enabled,
                                thermo_coupling_fingerprint,
                                thermo_constitutive_temperature_factor,
                                thermo_effective_modulus_scale,
                                thermo_constitutive_material_spread_ratio,
                                thermo_assignment_heterogeneity_index,
                                thermo_region_delta_count,
                                thermo_spatial_coverage_ratio,
                                thermo_nonlinear_severity,
                                electro_thermal_coupling_enabled,
                                electro_thermal_coupling_fingerprint,
                                electro_joule_heating_scale,
                                electro_conductivity_spread_ratio,
                                electro_transient_severity,
                                electro_nonlinear_severity,
                                plastic_nonlinear_severity,
                                plastic_nonlinear_severity_mean,
                                plastic_load_realization_ratio,
                                plastic_load_amplification_ratio,
                                contact_nonlinear_severity,
                                contact_nonlinear_severity_mean,
                                contact_load_realization_ratio,
                                contact_load_amplification_ratio,
                                thermal_max_relative_residual_norm,
                                thermal_min_temperature_k,
                                thermal_max_temperature_k,
                                thermal_conductivity_spread_ratio,
                                thermal_heat_capacity_spread_ratio,
                                electromagnetic_enabled,
                                electromagnetic_formulation_coverage_ratio,
                                electromagnetic_magnetostatic_curl_curl_coverage_ratio,
                                electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
                                electromagnetic_full_wave_displacement_current_coverage_ratio,
                                electromagnetic_displacement_to_conduction_ratio,
                                electromagnetic_material_frequency_response_coverage_ratio,
                                electromagnetic_reference_frequency_hz,
                                electromagnetic_applied_current_a,
                                electromagnetic_solve_quality,
                                electromagnetic_conductivity_spread_ratio,
                                electromagnetic_relative_permittivity_spread_ratio,
                                electromagnetic_relative_permeability_spread_ratio,
                                electromagnetic_material_heterogeneity_index,
                                electromagnetic_assignment_coverage_ratio,
                                electromagnetic_assigned_coefficient_coverage_ratio,
                                electromagnetic_region_coefficient_contrast_index,
                                electromagnetic_condition_number_estimate,
                                electromagnetic_source_realization_ratio,
                                electromagnetic_source_region_coverage_ratio,
                                electromagnetic_source_material_alignment_ratio,
                                electromagnetic_source_localization_ratio,
                                electromagnetic_source_overlap_ratio,
                                electromagnetic_source_interference_index,
                                electromagnetic_boundary_anchor_ratio,
                                electromagnetic_boundary_condition_localization_ratio,
                                electromagnetic_ground_anchor_effectiveness_ratio,
                                electromagnetic_insulation_leakage_ratio,
                                electromagnetic_flux_divergence_ratio,
                                electromagnetic_energy_imbalance_ratio,
                                electromagnetic_boundary_energy_ratio,
                                electromagnetic_boundary_penalty_conditioning_contribution,
                                electromagnetic_source_region_energy_consistency_ratio,
                                electromagnetic_real_residual_norm,
                                electromagnetic_imag_residual_norm,
                                electromagnetic_sweep_count,
                                electromagnetic_resonance_peak_frequency_hz,
                                electromagnetic_resonance_peak_flux_density,
                                electromagnetic_resonance_bandwidth_hz,
                                electromagnetic_resonance_quality_factor,
                                electromagnetic_resonance_flux_gain,
                                numerically_acceptable,
                                parity,
                                threshold_assertions,
                                failures,
                            };
                        }
                    };
                    if gpu_results.operation != "fea.results"
                        || gpu_results.op_version != "fea.results/v1"
                    {
                        failures.push("fea.results contract version mismatch".to_string());
                    }
                    thermo_coupling_enabled = gpu_results.data.summary.thermo_coupling_enabled;
                    thermo_coupling_fingerprint =
                        gpu_results.data.summary.thermo_coupling_fingerprint;
                    thermo_constitutive_temperature_factor = gpu_results
                        .data
                        .summary
                        .thermo_constitutive_temperature_factor;
                    thermo_effective_modulus_scale =
                        gpu_results.data.summary.thermo_effective_modulus_scale;
                    thermo_constitutive_material_spread_ratio = gpu_results
                        .data
                        .summary
                        .thermo_constitutive_material_spread_ratio;
                    thermo_assignment_heterogeneity_index = gpu_results
                        .data
                        .summary
                        .thermo_assignment_heterogeneity_index;
                    thermo_region_delta_count = gpu_results.data.summary.thermo_region_delta_count;
                    thermo_spatial_coverage_ratio =
                        gpu_results.data.summary.thermo_spatial_coverage_ratio;
                    thermo_nonlinear_severity = gpu_results.data.summary.thermo_nonlinear_severity;
                    electro_thermal_coupling_enabled =
                        gpu_results.data.summary.electro_thermal_coupling_enabled;
                    electro_thermal_coupling_fingerprint = gpu_results
                        .data
                        .summary
                        .electro_thermal_coupling_fingerprint;
                    electro_joule_heating_scale =
                        gpu_results.data.summary.electro_joule_heating_scale;
                    electro_conductivity_spread_ratio =
                        gpu_results.data.summary.electro_conductivity_spread_ratio;
                    electro_transient_severity =
                        gpu_results.data.summary.electro_transient_severity;
                    electro_nonlinear_severity =
                        gpu_results.data.summary.electro_nonlinear_severity;
                    plastic_nonlinear_severity =
                        gpu_results.data.summary.plastic_nonlinear_severity;
                    plastic_nonlinear_severity_mean =
                        gpu_results.data.summary.plastic_nonlinear_severity_mean;
                    plastic_load_realization_ratio =
                        gpu_results.data.summary.plastic_load_realization_ratio;
                    plastic_load_amplification_ratio =
                        gpu_results.data.summary.plastic_load_amplification_ratio;
                    contact_nonlinear_severity =
                        gpu_results.data.summary.contact_nonlinear_severity;
                    contact_nonlinear_severity_mean =
                        gpu_results.data.summary.contact_nonlinear_severity_mean;
                    contact_load_realization_ratio =
                        gpu_results.data.summary.contact_load_realization_ratio;
                    contact_load_amplification_ratio =
                        gpu_results.data.summary.contact_load_amplification_ratio;
                    thermal_max_relative_residual_norm =
                        gpu_results.data.summary.thermal_max_relative_residual_norm;
                    thermal_min_temperature_k = gpu_results.data.summary.thermal_min_temperature_k;
                    thermal_max_temperature_k = gpu_results.data.summary.thermal_max_temperature_k;
                    thermal_conductivity_spread_ratio =
                        gpu_results.data.summary.thermal_conductivity_spread_ratio;
                    thermal_heat_capacity_spread_ratio =
                        gpu_results.data.summary.thermal_heat_capacity_spread_ratio;
                    electromagnetic_enabled = gpu_results.data.summary.electromagnetic_enabled;
                    electromagnetic_formulation_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_formulation_coverage_ratio;
                    electromagnetic_magnetostatic_curl_curl_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_magnetostatic_curl_curl_coverage_ratio;
                    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio;
                    electromagnetic_full_wave_displacement_current_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_full_wave_displacement_current_coverage_ratio;
                    electromagnetic_displacement_to_conduction_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_displacement_to_conduction_ratio;
                    electromagnetic_material_frequency_response_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_material_frequency_response_coverage_ratio;
                    electromagnetic_reference_frequency_hz = gpu_results
                        .data
                        .summary
                        .electromagnetic_reference_frequency_hz;
                    electromagnetic_applied_current_a =
                        gpu_results.data.summary.electromagnetic_applied_current_a;
                    electromagnetic_solve_quality =
                        gpu_results.data.summary.electromagnetic_solve_quality;
                    electromagnetic_conductivity_spread_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_conductivity_spread_ratio;
                    electromagnetic_relative_permittivity_spread_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_relative_permittivity_spread_ratio;
                    electromagnetic_relative_permeability_spread_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_relative_permeability_spread_ratio;
                    electromagnetic_material_heterogeneity_index = gpu_results
                        .data
                        .summary
                        .electromagnetic_material_heterogeneity_index;
                    electromagnetic_assignment_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_assignment_coverage_ratio;
                    electromagnetic_assigned_coefficient_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_assigned_coefficient_coverage_ratio;
                    electromagnetic_region_coefficient_contrast_index = gpu_results
                        .data
                        .summary
                        .electromagnetic_region_coefficient_contrast_index;
                    electromagnetic_condition_number_estimate = gpu_results
                        .data
                        .summary
                        .electromagnetic_condition_number_estimate;
                    electromagnetic_source_realization_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_realization_ratio;
                    electromagnetic_source_region_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_region_coverage_ratio;
                    electromagnetic_source_material_alignment_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_material_alignment_ratio;
                    electromagnetic_source_localization_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_localization_ratio;
                    electromagnetic_source_overlap_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_overlap_ratio;
                    electromagnetic_source_interference_index = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_interference_index;
                    electromagnetic_boundary_anchor_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_anchor_ratio;
                    electromagnetic_boundary_condition_localization_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_condition_localization_ratio;
                    electromagnetic_ground_anchor_effectiveness_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_ground_anchor_effectiveness_ratio;
                    electromagnetic_insulation_leakage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_insulation_leakage_ratio;
                    electromagnetic_flux_divergence_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_flux_divergence_ratio;
                    electromagnetic_energy_imbalance_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_energy_imbalance_ratio;
                    electromagnetic_boundary_energy_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_energy_ratio;
                    electromagnetic_boundary_penalty_conditioning_contribution = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_penalty_conditioning_contribution;
                    electromagnetic_source_region_energy_consistency_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_region_energy_consistency_ratio;
                    electromagnetic_real_residual_norm =
                        gpu_results.data.summary.electromagnetic_real_residual_norm;
                    electromagnetic_imag_residual_norm =
                        gpu_results.data.summary.electromagnetic_imag_residual_norm;
                    electromagnetic_sweep_count =
                        gpu_results.data.summary.electromagnetic_sweep_count;
                    electromagnetic_resonance_peak_frequency_hz = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_peak_frequency_hz;
                    electromagnetic_resonance_peak_flux_density = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_peak_flux_density;
                    electromagnetic_resonance_bandwidth_hz = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_bandwidth_hz;
                    electromagnetic_resonance_quality_factor = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_quality_factor;
                    electromagnetic_resonance_flux_gain =
                        gpu_results.data.summary.electromagnetic_resonance_flux_gain;

                    if let Some(root) = filesystem_root {
                        runmat_runtime::analysis::storage::configure_artifact_store(
                            runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                                root: root.clone(),
                            },
                        )
                        .expect("reconfigure filesystem artifact store");

                        let persisted = analysis_results_by_run_id_op(
                            &gpu_envelope.data.run_id,
                            AnalysisResultsQuery {
                                include_fields: vec![gpu_primary_field_id],
                                include_field_values: true,

                                include_diagnostics: false,
                                diagnostic_codes: Vec::new(),
                                include_modal_results: false,
                                include_buckling_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
                                include_nonlinear_results: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-gpu-by-id-{}", spec.id)),
                                None,
                            ),
                        );
                        match persisted {
                            Ok(by_id) => {
                                if by_id.operation != "fea.results"
                                    || by_id.op_version != "fea.results/v1"
                                {
                                    failures.push(
                                        "fea.results by-run-id contract mismatch".to_string(),
                                    );
                                }
                            }
                            Err(err) => failures.push(format!(
                                "fea.results by-run-id failed for fixture {}: {}",
                                spec.id, err.error_code
                            )),
                        }
                    }

                    if let Some(max_sync) = spec.max_solver_host_sync_count {
                        let observed = gpu_envelope.data.provenance.solver_host_sync_count;
                        if observed > max_sync {
                            failures.push(format!(
                                "solver_host_sync_count exceeded for fixture {}: observed={} limit={}",
                                spec.id, observed, max_sync
                            ));
                        }
                    }
                    if let Some(min_ratio) = spec.min_solver_device_apply_k_ratio {
                        let observed = gpu_envelope.data.provenance.solver_device_apply_k_ratio;
                        if observed < min_ratio {
                            failures.push(format!(
                                "solver_device_apply_k_ratio below target for fixture {}: observed={} min={}",
                                spec.id, observed, min_ratio
                            ));
                        }
                    }

                    if let Some(expectation) = spec.published_field_storage_expectation {
                        match expectation {
                            PublishedFieldStorageExpectation::HostF64 => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::HostF64(_)
                                ) {
                                    failures.push(
                                        "expected authoritative host_f64 published field storage"
                                            .to_string(),
                                    );
                                }
                            }
                        }
                    }

                    if let Some(tol) = spec.parity_tolerance {
                        let cpu_primary_field_id =
                            primary_result_field_id(fixture_procedure_step(&model));
                        let cpu_results = analysis_results_op(
                            &cpu_envelope.data,
                            AnalysisResultsQuery {
                                include_fields: vec![cpu_primary_field_id],
                                include_field_values: true,
                                include_diagnostics: false,
                                diagnostic_codes: Vec::new(),
                                include_modal_results: false,
                                include_buckling_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
                                include_nonlinear_results: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-cpu-{}", spec.id)),
                                None,
                            ),
                        );
                        let cpu_results = match cpu_results {
                            Ok(value) => value,
                            Err(err) => {
                                failures.push(format!(
                                    "fea.results failed for cpu fixture {}: {}",
                                    spec.id, err.error_code
                                ));
                                return FixtureRunRecord {
                                    fixture_id: spec.id.to_string(),
                                    validate_ok,
                                    validate_error_code,
                                    run_ok,
                                    run_error_code,
                                    cpu_run_ms,
                                    gpu_run_ms,
                                    gpu_fallback_events,
                                    published_primary_field_storage,
                                    gpu_solver_host_sync_count,
                                    gpu_solver_device_apply_k_ratio,
                                    gpu_speedup_ratio,
                                    gpu_solver_backend,
                                    gpu_solver_prepared_build_ms,
                                    gpu_solver_solve_ms,
                                    gpu_solver_fallback_apply_count,
                                    thermo_coupling_enabled,
                                    thermo_coupling_fingerprint,
                                    thermo_constitutive_temperature_factor,
                                    thermo_effective_modulus_scale,
                                    thermo_constitutive_material_spread_ratio,
                                    thermo_assignment_heterogeneity_index,
                                    thermo_region_delta_count,
                                    thermo_spatial_coverage_ratio,
                                    thermo_nonlinear_severity,
                                    electro_thermal_coupling_enabled,
                                    electro_thermal_coupling_fingerprint,
                                    electro_joule_heating_scale,
                                    electro_conductivity_spread_ratio,
                                    electro_transient_severity,
                                    electro_nonlinear_severity,
                                    plastic_nonlinear_severity,
                                    plastic_nonlinear_severity_mean,
                                    plastic_load_realization_ratio,
                                    plastic_load_amplification_ratio,
                                    contact_nonlinear_severity,
                                    contact_nonlinear_severity_mean,
                                    contact_load_realization_ratio,
                                    contact_load_amplification_ratio,
                                    thermal_max_relative_residual_norm,
                                    thermal_min_temperature_k,
                                    thermal_max_temperature_k,
                                    thermal_conductivity_spread_ratio,
                                    thermal_heat_capacity_spread_ratio,
                                    electromagnetic_enabled,
                                    electromagnetic_formulation_coverage_ratio,
                                    electromagnetic_magnetostatic_curl_curl_coverage_ratio,
                                    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
                                    electromagnetic_full_wave_displacement_current_coverage_ratio,
                                    electromagnetic_displacement_to_conduction_ratio,
                                    electromagnetic_material_frequency_response_coverage_ratio,
                                    electromagnetic_reference_frequency_hz,
                                    electromagnetic_applied_current_a,
                                    electromagnetic_solve_quality,
                                    electromagnetic_conductivity_spread_ratio,
                                    electromagnetic_relative_permittivity_spread_ratio,
                                    electromagnetic_relative_permeability_spread_ratio,
                                    electromagnetic_material_heterogeneity_index,
                                    electromagnetic_assignment_coverage_ratio,
                                    electromagnetic_assigned_coefficient_coverage_ratio,
                                    electromagnetic_region_coefficient_contrast_index,
                                    electromagnetic_condition_number_estimate,
                                    electromagnetic_source_realization_ratio,
                                    electromagnetic_source_region_coverage_ratio,
                                    electromagnetic_source_material_alignment_ratio,
                                    electromagnetic_source_localization_ratio,
                                    electromagnetic_source_overlap_ratio,
                                    electromagnetic_source_interference_index,
                                    electromagnetic_boundary_anchor_ratio,
                                    electromagnetic_boundary_condition_localization_ratio,
                                    electromagnetic_ground_anchor_effectiveness_ratio,
                                    electromagnetic_insulation_leakage_ratio,
                                    electromagnetic_flux_divergence_ratio,
                                    electromagnetic_energy_imbalance_ratio,
                                    electromagnetic_boundary_energy_ratio,
                                    electromagnetic_boundary_penalty_conditioning_contribution,
                                    electromagnetic_source_region_energy_consistency_ratio,
                                    electromagnetic_real_residual_norm,
                                    electromagnetic_imag_residual_norm,
                                    electromagnetic_sweep_count,
                                    electromagnetic_resonance_peak_frequency_hz,
                                    electromagnetic_resonance_peak_flux_density,
                                    electromagnetic_resonance_bandwidth_hz,
                                    electromagnetic_resonance_quality_factor,
                                    electromagnetic_resonance_flux_gain,
                                    numerically_acceptable,
                                    parity,
                                    threshold_assertions,
                                    failures,
                                };
                            }
                        };

                        let cpu_values = cpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);
                        let gpu_values = gpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);

                        if cpu_values.is_empty() || gpu_values.is_empty() {
                            failures.push(
                                "parity check requested but host vectors were not available"
                                    .to_string(),
                            );
                        } else if cpu_values.len() != gpu_values.len() {
                            failures.push("parity vector length mismatch".to_string());
                        } else {
                            let summary = compute_parity(cpu_values, gpu_values);
                            if summary.max_abs_diff > tol.abs {
                                failures.push(format!(
                                    "parity abs diff {} exceeds {}",
                                    summary.max_abs_diff, tol.abs
                                ));
                            }
                            if summary.max_rel_diff > tol.rel {
                                failures.push(format!(
                                    "parity rel diff {} exceeds {}",
                                    summary.max_rel_diff, tol.rel
                                ));
                            }
                            parity = Some(summary);
                        }
                    }
                }
                Err(err) => {
                    run_error_code = Some(err.error_code.clone());
                    failures.push(format!(
                        "unexpected gpu run failure for fixture {}: {}: {}",
                        spec.id, err.error_code, err.message
                    ));
                }
            }
        }
    }

    if spec.expect_run_error.is_some() {
        run_error_code = rejection::check(spec, &model, solver_mesh_artifact_path, &mut failures);
    }

    FixtureRunRecord {
        fixture_id: spec.id.to_string(),
        validate_ok,
        validate_error_code,
        run_ok,
        run_error_code,
        cpu_run_ms,
        gpu_run_ms,
        gpu_fallback_events,
        published_primary_field_storage,
        gpu_solver_host_sync_count,
        gpu_solver_device_apply_k_ratio,
        gpu_speedup_ratio,
        gpu_solver_backend,
        gpu_solver_prepared_build_ms,
        gpu_solver_solve_ms,
        gpu_solver_fallback_apply_count,
        thermo_coupling_enabled,
        thermo_coupling_fingerprint,
        thermo_constitutive_temperature_factor,
        thermo_effective_modulus_scale,
        thermo_constitutive_material_spread_ratio,
        thermo_assignment_heterogeneity_index,
        thermo_region_delta_count,
        thermo_spatial_coverage_ratio,
        thermo_nonlinear_severity,
        electro_thermal_coupling_enabled,
        electro_thermal_coupling_fingerprint,
        electro_joule_heating_scale,
        electro_conductivity_spread_ratio,
        electro_transient_severity,
        electro_nonlinear_severity,
        plastic_nonlinear_severity,
        plastic_nonlinear_severity_mean,
        plastic_load_realization_ratio,
        plastic_load_amplification_ratio,
        contact_nonlinear_severity,
        contact_nonlinear_severity_mean,
        contact_load_realization_ratio,
        contact_load_amplification_ratio,
        thermal_max_relative_residual_norm,
        thermal_min_temperature_k,
        thermal_max_temperature_k,
        thermal_conductivity_spread_ratio,
        thermal_heat_capacity_spread_ratio,
        electromagnetic_enabled,
        electromagnetic_formulation_coverage_ratio,
        electromagnetic_magnetostatic_curl_curl_coverage_ratio,
        electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
        electromagnetic_full_wave_displacement_current_coverage_ratio,
        electromagnetic_displacement_to_conduction_ratio,
        electromagnetic_material_frequency_response_coverage_ratio,
        electromagnetic_reference_frequency_hz,
        electromagnetic_applied_current_a,
        electromagnetic_solve_quality,
        electromagnetic_conductivity_spread_ratio,
        electromagnetic_relative_permittivity_spread_ratio,
        electromagnetic_relative_permeability_spread_ratio,
        electromagnetic_material_heterogeneity_index,
        electromagnetic_assignment_coverage_ratio,
        electromagnetic_assigned_coefficient_coverage_ratio,
        electromagnetic_region_coefficient_contrast_index,
        electromagnetic_condition_number_estimate,
        electromagnetic_source_realization_ratio,
        electromagnetic_source_region_coverage_ratio,
        electromagnetic_source_material_alignment_ratio,
        electromagnetic_source_localization_ratio,
        electromagnetic_source_overlap_ratio,
        electromagnetic_source_interference_index,
        electromagnetic_boundary_anchor_ratio,
        electromagnetic_boundary_condition_localization_ratio,
        electromagnetic_ground_anchor_effectiveness_ratio,
        electromagnetic_insulation_leakage_ratio,
        electromagnetic_flux_divergence_ratio,
        electromagnetic_energy_imbalance_ratio,
        electromagnetic_boundary_energy_ratio,
        electromagnetic_boundary_penalty_conditioning_contribution,
        electromagnetic_source_region_energy_consistency_ratio,
        electromagnetic_real_residual_norm,
        electromagnetic_imag_residual_norm,
        electromagnetic_sweep_count,
        electromagnetic_resonance_peak_frequency_hz,
        electromagnetic_resonance_peak_flux_density,
        electromagnetic_resonance_bandwidth_hz,
        electromagnetic_resonance_quality_factor,
        electromagnetic_resonance_flux_gain,
        numerically_acceptable,
        parity,
        threshold_assertions,
        failures,
    }
}
