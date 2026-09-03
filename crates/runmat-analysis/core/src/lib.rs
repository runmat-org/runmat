//! Solver-agnostic analysis problem model and validation contracts.

pub mod problem {
    pub mod boundary;
    pub mod connector;
    pub mod constraint;
    pub mod coordinate_frame;
    pub mod domains;
    pub mod excitation;
    pub mod initial_condition;
    pub mod interfaces;
    pub mod loads;
    pub mod material_assignment;
    pub mod materials;
    pub mod model;
    pub mod quantity;
    pub mod reference_point;
    pub mod response_spectrum;
    pub mod schedule;
    pub mod signal;
    pub mod steps;
    pub mod structure;
    pub mod target;
}
pub mod field;
pub mod study;
pub mod validate;
mod validate_authored;
mod validate_excitations;
mod validate_signals;

pub use field::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
pub use problem::boundary::{AnalysisBoundary, BoundaryOperator};
pub use problem::connector::{AnalysisConnector, ConnectorKind};
pub use problem::constraint::{
    AcousticConstraint, AcousticConstraintKind, AnalysisConstraint, ElectricalConstraint,
    ElectricalConstraintKind, FluidConstraint, FluidConstraintKind, MagneticConstraint,
    MagneticConstraintKind, PairingPolicy, PrescribedRotations, PrescribedTranslations,
    ReferencePointCoupling, RotationDofValue, StructuralConstraint, StructuralConstraintEquation,
    StructuralConstraintEquationTerm, StructuralConstraintKind, StructuralRotationDof,
    StructuralTranslationDof, ThermalConstraint, ThermalConstraintKind, TranslationDofValue,
};
pub use problem::coordinate_frame::{AxialFrame, CartesianFrame, CoordinateFrame, SphericalFrame};
pub use problem::domains::{
    CfdDomain, CfdSolveFamily, CfdTimeProfilePoint, ElectroRegionConductivityScale,
    ElectroThermalDomain, ElectroTimeProfilePoint, ElectromagneticDomain,
    ThermoFieldInterpolationMode, ThermoFieldSource, ThermoMechanicalDomain,
    ThermoRegionTemperatureDelta, ThermoTimeProfilePoint,
};
pub use problem::excitation::{
    AnalysisExcitation, Direction, ExcitationAction, ExcitationCase, ExcitationCaseActivation,
    ExcitationCaseId, ExcitationDistribution, ExcitationId, ExcitationKinematics, ExcitationTarget,
    ProjectionRequirement,
};
pub use problem::initial_condition::{
    InitialCondition, InitialConditionTarget, InitialConditionValue, InitialFieldArtifactRef,
    InitialValue,
};
pub use problem::interfaces::{
    AnalysisInterface, AnalysisInterfaceKind, ConjugateHeatTransferInterfaceModel,
    ContactInterfaceModel, FluidStructureInterfaceModel,
};
pub use problem::loads::{LoadCase, LoadKind};
pub use problem::material_assignment::{EvidenceConfidence, MaterialAssignment};
pub use problem::materials::{
    ConductivityFrequencyPoint, MaterialAcousticModel, MaterialElectricalModel,
    MaterialMechanicalModel, MaterialModel, MaterialPlasticModel, MaterialThermalModel,
};
pub use problem::model::{AnalysisModel, AnalysisModelId};
pub use problem::quantity::{
    Acceleration, AngularAcceleration, AngularVelocity, BodyForceDensity, Damping, Displacement,
    ElectricCharge, ElectricChargeDensity, ElectricCurrent, ElectricCurrentDensity,
    ElectricPotential, Force, HeatFlux, HeatGenerationDensity, HeatTransferCoefficient, Length,
    MagneticVectorPotential, MassFlowRate, Moment, Position, Pressure, Rotation,
    SpecificAcousticImpedance, Stiffness, Temperature, Traction, Velocity,
};
pub use problem::reference_point::ReferencePoint;
pub use problem::response_spectrum::{
    DirectionalCombination, ModalCombination, ModalTruncationPolicy, ResponseSpectrum,
    ResponseSpectrumId, ShockResponseSpectrumMethod, SpectrumAbscissa, SpectrumConstruction,
    SpectrumDamping, SpectrumDirection, SpectrumEnvelope, SpectrumEvidence, SpectrumExtrapolation,
    SpectrumInterpolation, SpectrumResponse,
};
pub use problem::schedule::{
    FieldArtifactValue, ScheduleExtrapolation, ScheduleInterpolation, StepValue,
    StudyParameterValue, TabulatedSchedule, ValueSchedule,
};
pub use problem::signal::{
    BreakpointSide, GeneratorArgument, GeneratorDomain, GeneratorEffectPolicy, GeneratorProvenance,
    PhysicalQuantityKind, PhysicalSignal, RandomDistribution, Signal, SignalApproximationEvidence,
    SignalArtifactDomain, SignalArtifactRef, SignalAxis, SignalBreakpoint, SignalDomainKind,
    SignalExtrapolation, SignalGenerator, SignalId, SignalInterpolation, SignalOrdinate,
    SignalPrecision, SignalProvenance, SignalRepresentation, SignalSamplingPolicy, SignalShape,
    SignalValue, StandardWaveform, StandardWaveformKind, TabulatedSignal,
};
pub use problem::steps::{AnalysisStep, AnalysisStepKind};
pub use problem::structure::{
    BeamElementModel, BeamSectionModel, ShellElementModel, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel, StructuralNode,
};
pub use problem::target::{
    EntityDimension, ExpectedTopology, RegionTarget, ResolvedTopologyTarget, TargetCardinality,
};
pub use study::{
    AnalysisProcedure, AnalysisRunIdentity, CoupledPhysicsFamily, ExecutionFallbackPolicy,
    FormulationImplementation, FormulationMaturity, PhysicsFamily, SolutionSpace,
};
pub use validate::{validate_model, validate_model_against_geometry, AnalysisValidationError};

#[cfg(test)]
mod tests {
    use runmat_geometry_core::UnitSystem;

    use super::*;

    fn valid_model() -> AnalysisModel {
        AnalysisModel {
            model_id: AnalysisModelId("analysis_model_1".to_string()),
            geometry_id: "geo:model_1".to_string(),
            geometry_revision: 1,
            units: UnitSystem::Meter,
            frame: CoordinateFrame::Global,
            materials: vec![MaterialModel {
                material_id: "mat_steel".to_string(),
                name: "Steel".to_string(),
                mechanical: MaterialMechanicalModel {
                    youngs_modulus_pa: 200e9,
                    poisson_ratio: 0.3,
                    density_kg_per_m3: 7850.0,
                },
                thermal: MaterialThermalModel::default(),
                acoustic: None,
                electrical: None,
                plastic: None,
            }],
            material_assignments: Vec::new(),
            structural: None,
            thermo_mechanical: None,
            electro_thermal: None,
            electromagnetic: None,
            cfd: None,
            interfaces: Vec::new(),
            constraints: vec![AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: "bc_fixed_root".to_string(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Fixed {
                    target: RegionTarget::boundary("root"),
                },
            })],
            boundaries: Vec::new(),
            connectors: Vec::new(),
            signals: Vec::new(),
            excitations: Vec::new(),
            excitation_cases: Vec::new(),
            initial_conditions: Vec::new(),
            response_spectra: Vec::new(),
            loads: vec![LoadCase {
                load_id: "load_tip".to_string(),
                region_id: "tip".to_string(),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -1000.0,
                    fz: 0.0,
                },
            }],
            steps: vec![AnalysisStep {
                step_id: "step_static".to_string(),
                kind: AnalysisStepKind::Static,
            }],
        }
    }

    #[test]
    fn missing_material_bc_load_validation_failures() {
        let mut model = valid_model();
        model.materials.clear();
        assert_eq!(
            validate_model(&model).expect_err("expected material validation failure"),
            AnalysisValidationError::MissingMaterials
        );

        let mut model = valid_model();
        model.constraints.clear();
        assert_eq!(
            validate_model(&model).expect_err("expected boundary condition validation failure"),
            AnalysisValidationError::MissingConstraintsOrBoundaries
        );

        let mut model = valid_model();
        model.loads.clear();
        assert_eq!(
            validate_model(&model).expect_err("expected load validation failure"),
            AnalysisValidationError::MissingLoads
        );
    }

    #[test]
    fn invalid_moment_vectors_fail_validation() {
        let mut model = valid_model();
        model.loads[0].kind = LoadKind::Moment {
            mx: f64::NAN,
            my: 0.0,
            mz: 1.0,
        };
        assert_eq!(
            validate_model(&model).expect_err("expected nonfinite moment validation failure"),
            AnalysisValidationError::InvalidMomentVector {
                load_id: "load_tip".to_string()
            }
        );

        let mut model = valid_model();
        model.loads[0].kind = LoadKind::Moment {
            mx: 0.0,
            my: 0.0,
            mz: 0.0,
        };
        assert_eq!(
            validate_model(&model).expect_err("expected zero moment validation failure"),
            AnalysisValidationError::ZeroMomentVector {
                load_id: "load_tip".to_string()
            }
        );
    }

    #[test]
    fn invalid_wrench_vectors_fail_validation() {
        let mut model = valid_model();
        model.loads[0].kind = LoadKind::Wrench {
            fx: 0.0,
            fy: 1.0,
            fz: 0.0,
            mx: f64::INFINITY,
            my: 0.0,
            mz: 0.0,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
        };
        assert_eq!(
            validate_model(&model).expect_err("expected nonfinite wrench validation failure"),
            AnalysisValidationError::InvalidWrench {
                load_id: "load_tip".to_string()
            }
        );

        let mut model = valid_model();
        model.loads[0].kind = LoadKind::Wrench {
            fx: 0.0,
            fy: 0.0,
            fz: 0.0,
            mx: 0.0,
            my: 0.0,
            mz: 0.0,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
        };
        assert_eq!(
            validate_model(&model).expect_err("expected zero wrench validation failure"),
            AnalysisValidationError::ZeroWrench {
                load_id: "load_tip".to_string()
            }
        );
    }

    #[test]
    fn unit_frame_mismatch_rejection() {
        let mut model = valid_model();
        model.units = UnitSystem::Inch;
        let err =
            validate_model_against_geometry(&model, UnitSystem::Meter, &CoordinateFrame::Global)
                .expect_err("expected unit mismatch");
        assert!(matches!(
            err,
            AnalysisValidationError::UnitMismatch {
                model: UnitSystem::Inch,
                geometry: UnitSystem::Meter
            }
        ));

        let mut model = valid_model();
        model.frame = CoordinateFrame::Cartesian(CartesianFrame {
            frame_id: "fixture_frame".to_string(),
            origin: Position::new([0.0, 0.0, 0.0]),
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
        });
        let err =
            validate_model_against_geometry(&model, UnitSystem::Meter, &CoordinateFrame::Global)
                .expect_err("expected frame mismatch");
        let AnalysisValidationError::FrameMismatch { model, geometry } = err else {
            panic!("expected typed frame mismatch");
        };
        assert!(matches!(*model, CoordinateFrame::Cartesian(_)));
        assert_eq!(*geometry, CoordinateFrame::Global);
    }

    #[test]
    fn valid_model_is_accepted() {
        let model = valid_model();
        validate_model(&model).expect("model should be valid");
        validate_model_against_geometry(&model, UnitSystem::Meter, &CoordinateFrame::Global)
            .expect("model should match geometry context");
    }

    #[test]
    fn moment_load_kind_serializes_as_snake_case() {
        let load = LoadCase {
            load_id: "tip_moment".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Moment {
                mx: 1.0,
                my: 2.0,
                mz: 3.0,
            },
        };

        let json = serde_json::to_value(&load).expect("load should serialize");
        assert_eq!(json["kind"]["moment"]["mx"], 1.0);
        assert_eq!(json["kind"]["moment"]["my"], 2.0);
        assert_eq!(json["kind"]["moment"]["mz"], 3.0);

        let decoded: LoadCase = serde_json::from_value(json).expect("load should deserialize");
        assert_eq!(decoded, load);
    }

    #[test]
    fn wrench_load_kind_serializes_as_snake_case() {
        let load = LoadCase {
            load_id: "tip_wrench".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Wrench {
                fx: 1.0,
                fy: 2.0,
                fz: 3.0,
                mx: 4.0,
                my: 5.0,
                mz: 6.0,
                px: 0.1,
                py: 0.2,
                pz: 0.3,
            },
        };

        let json = serde_json::to_value(&load).expect("load should serialize");
        assert_eq!(json["kind"]["wrench"]["fx"], 1.0);
        assert_eq!(json["kind"]["wrench"]["mz"], 6.0);
        assert_eq!(json["kind"]["wrench"]["pz"], 0.3);

        let decoded: LoadCase = serde_json::from_value(json).expect("load should deserialize");
        assert_eq!(decoded, load);
    }

    #[test]
    fn prescribed_rotation_constraint_serializes_as_snake_case() {
        let constraint = AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "root_rotation".to_string(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::PrescribedRotations(PrescribedRotations {
                target: RegionTarget::boundary("root"),
                values: vec![RotationDofValue {
                    dof: StructuralRotationDof::Rz,
                    value: ValueSchedule::Constant(Rotation::new(0.125)),
                }],
            }),
        });

        let json = serde_json::to_value(&constraint).expect("constraint should serialize");
        assert_eq!(
            json["structural"]["kind"]["prescribed_rotations"]["values"][0]["value"]["constant"]
                ["radians"],
            0.125
        );

        let decoded: AnalysisConstraint =
            serde_json::from_value(json).expect("constraint should deserialize");
        assert_eq!(decoded, constraint);
    }

    #[test]
    fn structural_beam_model_round_trips() {
        let mut model = valid_model();
        model.structural = Some(StructuralModel {
            nodes: vec![
                StructuralNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                StructuralNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
            ],
            elements: vec![StructuralElement {
                element_id: "beam_1".to_string(),
                region_id: "beam_span".to_string(),
                kind: StructuralElementKind::Beam(BeamElementModel {
                    node_ids: [1, 2],
                    section_id: "section_1".to_string(),
                    reference_axis: [0.0, 0.0, 1.0],
                }),
            }],
            beam_sections: vec![BeamSectionModel {
                section_id: "section_1".to_string(),
                area_m2: 2.0e-4,
                iy_m4: 1.6e-9,
                iz_m4: 6.4e-9,
                torsion_j_m4: 2.4e-9,
                outer_fiber_y_m: 0.01,
                outer_fiber_z_m: 0.005,
                torsion_outer_radius_m: 0.011_180_339_887_498_949,
            }],
            shell_sections: Vec::new(),
        });

        let json = serde_json::to_value(&model).expect("model should serialize");
        assert_eq!(
            json["structural"]["elements"][0]["kind"]["beam"]["node_ids"][1],
            2
        );

        let decoded: AnalysisModel =
            serde_json::from_value(json).expect("model should deserialize");
        assert_eq!(decoded, model);
    }

    #[test]
    fn structural_shell_model_round_trips() {
        let mut model = valid_model();
        model.structural = Some(StructuralModel {
            nodes: vec![
                StructuralNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                StructuralNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                StructuralNode {
                    node_id: 3,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
            ],
            elements: vec![StructuralElement {
                element_id: "shell_1".to_string(),
                region_id: "shell_panel".to_string(),
                kind: StructuralElementKind::Shell(ShellElementModel {
                    node_ids: [1, 2, 3],
                    section_id: "panel_2mm".to_string(),
                    reference_axis: [1.0, 0.0, 0.0],
                }),
            }],
            beam_sections: Vec::new(),
            shell_sections: vec![ShellSectionModel {
                section_id: "panel_2mm".to_string(),
                thickness_m: 0.002,
                shear_correction: 5.0 / 6.0,
                drilling_stiffness_scale: 1.0e-4,
            }],
        });

        let json = serde_json::to_value(&model).expect("model should serialize");
        assert_eq!(
            json["structural"]["elements"][0]["kind"]["shell"]["node_ids"][2],
            3
        );

        let decoded: AnalysisModel =
            serde_json::from_value(json).expect("model should deserialize");
        assert_eq!(decoded, model);
    }

    #[test]
    fn electrical_model_defaults_frequency_response() {
        let electrical = MaterialElectricalModel::default();
        assert!(electrical.conductivity_frequency_response.is_empty());
    }

    #[test]
    fn duplicate_structural_dofs_are_rejected() {
        let mut model = valid_model();
        model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "duplicate_ux".to_string(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::PrescribedTranslations(PrescribedTranslations {
                target: RegionTarget::boundary("root"),
                values: vec![
                    TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: ValueSchedule::Constant(Displacement::new(0.0)),
                    },
                    TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: ValueSchedule::Constant(Displacement::new(1.0)),
                    },
                ],
            }),
        })];

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
    }

    #[test]
    fn malformed_schedule_is_rejected() {
        let mut model = valid_model();
        model.constraints = vec![AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "temperature".to_string(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary("root"),
                value: ValueSchedule::TimeHistory(TabulatedSchedule {
                    abscissa: vec![0.0, 1.0, 0.5],
                    values: vec![
                        Temperature::new(293.15),
                        Temperature::new(300.0),
                        Temperature::new(310.0),
                    ],
                    interpolation: ScheduleInterpolation::Linear,
                    extrapolation: ScheduleExtrapolation::Reject,
                }),
            },
        })];

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
    }

    #[test]
    fn incompatible_target_topology_is_rejected() {
        let mut model = valid_model();
        let AnalysisConstraint::Structural(constraint) = &mut model.constraints[0] else {
            panic!("expected structural constraint")
        };
        let StructuralConstraintKind::Fixed { target } = &mut constraint.kind else {
            panic!("expected fixed constraint")
        };
        target.entity_dimension = EntityDimension::Volume;

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidTarget { .. })
        ));
    }

    #[test]
    fn invalid_frame_and_boundary_coefficients_are_rejected() {
        let mut model = valid_model();
        model.frame = CoordinateFrame::Cartesian(CartesianFrame {
            frame_id: "parallel_axes".to_string(),
            origin: Position::new([0.0, 0.0, 0.0]),
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [2.0, 0.0, 0.0],
        });
        assert_eq!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidFrame)
        );

        let mut model = valid_model();
        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "convection".to_string(),
            target: RegionTarget::boundary("root"),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: ValueSchedule::Constant(Temperature::new(293.15)),
                coefficient: HeatTransferCoefficient::new(-1.0),
            },
        }];
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidBoundary { .. })
        ));
    }

    fn force_signal(id: &str, representation: SignalRepresentation<Force>) -> PhysicalSignal {
        PhysicalSignal::Force(Signal {
            signal_id: SignalId(id.to_string()),
            shape: SignalShape::Scalar,
            frame: None,
            representation,
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        })
    }

    #[test]
    fn malformed_signal_domain_and_generator_are_rejected() {
        let mut model = valid_model();
        model.signals = vec![force_signal(
            "force_history",
            SignalRepresentation::Tabulated(TabulatedSignal {
                axis: SignalAxis::TimeSeconds(vec![0.0, 1.0, 0.5]),
                values: vec![
                    SignalOrdinate::Real(SignalValue::Scalar(Force::new(1.0))),
                    SignalOrdinate::Real(SignalValue::Scalar(Force::new(2.0))),
                    SignalOrdinate::Real(SignalValue::Scalar(Force::new(3.0))),
                ],
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
                breakpoints: Vec::new(),
            }),
        )];
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidSignal { .. })
        ));

        model.signals = vec![force_signal(
            "generated_force",
            SignalRepresentation::Generator(SignalGenerator {
                qualified_function: "anonymousScript".to_string(),
                arguments: Vec::new(),
                domain: GeneratorDomain::TimeSeconds,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.0,
                    end: 1.0,
                    sample_count: 101,
                },
                effects: GeneratorEffectPolicy::Pure,
            }),
        )];
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidSignal { .. })
        ));
    }

    #[test]
    fn excitation_signal_quantity_and_distribution_are_checked() {
        let mut model = valid_model();
        model.signals = vec![force_signal(
            "tip_force",
            SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
        )];
        let excitation = AnalysisExcitation {
            excitation_id: ExcitationId("tip_pressure".to_string()),
            case_id: ExcitationCaseId("service_case".to_string()),
            target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerArea,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("tip_force".to_string()),
            active_step_ids: vec!["step_static".to_string()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1.0e-10,
            },
        };
        model.excitations = vec![ExcitationAction::Pressure(excitation)];
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidExcitation { .. })
        ));
    }

    #[test]
    fn typed_excitation_case_and_initial_condition_are_accepted() {
        let mut model = valid_model();
        model.signals = vec![force_signal(
            "tip_force",
            SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
        )];
        model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
            excitation_id: ExcitationId("tip_force_action".to_string()),
            case_id: ExcitationCaseId("service_case".to_string()),
            target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
            frame: CoordinateFrame::Global,
            direction: Some(Direction {
                vector: [0.0, -1.0, 0.0],
            }),
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("tip_force".to_string()),
            active_step_ids: vec!["step_static".to_string()],
            combination_group: Some("service".to_string()),
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: true,
                relative_tolerance: 1.0e-10,
            },
        })];
        model.excitation_cases = vec![ExcitationCase {
            case_id: ExcitationCaseId("service_case".to_string()),
            action_ids: vec![ExcitationId("tip_force_action".to_string())],
            activation: ExcitationCaseActivation::Always,
        }];
        model.initial_conditions = vec![InitialCondition::Velocity(InitialConditionValue {
            initial_condition_id: "initial_velocity".to_string(),
            target: InitialConditionTarget::Region(RegionTarget::volume("body")),
            frame: CoordinateFrame::Global,
            value: InitialValue::UniformVector3([
                Velocity::new(0.0),
                Velocity::new(0.0),
                Velocity::new(0.0),
            ]),
            applies_before_step_id: "step_static".to_string(),
        })];

        validate_model(&model).expect("typed excitation model should be valid");
        let encoded = serde_json::to_value(&model).expect("model should serialize");
        let decoded: AnalysisModel =
            serde_json::from_value(encoded).expect("model should deserialize");
        assert_eq!(decoded, model);
    }

    #[test]
    fn excitation_case_membership_is_reciprocal_and_exhaustive() {
        let mut model = valid_model();
        model.signals = vec![force_signal(
            "tip_force",
            SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
        )];
        model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
            excitation_id: ExcitationId("tip_force_action".to_string()),
            case_id: ExcitationCaseId("declared_owner".to_string()),
            target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
            frame: CoordinateFrame::Global,
            direction: Some(Direction {
                vector: [0.0, -1.0, 0.0],
            }),
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("tip_force".to_string()),
            active_step_ids: vec!["step_static".to_string()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1.0e-10,
            },
        })];
        model.excitation_cases = vec![ExcitationCase {
            case_id: ExcitationCaseId("different_case".to_string()),
            action_ids: vec![ExcitationId("tip_force_action".to_string())],
            activation: ExcitationCaseActivation::Always,
        }];

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidExcitationCase { .. })
        ));
    }

    #[test]
    fn scalar_directional_excitation_requires_direction() {
        let mut model = valid_model();
        model.signals = vec![force_signal(
            "tip_force",
            SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
        )];
        model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
            excitation_id: ExcitationId("tip_force_action".to_string()),
            case_id: ExcitationCaseId("service_case".to_string()),
            target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("tip_force".to_string()),
            active_step_ids: vec!["step_static".to_string()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1.0e-10,
            },
        })];

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidExcitation { .. })
        ));
    }

    #[test]
    fn initial_condition_shape_is_physics_typed() {
        let mut model = valid_model();
        model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
            initial_condition_id: "initial_temperature".to_string(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::UniformVector3([
                Temperature::new(293.15),
                Temperature::new(293.15),
                Temperature::new(293.15),
            ]),
            applies_before_step_id: "step_static".to_string(),
        })];

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidInitialCondition { .. })
        ));
    }

    #[test]
    fn invalid_response_spectrum_is_rejected() {
        let mut model = valid_model();
        model.response_spectra = vec![ResponseSpectrum {
            spectrum_id: ResponseSpectrumId("launch_srs".to_string()),
            abscissa: SpectrumAbscissa::FrequencyHertz(vec![10.0, 100.0]),
            ordinates: vec![5.0, 20.0],
            damping: SpectrumDamping::Ratio(1.2),
            response: SpectrumResponse::AbsoluteAcceleration,
            envelope: SpectrumEnvelope::Maximax,
            interpolation: SpectrumInterpolation::LogLog,
            extrapolation: SpectrumExtrapolation::Reject,
            direction: SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector: [0.0, 0.0, 1.0],
            },
            modal_combination: ModalCombination::CompleteQuadraticCombination,
            directional_combination: DirectionalCombination::Srss,
            truncation: ModalTruncationPolicy {
                minimum_cumulative_mass_ratio: 0.9,
                include_residual_mode: true,
                include_rigid_response: true,
                include_missing_mass: true,
            },
            construction: SpectrumConstruction::AuthoredDesign,
            evidence: SpectrumEvidence {
                sampling_adequacy_ratio: 1.0,
                max_interpolation_error: Some(0.01),
            },
            provenance_digest: "sha256:spectrum".to_string(),
        }];
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidResponseSpectrum { .. })
        ));

        model.response_spectra[0].damping = SpectrumDamping::Ratio(0.05);
        model.response_spectra[0].construction =
            SpectrumConstruction::DerivedFromAccelerationHistory {
                source_signal_id: SignalId("missing_acceleration".to_string()),
                method: ShockResponseSpectrumMethod::ExactPiecewiseLinear,
            };
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidResponseSpectrum { .. })
        ));
    }
}
