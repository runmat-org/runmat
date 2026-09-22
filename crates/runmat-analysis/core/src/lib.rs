//! Solver-agnostic analysis problem model and validation contracts.

pub mod problem {
    pub mod authored_constraints;
    pub mod authored_excitations;
    pub mod authored_expressions;
    pub mod authored_initial_conditions;
    pub mod authored_inputs;
    pub mod authored_model_physics;
    pub mod authored_parameterized_signals;
    pub mod authored_properties;
    pub mod boundary;
    pub mod connector;
    pub mod constraint;
    pub mod coordinate_frame;
    pub mod domains;
    pub mod excitation;
    pub mod initial_condition;
    pub mod interfaces;
    pub mod material_assignment;
    pub mod materials;
    pub mod model;
    pub mod model_authoring;
    pub mod model_realization;
    pub mod prescribed_value;
    pub mod quantity;
    pub mod random_vibration;
    pub mod reference_point;
    pub mod region_resolution;
    pub mod response_spectrum;
    pub mod signal;
    pub mod structure;
    pub mod target;
}
mod execution_policy;
pub mod field;
pub mod study;
pub mod validate;
mod validate_accuracy;
mod validate_authored;
mod validate_domains;
mod validate_excitations;
mod validate_materials;
mod validate_outputs;
mod validate_sequence;
mod validate_signals;

pub use execution_policy::{
    AcceleratorPreference, AuthoredExecutionPolicy, AuthoredExecutionPolicyError,
    AuthoredExecutionPolicyRef, BackendPreference, CheckpointPreference, CooperativePolicy,
    ExecutionResourceLimits, FanoutPolicy, InclusiveCountRange, OperatorPartitioner,
    ParallelismLimit, PartitionBalance, PlacementPreference, ResiliencePolicy,
};
pub use study::{
    attach_authored_outputs, authored_output_types, geometry_parameter_input_type,
    lower_authored_observations, validate_authored_outputs, AuthoredCaseOutputRef,
    AuthoredExtremaTiePolicy, AuthoredFieldDomain, AuthoredFieldProjection,
    AuthoredGeometryArgument, AuthoredGeometryArgumentValue, AuthoredGeometryInstance,
    AuthoredMeshRef, AuthoredModelArgument, AuthoredModelArgumentValue, AuthoredModelInstance,
    AuthoredObservationError, AuthoredOutput, AuthoredOutputError, AuthoredOutputExpression,
    AuthoredOutputPolicy, AuthoredOutputType, AuthoredOutputValue, AuthoredPrincipalSelection,
    AuthoredSeriesDomain, AuthoredSeriesReduction, AuthoredSpatialReduction, AuthoredStudy,
    AuthoredStudyError, AuthoredStudyInputBinding, AuthoredStudyInputRef,
    AuthoredStudyRealizationError, AuthoredStudyRef, AuthoredSweep, AuthoredSweepAxis,
    AuthoredSweepAxisValue, AuthoredSweepBinding, AuthoredSweepCase, AuthoredSweepDefinition,
    AuthoredSweepError, AuthoredSweepOutput, AuthoredSweepOutputExpression,
    AuthoredVectorComponent, CaseOutputProjection, LoweredAuthoredObservation,
    RealizedGeometryInstance, RealizedStudy, SweepAxisCombination, SweepExtremum,
};
pub use validate::physics_family_can_consume_state;
pub use validate_authored::validate_coordinate_frame;
pub use validate_outputs::spec::quantity_supported as procedure_produces_observation_quantity;
pub use validate_outputs::{
    observation_sampling_matches_domain, validate_authored_observation_request,
    validate_observation_sampling,
};
pub use validate_signals::validate_physical_signals;

pub use field::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
pub use problem::authored_constraints::{
    authored_motion_acceleration_is_compatible, materialized_motion_acceleration_is_compatible,
    AuthoredConstraint, AuthoredConstraintError, AuthoredConstraintKind, AuthoredConstraintTarget,
    AuthoredPairedConstraint, AuthoredPairingPolicy, AuthoredPeriodicConstraint,
    AuthoredPeriodicTransform, AuthoredPhaseShiftedPeriodicConstraint,
    AuthoredPrescribedStructuralMotion, AuthoredReferencePointCoupling, AuthoredRotationDofValue,
    AuthoredStructuralConstraintEquation, AuthoredStructuralConstraintEquationTerm,
    AuthoredStructuralDofs, AuthoredTargetCardinality, AuthoredThermalJump,
    AuthoredTranslationDofValue,
};
pub use problem::authored_excitations::{
    validate_authored_excitations, AuthoredExcitation, AuthoredExcitationCase,
    AuthoredExcitationKind, AuthoredExcitationTarget,
};
pub use problem::authored_expressions::{
    AuthoredExpressionError, AuthoredNumericExpression, AuthoredNumericExpressionKind,
    AuthoredNumericOperator,
};
pub use problem::authored_initial_conditions::{
    AuthoredInitialCondition, AuthoredInitialConditionError, AuthoredInitialConditionKind,
    AuthoredInitialConditionTarget,
};
pub use problem::authored_inputs::{
    AuthoredInputDefinition, AuthoredInputError, AuthoredInputType, AuthoredInputValue,
};
pub use problem::authored_model_physics::{
    AuthoredBoundary, AuthoredBoundaryKind, AuthoredConnector, AuthoredConnectorKind,
    AuthoredInterface, AuthoredInterfaceKind, AuthoredModelPhysicsError,
};
pub use problem::authored_parameterized_signals::{
    AuthoredParameterizedSignal, AuthoredParameterizedSignalError, AuthoredSignalExpression,
};
pub use problem::authored_properties::{
    AuthoredLayupDefinition, AuthoredLayupSource, AuthoredOrientationDefinition,
    AuthoredOrientationSource, AuthoredPly, AuthoredPropertyAssignment,
    AuthoredPropertyDefinitionRef, AuthoredPropertyError, AuthoredSectionDefinition,
    AuthoredSectionSource, AxisOrientation, CartesianOrientation, CircularBeamSection,
    GeneralBeamSection, HomogeneousShellSection, RectangularBeamSection, TubularBeamSection,
};
pub use problem::boundary::{AnalysisBoundary, BoundaryOperator};
pub use problem::connector::{AnalysisConnector, ConnectorKind};
pub use problem::constraint::{
    AccelerationSignalId, AcousticConstraint, AcousticConstraintKind, AnalysisConstraint,
    ElectricalConstraint, ElectricalConstraintKind, FluidConstraint, FluidConstraintKind,
    MagneticConstraint, MagneticConstraintKind, PairedConstraint, PairingPolicy,
    PeriodicConstraint, PeriodicTransform, PhaseShiftedPeriodicConstraint,
    PrescribedStructuralDofs, PrescribedStructuralMotion, ReferencePointCoupling, RotationDofValue,
    StructuralConstraint, StructuralConstraintEquation, StructuralConstraintEquationTerm,
    StructuralConstraintKind, StructuralKinematicHistory, StructuralMotionReconstruction,
    StructuralRotationDof, StructuralTranslationDof, ThermalConstraint, ThermalConstraintKind,
    TranslationDofMotion, TranslationDofValue,
};
pub use problem::coordinate_frame::{AxialFrame, CartesianFrame, CoordinateFrame, SphericalFrame};
pub use problem::domains::{
    CfdDomain, ElectroThermalDomain, ElectromagneticDomain, ThermalDomain, ThermoMechanicalDomain,
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
    AnalysisInterface, AnalysisInterfaceKind, ConjugateHeatTransferInterfaceModel, ContactFriction,
    ContactInterfaceModel, FluidStructureInterfaceModel, FluidStructureKinematics,
    ThermalInterfaceTransfer,
};
pub use problem::material_assignment::{
    EvidenceConfidence, MaterialAssignment, PropertyRegionScope, PropertyRegionSelection,
    PropertyRegionTarget,
};
pub use problem::materials::{
    ConductivityFrequencyPoint, ImportedCadEnvironmentCondition, ImportedCadMaterialAssociation,
    ImportedCadMaterialAssociationTarget, ImportedCadMaterialProperty,
    ImportedCadMaterialPropertyKind, ImportedCadMaterialProvenance, MaterialAcousticModel,
    MaterialCapability, MaterialElasticityModel, MaterialElectricalModel, MaterialMechanicalModel,
    MaterialModel, MaterialOrthotropicLaminaModel, MaterialPlasticModel, MaterialProvenance,
    MaterialThermalModel, MechanicalDampingModel, MissingMaterialCapability, SolverMaterialModel,
};
pub use problem::model::{
    AnalysisModel, AnalysisModelId, SolverAnalysisModel, SolverAnalysisSequence,
    SolverProcedureStep,
};
pub use problem::model_authoring::{
    AuthoredMaterialAssignment, AuthoredMaterialRef, AuthoredMaterialSource, AuthoredModel,
    AuthoredModelError, AuthoredModelInclude, AuthoredModelInputRef, AuthoredModelRef,
    AuthoredPropertyTarget, AuthoredRegionSource,
};
pub use problem::model_realization::{
    realize_analysis_model, AnalysisModelRealizationError, AnalysisModelRealizationInput,
    RealizedAnalysisModel, RealizedLayupDefinition, RealizedModelProperties,
    RealizedOrientationDefinition, RealizedPropertyAssignment, RealizedSectionDefinition,
};
pub use problem::prescribed_value::{PrescribedQuantity, PrescribedValue};
pub use problem::quantity::{
    Acceleration, AngularAcceleration, AngularVelocity, BodyForceDensity, CanonicalQuantity,
    ComplexPhase, Damping, Displacement, DynamicViscosity, ElectricCharge, ElectricChargeDensity,
    ElectricCurrent, ElectricCurrentDensity, ElectricPotential, Force, Frequency,
    FrictionCoefficient, HeatFlux, HeatGenerationDensity, HeatTransferCoefficient, Length,
    MagneticVectorPotential, MassDensity, MassFlowRate, Moment, Position, Pressure,
    RayleighMassCoefficient, RayleighStiffnessCoefficient, Rotation, SpecificAcousticImpedance,
    Stiffness, Temperature, TemperatureDifference, ThermalContactResistance, Traction, Translation,
    TurbulenceIntensity, Velocity,
};
pub use problem::random_vibration::{
    ComplexSpectralDensity, CrossPowerSpectralDensity, RandomExcitationChannel,
    RandomExcitationChannelId, RandomExcitationId, RandomExcitationQuantity,
    RandomExcitationSpectrum, RandomSpectrumEvidence, RandomSpectrumExtrapolation,
    RandomSpectrumInterpolation, SpectralDensityConvention,
};
pub use problem::reference_point::ReferencePoint;
pub use problem::region_resolution::{
    resolve_geometry_region_aliases, GeometryRegionAliases, GeometryRegionResolutionError,
};
pub use problem::response_spectrum::{
    DirectionalCombination, ModalCombination, ModalTruncationCorrection, ModalTruncationPolicy,
    ResponseSpectrum, ResponseSpectrumId, RigidResponsePolicy, ShockResponseSpectrumMethod,
    SpectrumAbscissa, SpectrumConstruction, SpectrumDamping, SpectrumDirection, SpectrumEnvelope,
    SpectrumEvidence, SpectrumExtrapolation, SpectrumInterpolation, SpectrumResponse,
};
pub use problem::signal::{
    GeneratorArgument, GeneratorCallable, GeneratorDomain, GeneratorEffectPolicy,
    GeneratorInvocation, GeneratorProvenance, ImportedFunctionIdentity, NumericalDataSource,
    PhysicalQuantityKind, PhysicalSignal, RandomDistribution, Signal, SignalApproximationEvidence,
    SignalArtifactDomain, SignalArtifactRef, SignalDomainKind, SignalExtrapolation,
    SignalGenerator, SignalId, SignalInterpolation, SignalKnot, SignalKnotValue, SignalOrdinate,
    SignalPointContinuity, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalSampleApproach, SignalSamplingPolicy, SignalShape, SignalValue, StandardWaveform,
    StandardWaveformKind, TabulatedSignal, TabulatedSignalDomain,
};
pub use problem::structure::{
    BeamElementModel, BeamSectionModel, ConformalSolidAttachment, HomogeneousShellReferenceSurface,
    HomogeneousShellSectionModel, LaminatePlyModel, LaminateShellSectionModel, PlyOrderDirection,
    ShellElementModel, ShellReferenceSurface, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel, StructuralNode,
};
pub use problem::target::{
    EntityDimension, ExpectedTopology, RegionTarget, ResolvedTopologyTarget, TargetCardinality,
};
pub use study::accuracy::{
    AccuracyCriterion, AccuracyLimitAction, AccuracyPlan, AccuracyPlanId, AccuracyResourceCeiling,
    AccuracyTarget, AccuracyTolerance, ErrorContributionKind, NumericalEquivalencePolicy,
    RequiredAccuracyEvidence, UnavailableEvidencePolicy,
};
pub use study::is_exposed_analysis_identity;
pub use study::output::{
    AccuracyObservationEvent, ComplexRepresentation, CompressionPolicy, DiagnosticKind,
    DiagnosticObservation, ExtremaTiePolicy, FieldLocation, MonitorCadence, MonitorObservation,
    MonitorStopCondition, ObservationComponent, ObservationCostEstimate, ObservationCostPolicy,
    ObservationEvent, ObservationId, ObservationPurpose, ObservationQuantity, ObservationRequest,
    ObservationResultQuantity, ObservationSpec, ObservationStoragePolicy, ObservationTarget,
    ObservationThreshold, ObservationValueShape, PartialAccessPolicy, RecoveryPolicy,
    ReductionPolicy, RetentionPolicy, SamplingPolicy, SeriesObservationEvent, SeriesReduction,
    SeriesWeighting, SignConvention, SolverObservationEvent, SpatialReduction,
};
pub use study::sequence::{
    ActivationLifecycle, AnalysisSequence, IndependentDomain, InitialStateConflictPolicy,
    ProcedureSettings, ProcedureStep, ProcedureStepId, ReferenceStatePolicy, RestartPolicy,
    SequenceBranch, SequenceBranchId, SequenceDependency, SequenceDependencyReason,
    StateCompleteness, StateId, StateInput, StateKind, StateOutput, StatePersistence, StateSource,
    StateTransition, StateTransitionMethod, StateUse, StepObjectActivation, StepObjectRef,
    StructuralRandomResponseSettings, TopologyTransitionPolicy, TransitionId,
    ValidatedAnalysisSequence,
};
pub use study::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CoupledPhysicsFamily, CouplingImplementation, ExactFormulationComponent,
    ExactFormulationRealization, ExecutionFallbackPolicy, FormulationAdmissionRejectionReason,
    FormulationAxis, FormulationMaturity, FormulationRequirement, PhysicsFamily,
    RecoveryImplementation, SolutionSpace, SourceImplementation, SpatialOperatorImplementation,
    StructuralFormulationRealization, StructuralOperatorImplementation,
    StructuralRecoveryImplementation, TemporalImplementation, TopologyImplementation,
};
pub use validate::{
    validate_analysis_sequence, validate_material_definitions, validate_model,
    validate_model_against_geometry, validate_model_sequence, AnalysisValidationError,
};

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
                provenance: MaterialProvenance::Authored,
                mass_density_kg_per_m3: Some(7850.0),
                elasticity: Some(MaterialElasticityModel::IsotropicLinear(
                    MaterialMechanicalModel {
                        youngs_modulus_pa: 200e9,
                        poisson_ratio: 0.3,
                    },
                )),
                damping: None,
                thermal: Some(MaterialThermalModel {
                    reference_temperature_k: 293.15,
                    modulus_temp_coeff_per_k: -2.5e-4,
                    conductivity_w_per_mk: 45.0,
                    specific_heat_j_per_kgk: 500.0,
                    expansion_coefficient_per_k: 1.2e-5,
                }),
                acoustic: None,
                electrical: None,
                plastic: None,
            }],
            material_assignments: Vec::new(),
            structural: None,
            thermal: None,
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
            signals: vec![PhysicalSignal::Force(Signal {
                signal_id: SignalId("tip_force_signal".to_string()),
                shape: SignalShape::Vector3,
                frame: Some(CoordinateFrame::Global),
                representation: SignalRepresentation::Constant(SignalValue::Vector3([
                    Force::new(0.0),
                    Force::new(-1000.0),
                    Force::new(0.0),
                ])),
                precision: SignalPrecision::Real64,
                evidence: SignalApproximationEvidence {
                    max_absolute_error: None,
                    max_relative_error: None,
                    derivative_max_absolute_error: None,
                    sample_count: 1,
                },
                provenance: SignalProvenance::Authored,
            })],
            excitations: vec![ExcitationAction::Force(AnalysisExcitation {
                excitation_id: ExcitationId("tip_force".to_string()),
                target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::PreserveResultant,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("tip_force_signal".to_string()),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1.0e-10,
                },
            })],
            excitation_cases: vec![ExcitationCase {
                case_id: ExcitationCaseId("primary_case".to_string()),
                action_ids: vec![ExcitationId("tip_force".to_string())],
                activation: ExcitationCaseActivation::Always,
            }],
            initial_conditions: Vec::new(),
            response_spectra: Vec::new(),
            random_excitations: Vec::new(),
            sequence: AnalysisSequence::from_steps(
                "analysis_sequence",
                vec![ProcedureStep::structural_static("step_static")],
            ),
        }
    }

    #[test]
    fn missing_material_and_condition_validation_failures() {
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
    }

    #[test]
    fn solver_identity_excludes_material_reconciliation_audit_fields() {
        let mut first = valid_model();
        first.material_assignments = vec![MaterialAssignment {
            target: PropertyRegionTarget::all_solids(),
            expected_material_id: "cad_steel".into(),
            assigned_material_id: "mat_steel".into(),
            confidence: EvidenceConfidence::Verified,
        }];
        let mut second = first.clone();
        second.material_assignments[0].expected_material_id = "legacy_material_name".into();
        second.material_assignments[0].confidence = EvidenceConfidence::Inferred;
        second.materials[0].name = "Imported steel display name".into();
        second.materials[0].provenance =
            MaterialProvenance::ImportedCad(ImportedCadMaterialProvenance {
                geometry_module_path: "parts/bracket.fea".into(),
                geometry_id: "bracket".into(),
                source_material_id: "cad-material-17".into(),
                source_material_name: "STEEL-GENERIC".into(),
                source_entity_id: 17,
                persistent_id: Some("material:steel".into()),
                specification: Some("imported evidence only".into()),
                associations: Vec::new(),
                adopted_properties: Vec::new(),
            });

        assert_eq!(
            serde_json::to_vec(&first.solver_identity_contract()).unwrap(),
            serde_json::to_vec(&second.solver_identity_contract()).unwrap()
        );

        second.material_assignments[0].assigned_material_id = "another_material".into();
        assert_ne!(
            serde_json::to_vec(&first.solver_identity_contract()).unwrap(),
            serde_json::to_vec(&second.solver_identity_contract()).unwrap()
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
    fn prescribed_rotation_constraint_serializes_as_snake_case() {
        let constraint = AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "root_rotation".to_string(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary("root"),
                translations: Vec::new(),
                rotations: vec![RotationDofValue {
                    dof: StructuralRotationDof::Rz,
                    value: PrescribedValue::Constant(Rotation::new(0.125)),
                }],
            }),
        });

        let json = serde_json::to_value(&constraint).expect("constraint should serialize");
        assert_eq!(
            json["structural"]["kind"]["prescribed"]["rotations"][0]["value"]["constant"]
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
                    solid_attachment: None,
                },
                StructuralNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                    solid_attachment: None,
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
                    solid_attachment: None,
                },
                StructuralNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                    solid_attachment: None,
                },
                StructuralNode {
                    node_id: 3,
                    coordinates_m: [0.0, 1.0, 0.0],
                    solid_attachment: None,
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
            shell_sections: vec![ShellSectionModel::Homogeneous(
                HomogeneousShellSectionModel {
                    section_id: "panel_2mm".to_string(),
                    thickness_m: 0.002,
                    reference_surface: HomogeneousShellReferenceSurface::MidSurface,
                    shear_correction: 5.0 / 6.0,
                },
            )],
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
    fn electrical_frequency_response_is_explicitly_empty() {
        let electrical = MaterialElectricalModel {
            reference_temperature_k: 293.15,
            conductivity_s_per_m: 1.0,
            resistive_heating_coefficient: 0.0,
            relative_permittivity: 1.0,
            relative_permeability: 1.0,
            conductivity_frequency_response: Vec::new(),
        };
        assert!(electrical.conductivity_frequency_response.is_empty());
    }

    #[test]
    fn duplicate_structural_dofs_are_rejected() {
        let mut model = valid_model();
        model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "duplicate_ux".to_string(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary("root"),
                translations: vec![
                    TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    },
                    TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: PrescribedValue::Constant(Displacement::new(1.0)),
                    },
                ],
                rotations: Vec::new(),
            }),
        })];

        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
    }

    #[test]
    fn unknown_prescribed_signal_is_rejected() {
        let mut model = valid_model();
        model.constraints = vec![AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "temperature".to_string(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary("root"),
                value: PrescribedValue::Signal(SignalId("missing_temperature".to_string())),
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
                ambient_temperature: PrescribedValue::Constant(Temperature::new(293.15)),
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
                domain: TabulatedSignalDomain::TimeSeconds,
                knots: vec![
                    SignalKnot::continuous(
                        0.0,
                        SignalOrdinate::Real(SignalValue::Scalar(Force::new(1.0))),
                    ),
                    SignalKnot::continuous(
                        1.0,
                        SignalOrdinate::Real(SignalValue::Scalar(Force::new(2.0))),
                    ),
                    SignalKnot::continuous(
                        0.5,
                        SignalOrdinate::Real(SignalValue::Scalar(Force::new(3.0))),
                    ),
                ],
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
        )];
        assert!(matches!(
            validate_model(&model),
            Err(AnalysisValidationError::InvalidSignal { .. })
        ));

        model.signals = vec![force_signal(
            "generated_force",
            SignalRepresentation::Generator(SignalGenerator {
                invocation: GeneratorInvocation {
                    callable: GeneratorCallable::ProjectFunction {
                        qualified_name: "anonymousScript".to_string(),
                    },
                    arguments: Vec::new(),
                    effects: GeneratorEffectPolicy::Pure,
                },
                domain: GeneratorDomain::TimeSeconds,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.0,
                    end: 1.0,
                    sample_count: 101,
                },
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
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
    fn excitation_case_membership_is_owned_by_cases_and_exhaustive() {
        let mut model = valid_model();
        model.signals = vec![force_signal(
            "tip_force",
            SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
        )];
        model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
            excitation_id: ExcitationId("tip_force_action".to_string()),
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

        validate_model(&model).expect("the case action inventory is authoritative");

        model.excitation_cases.push(ExcitationCase {
            case_id: ExcitationCaseId("second_case".to_string()),
            action_ids: vec![ExcitationId("tip_force_action".to_string())],
            activation: ExcitationCaseActivation::Always,
        });
        validate_model(&model).expect("one physical action may participate in several cases");
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
    fn initial_temperature_rejects_values_below_absolute_zero() {
        let mut model = valid_model();
        model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
            initial_condition_id: "invalid_initial_temperature".to_string(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::Uniform(Temperature::new(-0.01)),
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
            ordinates: NumericalDataSource::authored(vec![5.0, 20.0], "sha256:spectrum"),
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
                correction: ModalTruncationCorrection::MissingMass {
                    zero_period_acceleration: 20.0,
                },
            },
            rigid_response: RigidResponsePolicy::LindleyYow {
                zero_period_acceleration: 20.0,
            },
            construction: SpectrumConstruction::AuthoredDesign,
            evidence: SpectrumEvidence {
                sampling_adequacy_ratio: 1.0,
                max_interpolation_error: Some(0.01),
            },
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
