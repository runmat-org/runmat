use runmat_analysis_core::{
    AnalysisProcedure, AuthoredBoundaryKind, AuthoredConnectorKind, AuthoredConstraintKind,
    AuthoredExcitation, AuthoredExcitationCase, AuthoredExcitationKind,
    AuthoredInitialConditionKind, AuthoredInputType, AuthoredInputValue, AuthoredInterfaceKind,
    AuthoredMaterialAssignment, AuthoredPropertyAssignment, CoordinateFrame,
    ExcitationDistribution, ExcitationKinematics, PhysicalQuantityKind, PhysicsFamily,
    SignalDomainKind, SignalShape,
};
use runmat_analysis_language::FeaDiagnosticLocation;
use runmat_geometry_core::{
    GeometryRegionRef, GeometryRegionRole, GeometryRegionSelector, UnitSystem,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaStudyPresentation {
    pub study_id: String,
    pub geometry_id: String,
    pub mesh_id: String,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry_units: Option<UnitSystem>,
    pub inputs: Vec<FeaStudyInputPresentation>,
    pub objects: Vec<FeaStudyObjectPresentation>,
    pub procedures: Vec<FeaProcedurePresentation>,
    pub outputs: Vec<FeaOutputPresentation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaStudyInputPresentation {
    pub input_id: String,
    pub value_type: AuthoredInputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<AuthoredInputValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<AuthoredInputValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<AuthoredInputValue>,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<FeaDiagnosticLocation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaStudyObjectPresentation {
    pub object_id: String,
    pub model_id: String,
    pub kind: FeaStudyObjectKind,
    pub details: FeaStudyObjectDetails,
    pub targets: Vec<FeaPresentationTarget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame: Option<CoordinateFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution: Option<ExcitationDistribution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kinematics: Option<ExcitationKinematics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<FeaSignalPresentation>,
    pub active_step_ids: Vec<String>,
    pub active_case_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<FeaDiagnosticLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "category", content = "kind")]
pub enum FeaStudyObjectKind {
    MaterialAssignment,
    PropertyAssignment,
    Constraint(FeaConstraintPresentationKind),
    Boundary(FeaBoundaryPresentationKind),
    Connector(FeaConnectorPresentationKind),
    Interface(FeaInterfacePresentationKind),
    Signal,
    Excitation(AuthoredExcitationKind),
    ExcitationCase,
    InitialCondition(FeaInitialConditionPresentationKind),
    ResponseSpectrum,
    RandomExcitation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "category", content = "definition")]
pub enum FeaStudyObjectDetails {
    MaterialAssignment(AuthoredMaterialAssignment),
    PropertyAssignment(AuthoredPropertyAssignment),
    Constraint(AuthoredConstraintKind),
    Boundary(AuthoredBoundaryKind),
    Connector(AuthoredConnectorKind),
    Interface(AuthoredInterfaceKind),
    Signal(FeaSignalPresentation),
    Excitation(AuthoredExcitation),
    ExcitationCase(AuthoredExcitationCase),
    InitialCondition(AuthoredInitialConditionKind),
    ResponseSpectrum(FeaResponseSpectrumPresentation),
    RandomExcitation(FeaRandomExcitationPresentation),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaConstraintPresentationKind {
    StructuralFixed,
    StructuralSymmetry,
    StructuralAntisymmetry,
    StructuralPrescribed,
    StructuralPrescribedMotion,
    StructuralNormalDisplacement,
    StructuralRigidCoupling,
    StructuralDistributingCoupling,
    StructuralTie,
    StructuralPeriodic,
    StructuralEquation,
    ThermalTemperature,
    ThermalPeriodic,
    ThermalTie,
    ThermalAffineJump,
    ElectricalPotential,
    ElectricalEquipotential,
    ElectricalFloatingConductor,
    ElectricalIntegralCurrent,
    ElectricalPotentialReference,
    ElectricalPeriodic,
    MagneticVectorPotential,
    MagneticVectorPotentialGround,
    MagneticCoulombGauge,
    MagneticPeriodic,
    MagneticPhaseShiftedPeriodic,
    AcousticPressure,
    AcousticPeriodic,
    AcousticPhaseShiftedPeriodic,
    FluidVelocity,
    FluidMovingWall,
    FluidNoSlipWall,
    FluidSlipWall,
    FluidSymmetry,
    FluidPressureReference,
    FluidPeriodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaBoundaryPresentationKind {
    StructuralPressure,
    ThermalHeatFlux,
    ThermalConvection,
    MagneticInsulation,
    AcousticRigidWall,
    AcousticRadiation,
    AcousticImpedance,
    FluidOutletPressure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaConnectorPresentationKind {
    TranslationalSpring,
    TranslationalDashpot,
    ElasticFoundation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaInterfacePresentationKind {
    ContactFrictionless,
    ContactCoulomb,
    FluidStructureNoSlip,
    ConjugateHeatTransferPerfect,
    ConjugateHeatTransferConductance,
    ConjugateHeatTransferResistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaInitialConditionPresentationKind {
    Displacement,
    Velocity,
    AngularVelocity,
    Temperature,
    ElectricPotential,
    MagneticVectorPotential,
    AcousticPressure,
    FluidPressure,
    FluidVelocity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum FeaPresentationTarget {
    GeometryRegion(FeaGeometryRegionPresentation),
    ModelInput {
        input_id: String,
        role: GeometryRegionRole,
    },
    WholeModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaGeometryRegionPresentation {
    #[serde(flatten)]
    pub region: GeometryRegionRef,
    pub selector: GeometryRegionSelector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSignalPresentation {
    pub signal_id: String,
    pub quantity: PhysicalQuantityKind,
    pub domain: SignalDomainKind,
    pub shape: SignalShape,
    pub representation: FeaSignalRepresentation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constant: Option<FeaSignalConstantPresentation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSignalConstantPresentation {
    pub values: Vec<f64>,
    pub canonical_unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaResponseSpectrumPresentation {
    pub spectrum_id: String,
    pub sample_count: usize,
    pub frame: CoordinateFrame,
    pub direction: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaRandomExcitationPresentation {
    pub excitation_id: String,
    pub frequency_sample_count: usize,
    pub channel_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeaSignalRepresentation {
    Constant,
    Tabulated,
    Waveform,
    Artifact,
    Parameterized {
        input_ids: Vec<String>,
    },
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        module_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_digest: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaProcedurePresentation {
    pub step_id: String,
    pub physics_family: PhysicsFamily,
    pub analysis_procedure: AnalysisProcedure,
    pub depends_on: Vec<String>,
    pub active_object_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<FeaDiagnosticLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaOutputPresentation {
    pub output_id: String,
    pub attributed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<FeaDiagnosticLocation>,
}
