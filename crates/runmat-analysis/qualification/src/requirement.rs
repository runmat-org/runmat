//! Requirement declarations are durable acceptance data, kept in the capability ledger
//! (docs/simulation/ledger.json) and generated into Rust by build.rs. Evidence runs are separate.
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "docs-export", serde(rename_all = "snake_case"))]
pub enum Stage {
    Prerequisite,
    Baseline,
    Authoring,
    ExecutionContracts,
    SharedAuthorities,
    LinearStructures,
    Dynamics,
    Thermal,
    NonlinearContact,
    Electromagnetics,
    Acoustics,
    Fluid,
    FluidStructureInteraction,
    Adaptation,
    Release,
    Meshing,
    Scale,
    PublicSurface,
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "docs-export", serde(rename_all = "snake_case"))]
pub enum Formulation {
    SharedInfrastructure,
    LinearStructural,
    ModalStructural,
    PrestressedStructural,
    StructuralStability,
    LinearStructuralDynamics,
    NonlinearStructural,
    PenaltyCoulombContact,
    AdaptiveAnalysis,
    Electrostatic,
    ElectricalConduction,
    Magnetostatic,
    HarmonicMaxwell,
    Thermoelastic,
    Electrothermal,
    ConjugateHeatTransfer,
    FluidStructureInteraction,
    SmallStrainJ2,
    SmallStrainJ2PenaltyCoulombContact,
    ThermalSteady,
    ThermalTransient,
    AcousticHarmonic,
    IncompressibleLaminarSteady,
    IncompressibleLaminarTransient,
    AcousticModal,
    AcousticTransient,
    AcousticStructureCoupled,
    ElectromagneticTransient,
    SaturatingMagnetics,
    LowMachFlow,
    GeneralizedNewtonianFlow,
    ViscoelasticFlow,
    PorousMediaFlow,
    ScalarTransport,
    FreeSurfaceFlow,
    RotatingFrameFlow,
    ThinFilmLubrication,
    Electrokinetics,
    ParticleTracing,
    CompressibleLaminarFlow,
    PipeNetwork,
    QuasiStaticTem,
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "docs-export", serde(rename_all = "snake_case"))]
pub enum MeshOrder {
    MeshIndependent,
    Beam,
    Shell,
    EdgeElements,
    CanonicalMixedTopology,
    Tet4,
    Tet10,
    CanonicalTetrahedra,
    Tri3,
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "docs-export", serde(rename_all = "snake_case"))]
pub enum ExecutionEnvelope {
    SourceCheck,
    DesktopIntegration,
    ReleasePipeline,
    PreparedController,
    NativeProcess,
    NumericalKernel,
    PublicStudy,
    Durable,
    ParallelCpu,
    RemoteExecution,
    DomainDecomposed,
    ClaimedAccelerator,
}
impl ExecutionEnvelope {
    pub const ALL: &'static [ExecutionEnvelope] = &[
        ExecutionEnvelope::SourceCheck,
        ExecutionEnvelope::DesktopIntegration,
        ExecutionEnvelope::ReleasePipeline,
        ExecutionEnvelope::PreparedController,
        ExecutionEnvelope::NativeProcess,
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::Durable,
        ExecutionEnvelope::ParallelCpu,
        ExecutionEnvelope::RemoteExecution,
        ExecutionEnvelope::DomainDecomposed,
        ExecutionEnvelope::ClaimedAccelerator,
    ];
}

/// Wire name of a register enum value: the snake_case form of its variant name. This is
/// the spelling used in witness tags and in exported JSON.
pub fn snake_name(debug: &str) -> String {
    let mut out = String::with_capacity(debug.len() + 4);
    for (index, character) in debug.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index != 0 {
                out.push('_');
            }
            out.push(character.to_ascii_lowercase());
        } else {
            out.push(character);
        }
    }
    out
}
macro_rules! wire_names {
    ($name:ident) => {
        impl $name {
            pub fn wire_name(self) -> String {
                snake_name(&format!("{self:?}"))
            }
            pub fn parse(text: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|v| v.wire_name() == text)
            }
        }
    };
}
impl Stage {
    pub const ALL: &'static [Stage] = &[
        Stage::Prerequisite,
        Stage::Baseline,
        Stage::Authoring,
        Stage::ExecutionContracts,
        Stage::SharedAuthorities,
        Stage::LinearStructures,
        Stage::Dynamics,
        Stage::Thermal,
        Stage::NonlinearContact,
        Stage::Electromagnetics,
        Stage::Acoustics,
        Stage::Fluid,
        Stage::FluidStructureInteraction,
        Stage::Adaptation,
        Stage::Release,
        Stage::Meshing,
        Stage::Scale,
        Stage::PublicSurface,
    ];
}
wire_names!(Stage);
impl Formulation {
    pub const ALL: &'static [Formulation] = &[
        Formulation::SharedInfrastructure,
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::PrestressedStructural,
        Formulation::StructuralStability,
        Formulation::LinearStructuralDynamics,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::AdaptiveAnalysis,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::SmallStrainJ2,
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::AcousticHarmonic,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::AcousticStructureCoupled,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::LowMachFlow,
        Formulation::GeneralizedNewtonianFlow,
        Formulation::ViscoelasticFlow,
        Formulation::PorousMediaFlow,
        Formulation::ScalarTransport,
        Formulation::FreeSurfaceFlow,
        Formulation::RotatingFrameFlow,
        Formulation::ThinFilmLubrication,
        Formulation::Electrokinetics,
        Formulation::ParticleTracing,
        Formulation::CompressibleLaminarFlow,
        Formulation::PipeNetwork,
        Formulation::QuasiStaticTem,
    ];
}
wire_names!(Formulation);
impl MeshOrder {
    pub const ALL: &'static [MeshOrder] = &[
        MeshOrder::MeshIndependent,
        MeshOrder::Beam,
        MeshOrder::Shell,
        MeshOrder::EdgeElements,
        MeshOrder::CanonicalMixedTopology,
        MeshOrder::Tet4,
        MeshOrder::Tet10,
        MeshOrder::CanonicalTetrahedra,
        MeshOrder::Tri3,
    ];
}
wire_names!(MeshOrder);
wire_names!(ExecutionEnvelope);
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub formulation: Formulation,
    pub mesh: MeshOrder,
    pub execution: ExecutionEnvelope,
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "docs-export", serde(rename_all = "snake_case"))]
pub enum EvidenceKind {
    OriginalStateComparison,
    RejectionInvariant,
    ColdReplay,
    ControllerChronology,
    ProcessLifecycle,
    IndependentReference,
    SpatialConvergence,
    TemporalConvergence,
    PhysicalBalance,
    SemanticRealization,
    ExecutionEquivalence,
    AccuracyAcceptance,
    ResourceAdmission,
    PublicationIntegrity,
    /// Measured wall time, memory or scaling on stated hardware with the method recorded.
    PerformanceMeasurement,
    /// The public command runs the published study to its published outputs.
    PublicRun,
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct Obligation {
    pub id: &'static str,
    pub criterion: &'static str,
    pub evidence_kind: EvidenceKind,
    pub required_envelopes: &'static [Envelope],
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct Requirement {
    pub id: &'static str,
    pub title: &'static str,
    pub stages: &'static [Stage],
    pub acceptance: &'static str,
    pub owners: &'static [&'static str],
    pub obligations: &'static [Obligation],
    pub dependencies: &'static [&'static str],
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Debug, PartialEq, Eq)]
pub enum RegistryError {
    Empty(&'static str),
    Duplicate(&'static str),
}
/// Structural integrity of the declarations only. Witnesses are declared at the tests
/// (see `witness`) and validated there; nothing here executes tests.
pub fn validate<'a>(
    registry: impl IntoIterator<Item = &'a Requirement>,
) -> Result<(), RegistryError> {
    let mut requirements = std::collections::BTreeSet::new();
    for requirement in registry {
        if requirement.id.is_empty()
            || requirement.title.is_empty()
            || requirement.acceptance.is_empty()
            || requirement.stages.is_empty()
            || requirement.owners.is_empty()
            || requirement.obligations.is_empty()
        {
            return Err(RegistryError::Empty(requirement.id));
        }
        if !requirements.insert(requirement.id) {
            return Err(RegistryError::Duplicate(requirement.id));
        }
        if requirement.owners.iter().any(|p| p.is_empty()) {
            return Err(RegistryError::Empty(requirement.id));
        }
        let mut obligations = std::collections::BTreeSet::new();
        for obligation in requirement.obligations {
            if obligation.id.is_empty()
                || obligation.criterion.is_empty()
                || obligation.required_envelopes.is_empty()
            {
                return Err(RegistryError::Empty(obligation.id));
            }
            if !obligations.insert(obligation.id) {
                return Err(RegistryError::Duplicate(obligation.id));
            }
            for (i, envelope) in obligation.required_envelopes.iter().enumerate() {
                if obligation.required_envelopes[..i].contains(envelope) {
                    return Err(RegistryError::Duplicate(obligation.id));
                }
            }
        }
    }
    Ok(())
}
