//! Requirement declarations are durable acceptance data. Evidence runs are separate.
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
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "docs-export", serde(rename_all = "snake_case"))]
pub enum ExecutionEnvelope {
    SourceCheck,
    DesktopIntegration,
    ReleasePipeline,
    SerialCpuPreparedController,
    SerialCpuNativeProcess,
    SerialCpuNumericalKernel,
    SerialCpuPublicStudy,
    Durable,
    EncryptedRemote,
    PartitionedCpu,
    ClaimedAccelerator,
}
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CargoTarget {
    Library,
    Integration(&'static str),
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessLocator {
    CargoTest {
        package: &'static str,
        target: CargoTarget,
        name: &'static str,
        /// Relative to the runmat-analysis repository.
        source: &'static str,
    },
    GovernanceCheck {
        source: &'static str,
        check: &'static str,
    },
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct CoverageClaim {
    pub obligation: &'static str,
    pub envelopes: &'static [Envelope],
    /// What the assertions support, including important exclusions.
    pub boundary: &'static str,
}
#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct Witness {
    pub id: &'static str,
    pub locator: WitnessLocator,
    pub claims: &'static [CoverageClaim],
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
    pub witnesses: &'static [Witness],
    pub dependencies: &'static [&'static str],
}
/// Common declaration constructors keep domain inventories uniform.
pub const fn envelope(
    formulation: Formulation,
    mesh: MeshOrder,
    execution: ExecutionEnvelope,
) -> Envelope {
    Envelope {
        formulation,
        mesh,
        execution,
    }
}
pub const fn obligation(
    id: &'static str,
    criterion: &'static str,
    evidence_kind: EvidenceKind,
    required_envelopes: &'static [Envelope],
) -> Obligation {
    Obligation {
        id,
        criterion,
        evidence_kind,
        required_envelopes,
    }
}

#[cfg_attr(feature = "docs-export", derive(serde::Serialize))]
#[derive(Debug, PartialEq, Eq)]
pub enum RegistryError {
    Empty(&'static str),
    Duplicate(&'static str),
    UnknownObligation(&'static str),
    UnsupportedClaim(&'static str),
    ConflictingWitness(&'static str),
}
/// Structural integrity only. An obligation without witnesses is valid and open.
/// Nothing here executes tests, reads results or promotes a requirement.
pub fn validate<'a>(
    registry: impl IntoIterator<Item = &'a Requirement>,
) -> Result<(), RegistryError> {
    let mut requirements = std::collections::BTreeSet::new();
    let mut witnesses = std::collections::BTreeMap::new();
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
        let mut local_witnesses = std::collections::BTreeSet::new();
        for witness in requirement.witnesses {
            if witness.id.is_empty() || witness.claims.is_empty() {
                return Err(RegistryError::Empty(witness.id));
            }
            if !local_witnesses.insert(witness.id) {
                return Err(RegistryError::Duplicate(witness.id));
            }
            if let Some(prior) = witnesses.insert(witness.id, witness.locator) {
                if prior != witness.locator {
                    return Err(RegistryError::ConflictingWitness(witness.id));
                }
            }
            match witness.locator {
                WitnessLocator::CargoTest {
                    package,
                    name,
                    source,
                    ..
                } if package.is_empty() || name.is_empty() || source.is_empty() => {
                    return Err(RegistryError::Empty(witness.id))
                }
                WitnessLocator::GovernanceCheck { source, check }
                    if source.is_empty() || check.is_empty() =>
                {
                    return Err(RegistryError::Empty(witness.id))
                }
                _ => (),
            }
            for claim in witness.claims {
                let obligation = requirement
                    .obligations
                    .iter()
                    .find(|obligation| obligation.id == claim.obligation)
                    .ok_or(RegistryError::UnknownObligation(claim.obligation))?;
                if claim.boundary.is_empty() || claim.envelopes.is_empty() {
                    return Err(RegistryError::Empty(witness.id));
                }
                if claim
                    .envelopes
                    .iter()
                    .any(|envelope| !obligation.required_envelopes.contains(envelope))
                {
                    return Err(RegistryError::UnsupportedClaim(witness.id));
                }
            }
        }
    }
    Ok(())
}
