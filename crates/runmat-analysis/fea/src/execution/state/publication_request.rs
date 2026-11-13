use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticStatePublicationRequest {
    pub schema_version: u16,
    pub source: SemanticStatePublicationSource,
    pub producer: StepIdentity,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
}

/// Typed interpretation of the solver result consumed by state publication.
///
/// `SolverResult` is deliberately a generic transport object kind. This discriminator prevents a
/// worker from guessing its schema or accepting one procedure's result as another state kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticStatePublicationSource {
    NonlinearSolidOriginalState,
    FluidPrimaryFields,
    StructuralPrimaryFields,
    ThermalPrimaryField,
    AcousticNativeFields,
    AcousticResponseFields,
    StaticElectricalNativeFields,
    MagnetostaticNativeFields,
    MaxwellNativeFields,
    StructuralLinearization,
    ModalBasis,
    StructuralResponseEnvelope,
    StructuralRandomStatistics,
}

impl SemanticStatePublicationRequest {
    pub fn new(
        source: SemanticStatePublicationSource,
        producer: StepIdentity,
        declaration: StateOutput,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 1,
            source,
            producer,
            declaration,
            topology_digest,
            source_result,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.declaration.state_id.0.trim().is_empty()
            || self.declaration.completeness != StateCompleteness::Complete
            || self.declaration.persistence == StatePersistence::CheckpointOnly
            || !matches!(
                (self.source, self.declaration.kind),
                (
                    SemanticStatePublicationSource::NonlinearSolidOriginalState,
                    StateKind::MaterialHistory
                        | StateKind::StructuralPrimary
                        | StateKind::StructuralStressStrain
                ) | (
                    SemanticStatePublicationSource::FluidPrimaryFields,
                    StateKind::Fluid | StateKind::CoupledInterface
                ) | (
                    SemanticStatePublicationSource::StructuralPrimaryFields,
                    StateKind::StructuralPrimary
                ) | (
                    SemanticStatePublicationSource::ThermalPrimaryField,
                    StateKind::Thermal
                ) | (
                    SemanticStatePublicationSource::AcousticNativeFields,
                    StateKind::AcousticComplex
                ) | (
                    SemanticStatePublicationSource::AcousticResponseFields,
                    StateKind::AcousticModes | StateKind::AcousticPressure
                ) | (
                    SemanticStatePublicationSource::StaticElectricalNativeFields,
                    StateKind::ElectromagneticStatic
                ) | (
                    SemanticStatePublicationSource::MagnetostaticNativeFields,
                    StateKind::ElectromagneticStatic
                ) | (
                    SemanticStatePublicationSource::MaxwellNativeFields,
                    StateKind::ElectromagneticComplex
                ) | (
                    SemanticStatePublicationSource::StructuralLinearization,
                    StateKind::GeometricStiffness
                ) | (
                    SemanticStatePublicationSource::ModalBasis,
                    StateKind::ModalBasis
                ) | (
                    SemanticStatePublicationSource::StructuralResponseEnvelope,
                    StateKind::StructuralResponseEnvelope
                ) | (
                    SemanticStatePublicationSource::StructuralRandomStatistics,
                    StateKind::StructuralRandomStatistics
                )
            )
        {
            return Err(AnalysisExecutionError::Invalid(
                "semantic-state publication source does not match its complete persisted declaration"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            STATE_PREFIX,
            "semantic-state-publication-request",
            self,
            STATE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request: Self = decode_contract(
            STATE_PREFIX,
            "semantic-state-publication-request",
            bytes,
            STATE_LIMITS,
        )?;
        request.validate()?;
        Ok(request)
    }
}
