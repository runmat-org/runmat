//! Typed semantic-state payload inventory and variant accessors.

use runmat_analysis_core::AnalysisField;
use serde::{Deserialize, Serialize};

use super::{
    CanonicalAcousticStatePayload, CanonicalElectricalStaticStatePayload,
    CanonicalThermalStatePayload,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum CanonicalSemanticStatePayload {
    Fields {
        fields: Vec<AnalysisField>,
    },
    Thermal {
        state: Box<CanonicalThermalStatePayload>,
    },
    Acoustic {
        state: Box<CanonicalAcousticStatePayload>,
    },
    ElectricalStatic {
        state: Box<CanonicalElectricalStaticStatePayload>,
    },
    StructuralFrequency {
        publication: Box<crate::execution::StructuralFrequencyPublicationManifest>,
    },
    ModalBasis {
        basis: Box<crate::pipeline::modal_basis::ModalBasisArtifact>,
    },
    StructuralLinearization {
        linearization:
            Box<crate::pipeline::structural_linearization::StructuralLinearizationArtifact>,
    },
    StructuralResponseEnvelope {
        envelope: Box<crate::execution::StructuralResponseEnvelope>,
    },
    StructuralRandomStatistics {
        statistics: Box<crate::execution::StructuralRandomStatistics>,
    },
}

impl CanonicalSemanticStatePayload {
    pub fn electrical_static(&self) -> Option<&CanonicalElectricalStaticStatePayload> {
        match self {
            Self::ElectricalStatic { state } => Some(state.as_ref()),
            _ => None,
        }
    }
    pub fn fields(&self) -> Option<&[AnalysisField]> {
        match self {
            Self::Fields { fields } => Some(fields),
            Self::Thermal { state } => Some(std::slice::from_ref(&state.temperature)),
            _ => None,
        }
    }

    pub fn structural_frequency(
        &self,
    ) -> Option<&crate::execution::StructuralFrequencyPublicationManifest> {
        match self {
            Self::StructuralFrequency { publication } => Some(publication.as_ref()),
            _ => None,
        }
    }

    pub fn modal_basis(&self) -> Option<&crate::pipeline::modal_basis::ModalBasisArtifact> {
        match self {
            Self::ModalBasis { basis } => Some(basis.as_ref()),
            _ => None,
        }
    }

    pub fn structural_linearization(
        &self,
    ) -> Option<&crate::pipeline::structural_linearization::StructuralLinearizationArtifact> {
        match self {
            Self::StructuralLinearization { linearization } => Some(linearization.as_ref()),
            _ => None,
        }
    }

    pub fn structural_response_envelope(
        &self,
    ) -> Option<&crate::execution::StructuralResponseEnvelope> {
        match self {
            Self::StructuralResponseEnvelope { envelope } => Some(envelope.as_ref()),
            _ => None,
        }
    }

    pub fn structural_random_statistics(
        &self,
    ) -> Option<&crate::execution::StructuralRandomStatistics> {
        match self {
            Self::StructuralRandomStatistics { statistics } => Some(statistics.as_ref()),
            _ => None,
        }
    }

    pub fn acoustic(&self) -> Option<&CanonicalAcousticStatePayload> {
        match self {
            Self::Acoustic { state } => Some(state),
            _ => None,
        }
    }
}
