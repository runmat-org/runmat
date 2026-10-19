//! Typed whole-study numerical results, including retained solved thermal producers.
use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub enum StructuralStudyNumericalResult {
    ThermalPrestress {
        thermal: Box<crate::analysis::ThermalStudyCompletion>,
        #[serde(skip_serializing)]
        thermal_source: ThermalStudyObjectSource,
        structural: Box<StructuralStudyNumericalResult>,
    },
    LinearSequence(StructuralStudySequenceNumericalResult),
    StructuralModal(Box<StructuralStudyModalNumericalResult>),
    StructuralModalTransient(Box<StructuralStudyModalTransientNumericalResult>),
    StructuralModalResponseSpectrum(Box<StructuralStudyModalResponseSpectrumNumericalResult>),
    StructuralRandomResponse(Box<StructuralStudyRandomResponseNumericalResult>),
    StructuralBuckling(Box<StructuralStudyBucklingNumericalResult>),
    StructuralPrestressedModal(Box<StructuralStudyPrestressedModalNumericalResult>),
    StructuralPrestressedFrequency(Box<StructuralStudyPrestressedFrequencyNumericalResult>),
    StructuralTransient(Box<StructuralStudyTransientNumericalResult>),
    StructuralFrequency(Box<StructuralStudyFrequencyNumericalResult>),
}
