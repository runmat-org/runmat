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
    /// Modal extraction followed by a modal-reduced frequency sweep (`modal_harmonic`).
    StructuralModalFrequency(Box<StructuralStudyModalFrequencyNumericalResult>),
    StructuralModalResponseSpectrum(Box<StructuralStudyModalResponseSpectrumNumericalResult>),
    StructuralRandomResponse(Box<StructuralStudyRandomResponseNumericalResult>),
    StructuralBuckling(Box<StructuralStudyBucklingNumericalResult>),
    StructuralPrestressedModal(Box<StructuralStudyPrestressedModalNumericalResult>),
    StructuralPrestressedFrequency(Box<StructuralStudyPrestressedFrequencyNumericalResult>),
    StructuralTransient(Box<StructuralStudyTransientNumericalResult>),
    StructuralFrequency(Box<StructuralStudyFrequencyNumericalResult>),
}
