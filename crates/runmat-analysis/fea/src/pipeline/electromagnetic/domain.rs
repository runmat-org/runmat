//! The reduced harmonic pipeline must never reinterpret static electrical intent.
use crate::contracts::FeaRunError;
use runmat_analysis_core::{AnalysisModel, ElectromagneticDomain, Frequency};

pub(crate) fn harmonic_frequency(model: &AnalysisModel) -> Result<Frequency, FeaRunError> {
    match &model.electromagnetic {
        Some(ElectromagneticDomain::Harmonic {
            reference_frequency,
        }) if reference_frequency.is_finite() && reference_frequency.hertz > 0.0 => {
            Ok(*reference_frequency)
        }
        Some(ElectromagneticDomain::Static { .. }) => Err(FeaRunError::InvalidModel(
            "legacy harmonic electromagnetic solve does not admit static electrical intent".into(),
        )),
        _ => Err(FeaRunError::InvalidModel(
            "legacy harmonic electromagnetic solve requires a finite positive harmonic frequency"
                .into(),
        )),
    }
}
