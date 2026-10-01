//! The reduced harmonic pipeline must never reinterpret static physical intent.
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
        Some(ElectromagneticDomain::Magnetostatic { .. }) => Err(FeaRunError::InvalidModel(
            "legacy harmonic electromagnetic solve does not admit magnetostatic intent".into(),
        )),
        _ => Err(FeaRunError::InvalidModel(
            "legacy harmonic electromagnetic solve requires a finite positive harmonic frequency"
                .into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_core::MagnetostaticFormulation;

    #[test]
    fn magnetostatic_intent_is_not_a_zero_frequency_harmonic_problem() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
            formulation: MagnetostaticFormulation::MagneticVectorPotential,
        });
        let error = harmonic_frequency(&model).unwrap_err();
        assert!(error
            .to_string()
            .contains("does not admit magnetostatic intent"));
    }
}
