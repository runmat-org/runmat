use crate::excitation::ProjectionSample;

use super::ExcitationSamplePoint;

impl From<ExcitationSamplePoint<'_>> for ProjectionSample {
    fn from(sample: ExcitationSamplePoint<'_>) -> Self {
        match sample {
            ExcitationSamplePoint::Static => Self::Static,
            ExcitationSamplePoint::TimeSeconds { value, approach } => {
                Self::TimeSeconds { value, approach }
            }
            ExcitationSamplePoint::FrequencyHertz { value, approach } => {
                Self::FrequencyHertz { value, approach }
            }
            ExcitationSamplePoint::StudyParameter {
                parameter_id,
                canonical_value,
                approach,
            } => Self::StudyParameter {
                parameter_id: parameter_id.to_string(),
                canonical_value,
                approach,
            },
        }
    }
}
