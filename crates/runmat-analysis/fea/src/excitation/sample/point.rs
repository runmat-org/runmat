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

impl<'a> TryFrom<&'a ProjectionSample> for ExcitationSamplePoint<'a> {
    type Error = super::super::RealizationError;
    fn try_from(sample: &'a ProjectionSample) -> Result<Self, Self::Error> {
        let invalid = || Self::Error::new("invalid canonical excitation sample point");
        Ok(match sample {
            ProjectionSample::Static => Self::Static,
            ProjectionSample::TimeSeconds { value, approach }
                if value.is_finite() && !(*value == 0.0 && value.is_sign_negative()) =>
            {
                Self::TimeSeconds {
                    value: *value,
                    approach: *approach,
                }
            }
            ProjectionSample::FrequencyHertz { value, approach }
                if value.is_finite()
                    && *value >= 0.0
                    && !(*value == 0.0 && value.is_sign_negative()) =>
            {
                Self::FrequencyHertz {
                    value: *value,
                    approach: *approach,
                }
            }
            ProjectionSample::StudyParameter {
                parameter_id,
                canonical_value,
                approach,
            } if !parameter_id.trim().is_empty() && canonical_value.is_finite() => {
                Self::StudyParameter {
                    parameter_id,
                    canonical_value: *canonical_value,
                    approach: *approach,
                }
            }
            _ => return Err(invalid()),
        })
    }
}
