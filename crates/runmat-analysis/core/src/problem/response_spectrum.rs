use serde::{Deserialize, Serialize};

use super::{
    coordinate_frame::CoordinateFrame,
    signal::{NumericalDataSource, SignalId},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResponseSpectrumId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpectrum {
    pub spectrum_id: ResponseSpectrumId,
    pub abscissa: SpectrumAbscissa,
    pub ordinates: NumericalDataSource<Vec<f64>>,
    pub damping: SpectrumDamping,
    pub response: SpectrumResponse,
    pub envelope: SpectrumEnvelope,
    pub interpolation: SpectrumInterpolation,
    pub extrapolation: SpectrumExtrapolation,
    pub direction: SpectrumDirection,
    pub modal_combination: ModalCombination,
    pub directional_combination: DirectionalCombination,
    pub truncation: ModalTruncationPolicy,
    pub rigid_response: RigidResponsePolicy,
    pub construction: SpectrumConstruction,
    pub evidence: SpectrumEvidence,
}

impl ResponseSpectrum {
    pub fn resolved_ordinates(&self) -> Result<&[f64], String> {
        self.ordinates.values().map(Vec::as_slice).ok_or_else(|| {
            format!(
                "response spectrum `{}` must be materialized before numerical use",
                self.spectrum_id.0
            )
        })
    }

    pub fn validate_intrinsic(&self) -> Result<(), String> {
        if self.spectrum_id.0.trim().is_empty() {
            return Err("response-spectrum id must not be empty".to_string());
        }
        let axis = self.abscissa.canonical_points();
        if (matches!(&self.abscissa, SpectrumAbscissa::FrequencyHertz(_))
            && self.interpolation == SpectrumInterpolation::LinearPeriod)
            || (matches!(&self.abscissa, SpectrumAbscissa::PeriodSeconds(_))
                && self.interpolation == SpectrumInterpolation::LinearFrequency)
        {
            return Err("spectrum interpolation must match its abscissa".to_string());
        }
        if axis.is_empty()
            || self
                .ordinates
                .values()
                .is_some_and(|ordinates| axis.len() != ordinates.len())
            || axis.iter().any(|value| !value.is_finite() || *value <= 0.0)
            || axis.windows(2).any(|pair| pair[0] >= pair[1])
            || self.ordinates.values().is_some_and(|ordinates| {
                ordinates
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Err(
                "spectrum axes and ordinates must be aligned, positive, and increasing".to_string(),
            );
        }
        match self.damping {
            SpectrumDamping::Ratio(value) if !value.is_finite() || !(0.0..1.0).contains(&value) => {
                return Err("damping ratio must be between zero and one".to_string())
            }
            SpectrumDamping::QualityFactor(value) if !value.is_finite() || value <= 0.0 => {
                return Err("quality factor must be finite and positive".to_string())
            }
            _ => {}
        }
        if let DirectionalCombination::PercentageRule { primary, secondary } =
            self.directional_combination
        {
            if !primary.is_finite()
                || !secondary.is_finite()
                || !(0.0..=1.0).contains(&primary)
                || !(0.0..=1.0).contains(&secondary)
                || primary < secondary
            {
                return Err(
                    "directional percentage coefficients must be finite fractions with primary not less than secondary"
                        .to_string(),
                );
            }
        }
        if self.interpolation == SpectrumInterpolation::LogLog
            && self
                .ordinates
                .values()
                .is_some_and(|ordinates| ordinates.iter().any(|value| *value <= 0.0))
        {
            return Err("log-log spectra require positive ordinates".to_string());
        }
        crate::validate_coordinate_frame(&self.direction.frame)
            .map_err(|error| error.to_string())?;
        if self.direction.vector.iter().any(|value| !value.is_finite())
            || self
                .direction
                .vector
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                <= f64::EPSILON
        {
            return Err("spectrum direction must be finite and nonzero".to_string());
        }
        if !self.truncation.minimum_cumulative_mass_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.truncation.minimum_cumulative_mass_ratio)
            || !self.evidence.sampling_adequacy_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.evidence.sampling_adequacy_ratio)
            || self
                .evidence
                .max_interpolation_error
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("spectrum policy, evidence, and provenance must be valid".to_string());
        }
        self.ordinates.validate_provenance()?;
        match self.truncation.correction {
            ModalTruncationCorrection::None | ModalTruncationCorrection::ResidualMode => {}
            ModalTruncationCorrection::MissingMass {
                zero_period_acceleration,
            } => {
                if !zero_period_acceleration.is_finite()
                    || zero_period_acceleration < 0.0
                    || !matches!(
                        self.response,
                        SpectrumResponse::AbsoluteAcceleration
                            | SpectrumResponse::PseudoAcceleration
                    )
                {
                    return Err(
                        "missing-mass correction requires a finite nonnegative ZPA and an acceleration spectrum"
                            .to_string(),
                    );
                }
            }
        }
        match self.rigid_response {
            RigidResponsePolicy::Disabled => {}
            RigidResponsePolicy::Gupta {
                periodic_frequency_hz,
                rigid_frequency_hz,
            } => {
                if !periodic_frequency_hz.is_finite()
                    || !rigid_frequency_hz.is_finite()
                    || periodic_frequency_hz <= 0.0
                    || rigid_frequency_hz <= periodic_frequency_hz
                {
                    return Err(
                        "Gupta rigid response requires positive ordered periodic and rigid frequencies"
                            .to_string(),
                    );
                }
            }
            RigidResponsePolicy::LindleyYow {
                zero_period_acceleration,
            } if !zero_period_acceleration.is_finite() || zero_period_acceleration < 0.0 => {
                return Err(
                    "Lindley-Yow rigid response requires a finite nonnegative ZPA".to_string(),
                );
            }
            RigidResponsePolicy::LindleyYow { .. } => {}
        }
        if self.rigid_response != RigidResponsePolicy::Disabled
            && (!matches!(
                self.response,
                SpectrumResponse::AbsoluteAcceleration | SpectrumResponse::PseudoAcceleration
            ) || self.modal_combination != ModalCombination::CompleteQuadraticCombination)
        {
            return Err(
                "rigid-response separation requires an acceleration spectrum and CQC modal combination"
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumAbscissa {
    FrequencyHertz(
        #[serde(with = "runmat_types::physical::canonical_storage")]
        runmat_types::physical::Quantity<runmat_types::physical::kind::Frequency, Vec<f64>>,
    ),
    PeriodSeconds(
        #[serde(with = "runmat_types::physical::canonical_storage")]
        runmat_types::physical::Quantity<runmat_types::physical::kind::Time, Vec<f64>>,
    ),
}

impl SpectrumAbscissa {
    /// Borrow numerical stencil coordinates while retaining their axis owner.
    pub fn canonical_points(&self) -> &[f64] {
        match self {
            Self::FrequencyHertz(values) => values.canonical_values(),
            Self::PeriodSeconds(values) => values.canonical_values(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumDamping {
    Ratio(f64),
    QualityFactor(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumResponse {
    RelativeDisplacement,
    RelativeVelocity,
    RelativeAcceleration,
    AbsoluteAcceleration,
    PseudoVelocity,
    PseudoAcceleration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumEnvelope {
    Positive,
    Negative,
    Maximax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumInterpolation {
    LinearFrequency,
    LinearPeriod,
    LogLog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumExtrapolation {
    Reject,
    HoldEndpoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectrumDirection {
    pub frame: CoordinateFrame,
    pub vector: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalCombination {
    Srss,
    CompleteQuadraticCombination,
    TenPercent,
    AbsoluteSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionalCombination {
    Srss,
    AbsoluteSum,
    PercentageRule { primary: f64, secondary: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumConstruction {
    AuthoredDesign,
    DerivedFromAccelerationHistory {
        source_signal_id: SignalId,
        method: ShockResponseSpectrumMethod,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShockResponseSpectrumMethod {
    ExactPiecewiseLinear,
    RecursiveFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalTruncationPolicy {
    pub minimum_cumulative_mass_ratio: f64,
    pub correction: ModalTruncationCorrection,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalTruncationCorrection {
    None,
    ResidualMode,
    MissingMass { zero_period_acceleration: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RigidResponsePolicy {
    Disabled,
    Gupta {
        periodic_frequency_hz: f64,
        rigid_frequency_hz: f64,
    },
    LindleyYow {
        zero_period_acceleration: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectrumEvidence {
    pub sampling_adequacy_ratio: f64,
    pub max_interpolation_error: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CoordinateFrame;

    fn spectrum() -> ResponseSpectrum {
        ResponseSpectrum {
            spectrum_id: ResponseSpectrumId("design".into()),
            abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Frequency,
                _,
            >::new(vec![1.0, 10.0])),
            ordinates: NumericalDataSource::authored(vec![2.0, 1.0], "sha256:design"),
            damping: SpectrumDamping::Ratio(0.05),
            response: SpectrumResponse::AbsoluteAcceleration,
            envelope: SpectrumEnvelope::Maximax,
            interpolation: SpectrumInterpolation::LinearFrequency,
            extrapolation: SpectrumExtrapolation::Reject,
            direction: SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector: [1.0, 0.0, 0.0],
            },
            modal_combination: ModalCombination::CompleteQuadraticCombination,
            directional_combination: DirectionalCombination::Srss,
            truncation: ModalTruncationPolicy {
                minimum_cumulative_mass_ratio: 0.9,
                correction: ModalTruncationCorrection::MissingMass {
                    zero_period_acceleration: 1.0,
                },
            },
            rigid_response: RigidResponsePolicy::LindleyYow {
                zero_period_acceleration: 1.0,
            },
            construction: SpectrumConstruction::AuthoredDesign,
            evidence: SpectrumEvidence {
                sampling_adequacy_ratio: 1.0,
                max_interpolation_error: None,
            },
        }
    }

    #[test]
    fn correction_policies_are_typed_and_physically_compatible() {
        let mut value = spectrum();
        value.validate_intrinsic().unwrap();

        value.response = SpectrumResponse::RelativeDisplacement;
        assert!(value.validate_intrinsic().is_err());
        value.response = SpectrumResponse::AbsoluteAcceleration;
        value.modal_combination = ModalCombination::Srss;
        assert!(value.validate_intrinsic().is_err());
        value.modal_combination = ModalCombination::CompleteQuadraticCombination;
        value.rigid_response = RigidResponsePolicy::Gupta {
            periodic_frequency_hz: 10.0,
            rigid_frequency_hz: 5.0,
        };
        assert!(value.validate_intrinsic().is_err());
    }
}
