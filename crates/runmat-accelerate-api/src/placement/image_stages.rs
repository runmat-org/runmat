//! Typed provider-hook program for three staged image-normalization outputs.
//! Source bindings and numerical equivalence are proved by separate owners.
use crate::scalar_program::ScalarLiteral;
use serde::{Deserialize, Serialize};

mod moment_publication;
mod output_program;
mod power;
pub use moment_publication::{
    moment_precision, ProviderImageMomentConversion, ProviderImageMomentProgram,
    ProviderImageMomentPublication,
};
pub use output_program::{
    ProviderImageOutputCall, ProviderImageOutputHook, ProviderImageOutputOperand,
    ProviderImageOutputProgram,
};
pub use power::{ProviderImagePowerCall, ProviderImagePowerProgram};

/// Checked work inventory from the same move-owned selected stage recipes.
/// Descriptive only: it cannot authorize execution, allocation, or feedback
/// without the prepared session and its live source/provider snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderImageStagePhysicalWork {
    pub requested_bytes: u64,
    pub dispatches: u32,
    pub queue_submissions: u32,
}

impl ProviderImageStagePhysicalWork {
    pub fn validate(self) -> bool {
        self.requested_bytes > 0 && self.dispatches > 0 && self.queue_submissions > 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderImageMeanClass {
    Default,
    Native,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderImageMeanHook {
    ReduceNd([u8; 2]),
    ReduceDim(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageMeanCall {
    pub hook: ProviderImageMeanHook,
    pub output_class: ProviderImageMeanClass,
    pub shape: [u64; 3],
}

/// A fixed requested-hook program: mu reduction; subtract, square POWER,
/// variance reduction, add epsilon, sqrt; subtract, divide, final POWER.
/// Legacy literal fixtures retain F32 control slots; selected invocations
/// carry separate original-class POWER and Sigma controls. Returned handles are separately
/// published after their stage; they coexist with later scratch. This is not
/// a free-form operation list or a backend dispatch/cost assertion.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageStageWorkload {
    pub input_shape: [u64; 3],
    pub mu: Vec<ProviderImageMeanCall>,
    pub variance: Vec<ProviderImageMeanCall>,
    pub square_exponent_f32_bits: u32,
    pub final_exponent_f32_bits: u32,
    /// Versioned selected-source authority. Legacy F32 controls remain only
    /// for the held literal fixture and must never stand in for original bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_powers: Option<ProviderImagePowerProgram>,
    /// Exact original-class Sigma Add scalar. Legacy unselected execution
    /// admits only F32; selected execution is additionally checked against
    /// its canonical source and the provider's typed arithmetic proposal.
    pub epsilon: ScalarLiteral,
    /// Exact source-ordered output hooks, when a selected source owner has
    /// supplied them. The legacy three-stage workload leaves this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_program: Option<ProviderImageOutputProgram>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moment_program: Option<ProviderImageMomentProgram>,
}

impl ProviderImageStageWorkload {
    pub fn validate(&self) -> bool {
        let [batch, height, width] = self.input_shape;
        batch > 0
            && height > 0
            && width > 0
            && batch
                .checked_mul(height)
                .and_then(|n| n.checked_mul(width))
                .is_some()
            && (self.selected_powers.is_some()
                || (self.square_exponent_f32_bits == 2.0f32.to_bits()
                    && f32::from_bits(self.final_exponent_f32_bits).is_finite()))
            && self
                .selected_powers
                .as_ref()
                .is_none_or(|powers| powers.validate())
            && match (self.selected_powers.is_some(), self.epsilon) {
                (_, ScalarLiteral::F32Bits(bits)) => {
                    f32::from_bits(bits).is_finite() && f32::from_bits(bits) > 0.0
                }
                (true, ScalarLiteral::F64Bits(bits)) => {
                    f64::from_bits(bits).is_finite() && f64::from_bits(bits) > 0.0
                }
                (false, ScalarLiteral::F64Bits(_)) => false,
            }
            && self.valid_mean(&self.mu)
            && self.valid_mean(&self.variance)
            && self.moment_program.as_ref().is_none_or(|program| {
                program.validate()
                    && program.mu.len() == self.mu.len()
                    && program.variance.len() == self.variance.len()
                    && program
                        .mu
                        .iter()
                        .zip(&self.mu)
                        .chain(program.variance.iter().zip(&self.variance))
                        .all(|(publication, call)| {
                            publication
                                .raw_fact
                                .shape
                                .known_dims()
                                .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
                                .is_some_and(|dims| {
                                    dims.into_iter().map(|dim| dim as u64).eq(call.shape)
                                })
                        })
                    && program.sigma.raw_fact.shape
                        == program
                            .variance
                            .last()
                            .expect("validated route")
                            .public_fact()
                            .shape
            })
            && self.output_program.as_ref().is_none_or(|program| {
                program.validate()
                    && self.moment_program.as_ref().is_none_or(|moments| {
                        program.calls.iter().all(|call| {
                            !moments
                                .source_outputs()
                                .any(|output| output == call.source_output)
                        })
                    })
                    && program
                        .calls
                        .iter()
                        .all(|call| call.output_shape == self.input_shape)
                    && self.selected_powers.as_ref().is_none_or(|powers| {
                        program
                            .calls
                            .iter()
                            .filter(|call| call.hook == ProviderImageOutputHook::ElemPow)
                            .count()
                            == 1
                            && program.calls.iter().any(|call| {
                                call.source_output == u64::from(powers.final_power.output)
                                    && call.hook == ProviderImageOutputHook::ElemPow
                                    && call.output_precision
                                        == match powers.final_power.output_class {
                                            runmat_types::NumericClass::Single => {
                                                crate::ProviderPrecision::F32
                                            }
                                            runmat_types::NumericClass::Double => {
                                                crate::ProviderPrecision::F64
                                            }
                                            _ => return false,
                                        }
                                    && call.operands.as_slice()
                                        == [
                                            ProviderImageOutputOperand::Produced(u64::from(
                                                powers.final_power.base,
                                            )),
                                            match powers.final_power.control {
                                                crate::PowerScalarControl::SingleBits(bits) => {
                                                    ProviderImageOutputOperand::ScalarF32Bits(bits)
                                                }
                                                crate::PowerScalarControl::DoubleBits(bits) => {
                                                    ProviderImageOutputOperand::ScalarF64Bits(bits)
                                                }
                                            },
                                        ]
                            })
                    })
            })
    }

    fn valid_mean(&self, calls: &[ProviderImageMeanCall]) -> bool {
        let [batch, height, width] = self.input_shape;
        match calls {
            [ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceNd([1, 2]),
                shape: [b, 1, 1],
                ..
            }] => *b == batch,
            [first, second] => match (first.hook, second.hook) {
                (ProviderImageMeanHook::ReduceDim(1), ProviderImageMeanHook::ReduceDim(2)) => {
                    first.shape == [batch, 1, width] && second.shape == [batch, 1, 1]
                }
                (ProviderImageMeanHook::ReduceDim(2), ProviderImageMeanHook::ReduceDim(1)) => {
                    first.shape == [batch, height, 1] && second.shape == [batch, 1, 1]
                }
                _ => false,
            },
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload() -> ProviderImageStageWorkload {
        ProviderImageStageWorkload {
            input_shape: [2, 3, 4],
            mu: vec![ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceNd([1, 2]),
                output_class: ProviderImageMeanClass::Native,
                shape: [2, 1, 1],
            }],
            variance: vec![ProviderImageMeanCall {
                hook: ProviderImageMeanHook::ReduceNd([1, 2]),
                output_class: ProviderImageMeanClass::Default,
                shape: [2, 1, 1],
            }],
            square_exponent_f32_bits: 2.0f32.to_bits(),
            final_exponent_f32_bits: 1.0f32.to_bits(),
            selected_powers: None,
            epsilon: ScalarLiteral::F32Bits(0.25f32.to_bits()),
            output_program: None,
            moment_program: None,
        }
    }

    #[test]
    fn fixed_program_roundtrip_and_tamper_fail_closed() {
        let good = workload();
        assert!(good.validate());
        let wire = serde_json::to_string(&good).unwrap();
        let decoded: ProviderImageStageWorkload = serde_json::from_str(&wire).unwrap();
        assert_eq!(decoded, good);
        let mut bad = good.clone();
        bad.square_exponent_f32_bits = 1.0f32.to_bits();
        assert!(!bad.validate());
        bad = good.clone();
        bad.mu[0].shape = [2, 1, 4];
        assert!(!bad.validate());
        bad = good.clone();
        bad.variance.push(bad.variance[0]);
        assert!(!bad.validate());
        bad = good;
        bad.epsilon = ScalarLiteral::F32Bits(f32::NAN.to_bits());
        assert!(!bad.validate());
    }
}
