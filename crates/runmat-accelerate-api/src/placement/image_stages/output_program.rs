//! Direct provider-hook requests for the source-ordered output checkpoints.
//!
//! These are physical hook requests, not an arithmetic-equivalence certificate
//! or a provider resource receipt. The source owner assigns output IDs and
//! checkpoint positions; this type only validates a forward, non-replaying
//! dependency order before a provider can consider a route.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::ProviderPrecision;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderImageOutputOperand {
    Input,
    Mean,
    Sigma,
    Produced(u64),
    ScalarF32Bits(u32),
    ScalarF64Bits(u64),
}

impl ProviderImageOutputOperand {
    fn is_tensor(self, produced: &HashSet<u64>) -> bool {
        matches!(self, Self::Input | Self::Mean | Self::Sigma)
            || matches!(self, Self::Produced(value) if produced.contains(&value))
    }

    fn is_scalar(self) -> bool {
        matches!(self, Self::ScalarF32Bits(_) | Self::ScalarF64Bits(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderImageOutputHook {
    ElemSub,
    ElemDiv,
    ElemPow,
    ElemMul,
    ElemAdd,
    ScalarAdd,
    ScalarMul,
    ScalarMax,
    CastSingle,
    CastDouble,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageOutputCall {
    /// The exact selected graph value, never an inferred slot or stage name.
    pub source_output: u64,
    pub hook: ProviderImageOutputHook,
    pub operands: Vec<ProviderImageOutputOperand>,
    /// Canonically resolved output geometry/class for this exact source step.
    pub output_shape: [u64; 3],
    pub output_precision: ProviderPrecision,
    /// An actual source store joined by the VM, not a pattern-optional guess.
    pub checkpoint: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageOutputProgram {
    pub calls: Vec<ProviderImageOutputCall>,
}

impl ProviderImageOutputProgram {
    pub fn validate(&self) -> bool {
        let mut produced = HashSet::with_capacity(self.calls.len());
        let mut has_checkpoint = false;
        for call in &self.calls {
            if produced.contains(&call.source_output)
                || call.output_shape.contains(&0)
                || call.output_shape[0]
                    .checked_mul(call.output_shape[1])
                    .and_then(|elements| elements.checked_mul(call.output_shape[2]))
                    .is_none()
            {
                return false;
            }
            let valid = match (call.hook, call.operands.as_slice()) {
                (
                    ProviderImageOutputHook::ElemSub
                    | ProviderImageOutputHook::ElemDiv
                    | ProviderImageOutputHook::ElemMul
                    | ProviderImageOutputHook::ElemAdd,
                    [left, right],
                ) => left.is_tensor(&produced) && right.is_tensor(&produced),
                (ProviderImageOutputHook::ElemPow, [base, exponent]) => {
                    base.is_tensor(&produced)
                        && (exponent.is_tensor(&produced) || exponent.is_scalar())
                }
                (
                    ProviderImageOutputHook::ScalarAdd
                    | ProviderImageOutputHook::ScalarMul
                    | ProviderImageOutputHook::ScalarMax,
                    [tensor, scalar],
                ) => tensor.is_tensor(&produced) && scalar.is_scalar(),
                (
                    ProviderImageOutputHook::CastSingle | ProviderImageOutputHook::CastDouble,
                    [tensor],
                ) => tensor.is_tensor(&produced),
                _ => false,
            };
            let cast_class_matches = match call.hook {
                ProviderImageOutputHook::CastSingle => {
                    call.output_precision == ProviderPrecision::F32
                }
                ProviderImageOutputHook::CastDouble => {
                    call.output_precision == ProviderPrecision::F64
                }
                _ => true,
            };
            if !valid || !cast_class_matches {
                return false;
            }
            has_checkpoint |= call.checkpoint;
            produced.insert(call.source_output);
        }
        has_checkpoint && self.calls.last().is_some_and(|call| call.checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_order_allows_one_or_many_checkpoints_but_not_future_inputs() {
        let first = ProviderImageOutputCall {
            source_output: 11,
            hook: ProviderImageOutputHook::ElemSub,
            operands: vec![
                ProviderImageOutputOperand::Input,
                ProviderImageOutputOperand::Mean,
            ],
            output_shape: [2, 3, 4],
            output_precision: ProviderPrecision::F32,
            checkpoint: false,
        };
        let final_call = ProviderImageOutputCall {
            source_output: 13,
            hook: ProviderImageOutputHook::ScalarMax,
            operands: vec![
                ProviderImageOutputOperand::Produced(11),
                ProviderImageOutputOperand::ScalarF32Bits(0.0f32.to_bits()),
            ],
            output_shape: [2, 3, 4],
            output_precision: ProviderPrecision::F32,
            checkpoint: true,
        };
        let one = ProviderImageOutputProgram {
            calls: vec![first.clone(), final_call.clone()],
        };
        assert!(one.validate());
        let mut multiple = one.clone();
        multiple.calls[0].checkpoint = true;
        assert!(multiple.validate());
        let mut future = one.clone();
        future.calls[0].operands[0] = ProviderImageOutputOperand::Produced(13);
        assert!(!future.validate());
        let mut duplicate = one.clone();
        duplicate.calls[1].source_output = 11;
        assert!(!duplicate.validate());
        let mut missing_final = one;
        missing_final.calls[1].checkpoint = false;
        assert!(!missing_final.validate());
    }
}
