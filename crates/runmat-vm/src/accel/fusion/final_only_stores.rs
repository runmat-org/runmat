//! Final-only fusion results cannot replace observable VM StoreVar/StoreLocal
//! instructions. Elementwise has its separate ordered store materialization.

use crate::bytecode::Instr;
use runmat_accelerate::{FusionKind, InstrSpan};

pub fn final_only_span_requires_ordinary_stores(
    kind: &FusionKind,
    instructions: &[Instr],
    span: &InstrSpan,
) -> bool {
    match kind {
        FusionKind::ElementwiseChain => false,
        FusionKind::Reduction
        | FusionKind::MatmulEpilogue
        | FusionKind::CenteredGram
        | FusionKind::ImageNormalize
        | FusionKind::PowerStepNormalize
        | FusionKind::ExplainedVariance => {
            instructions.get(span.start..=span.end).is_none_or(|slice| {
                slice.iter().any(|instruction| {
                    matches!(instruction, Instr::StoreVar(_) | Instr::StoreLocal(_))
                })
            })
        }
    }
}

#[cfg(test)]
#[path = "final_only_stores/tests.rs"]
mod tests;
