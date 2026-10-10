//! Final-only fusion results cannot replace observable VM StoreVar/StoreLocal
//! instructions. Elementwise has its separate ordered store materialization.

use crate::bytecode::Instr;
use runmat_accelerate::{FusionKind, InstrSpan};

pub fn final_only_span_requires_ordinary_stores(
    kind: &FusionKind,
    instructions: &[Instr],
    span: &InstrSpan,
) -> bool {
    final_only_span_requires_ordinary_stores_with_private(kind, instructions, span, &[])
}

pub(crate) fn final_only_span_requires_ordinary_stores_with_private(
    kind: &FusionKind,
    instructions: &[Instr],
    span: &InstrSpan,
    private_store_pcs: &[usize],
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
                slice.iter().enumerate().any(|(offset, instruction)| {
                    matches!(instruction, Instr::StoreLocal(_))
                        || matches!(instruction, Instr::StoreVar(_))
                            && !private_store_pcs.contains(&(span.start + offset))
                })
            })
        }
    }
}

#[cfg(test)]
#[path = "final_only_stores/tests.rs"]
mod tests;
