//! Group already selected, PC-ordered fusion signals into executable windows.
//! Candidate ownership is decided by the caller, not by this grouping step.

use crate::bytecode::{FusionInstructionKind, FusionInstructionWindow};

#[derive(Default)]
pub(crate) struct FusionWindowGroup {
    current: Option<FusionInstructionWindow>,
}

impl FusionWindowGroup {
    pub(crate) fn push(
        &mut self,
        pc: usize,
        kind: FusionInstructionKind,
        windows: &mut Vec<FusionInstructionWindow>,
    ) {
        if let Some(window) = self.current.as_mut() {
            if window.span.end.checked_add(1) == Some(pc) {
                window.span.end = pc;
                if kind == FusionInstructionKind::Matmul
                    || (kind == FusionInstructionKind::Reduction
                        && window.kind != FusionInstructionKind::Matmul)
                {
                    window.kind = kind;
                }
                return;
            }
        }
        self.finish(windows);
        self.current = Some(FusionInstructionWindow {
            span: runmat_accelerate::graph::InstrSpan { start: pc, end: pc },
            kind,
        });
    }

    pub(crate) fn finish(&mut self, windows: &mut Vec<FusionInstructionWindow>) {
        if let Some(window) = self.current.take() {
            windows.push(window);
        }
    }
}
