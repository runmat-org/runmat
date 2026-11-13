//! Function-local fusion metadata uses the same MIR gate and instruction
//! window derivation as entrypoint bytecode, but is never borrowed from an
//! unrelated entrypoint or assembled from test-only graph guesses.

use super::{derive_semantic_fusion_instruction_windows, derive_semantic_fusion_metadata};
use crate::bytecode::{
    program::{
        selected_function_windows, FunctionFusionCertificateIndex, FunctionFusionScaffold,
        FunctionPrivateSlotLayout,
    },
    FunctionBytecode, FusionCandidateGroup, Instr,
};
use runmat_mir::analysis::FusionStageFactStatus;
use runmat_types::ProgramPointId;

pub(super) fn derive_function_scaffold(
    mir: &runmat_mir::MirAssembly,
    function: runmat_hir::FunctionId,
    layout: &crate::layout::VmFunctionLayout,
    resume_points: &std::collections::BTreeMap<runmat_types::ProgramPointId, usize>,
    projection: &crate::bytecode::FusionStageProjection,
    instructions: &[Instr],
    instr_spans: &[runmat_hir::Span],
) -> Option<FunctionFusionScaffold> {
    let metadata = derive_semantic_fusion_metadata(mir, Some(function));
    if metadata.mir_fusion_candidate_group_count == 0 {
        return None;
    }
    let mut windows = derive_semantic_fusion_instruction_windows(
        instructions,
        instr_spans,
        &metadata.mir_fusion_candidate_groups,
    );
    windows.sort_by_key(|window| (window.span.start, window.span.end));
    let scaffold = FunctionFusionScaffold {
        signal_count: metadata.mir_fusion_signal_count,
        candidates: metadata.mir_fusion_candidate_groups,
        windows,
        private_layout: FunctionPrivateSlotLayout::from_compiler(
            layout,
            resume_points,
            projection.portable_origin,
        ),
    };
    (scaffold.valid_for(function, instructions, instr_spans)
        && scaffold.source_windows_match(instructions, instr_spans))
    .then_some(scaffold)
}

/// The MIR run finder intentionally sees structural signals before selected
/// statement effects. Slice only compiler-produced hints here, against the
/// exact function stream and the same strict certificate used by ProgramV2.
/// A failing hint never becomes artifact authority, but valid neighboring
/// groups remain eligible for runtime admission.
pub(super) fn certify_compiled_function_scaffold(
    body: &mut FunctionBytecode,
    mir: &runmat_mir::MirAssembly,
) -> Result<(), String> {
    let Some(original) = body.fusion_scaffold.take() else {
        return Ok(());
    };
    let owner = body
        .fusion_stage_projection
        .as_ref()
        .ok_or_else(|| "compiler fusion scaffold has no selected statement projection".to_string())?
        .owner;
    let mir_body = mir
        .bodies
        .get(&body.function)
        .ok_or_else(|| "compiler fusion scaffold has no matching MIR function body".to_string())?;
    let blocks = mir_body
        .blocks
        .iter()
        .map(|block| (block.id, block))
        .collect::<std::collections::HashMap<_, _>>();
    let accepted = {
        let index = FunctionFusionCertificateIndex::new(body)?;
        let mut accepted = Vec::new();
        for candidate in &original.candidates {
            let block = blocks
                .get(&candidate.block)
                .copied()
                .ok_or_else(|| "compiler fusion candidate has no matching MIR block".to_string())?;
            let block_id = u32::try_from(candidate.block.0)
                .map_err(|_| "compiler fusion block exceeds portable identity".to_string())?;
            let slice = CandidateSlice {
                index: &index,
                block,
                original_candidate: candidate,
                layout: original.private_layout.as_ref(),
            };
            let mut start = candidate.stmt_start;
            let mut position = start;
            let mut pending_controls = Vec::new();
            while position < candidate.stmt_end {
                let point = ProgramPointId {
                    function: owner,
                    block: block_id,
                    position: u32::try_from(position).map_err(|_| {
                        "compiler fusion statement position exceeds portable identity".to_string()
                    })?,
                };
                let statement = index.statement_at(point).ok_or_else(|| {
                    "compiler fusion candidate has no selected statement".to_string()
                })?;
                let signals = index.signal_pcs(point).map_or(0, <[usize]>::len);
                let selected = matches!(
                    statement.contract.status,
                    FusionStageFactStatus::Inferred | FusionStageFactStatus::UnknownInputs
                );
                if selected
                    && signals == 1
                    && crate::bytecode::fusion_effects::elementwise_statement_effects_are_compatible(
                        &body.instructions,
                        statement,
                    )
                {
                    position += 1;
                    continue;
                }
                if selected && statement.contract.effects.0.is_empty() && signals == 0 {
                    let certified_bridge = statement
                        .contract
                        .defs
                        .as_slice()
                        .first()
                        .and_then(|definition| index.unique_user(*definition))
                        .filter(|terminal| {
                            terminal.block == block_id
                                && terminal.position > point.position
                                && (terminal.position as usize) < candidate.stmt_end
                        })
                        .and_then(|terminal| index.statement_at(terminal))
                        .is_some_and(|terminal| {
                            index.certified_empty_control(
                                statement,
                                terminal,
                                original.private_layout.as_ref(),
                            )
                        });
                    if certified_bridge {
                        pending_controls.push(position);
                        position += 1;
                        continue;
                    }
                    slice.flush_pure_parts(start, position, &pending_controls, true, &mut accepted);
                    pending_controls.clear();
                    start = position + 1;
                    position = start;
                    continue;
                }
                // Outside the shared canonical elementwise allowance, effects
                // can close a candidate only through its terminal reduction.
                let terminal_admitted = selected
                    && signals == 1
                    && slice.try_add_segment(start, position + 1, &mut accepted);
                if !terminal_admitted {
                    let tail = slice.flush_pure_parts(
                        start,
                        position,
                        &pending_controls,
                        false,
                        &mut accepted,
                    );
                    // A different terminal reduction may still certify the
                    // pure suffix after an earlier bridge was interrupted.
                    let tail_terminal_admitted = !pending_controls.is_empty()
                        && selected
                        && signals == 1
                        && tail < position
                        && slice.try_add_segment(tail, position + 1, &mut accepted);
                    if !tail_terminal_admitted {
                        slice.try_add_segment(tail, position, &mut accepted);
                    }
                }
                pending_controls.clear();
                start = position + 1;
                position = start;
            }
            if !slice.try_add_segment(start, candidate.stmt_end, &mut accepted)
                && !pending_controls.is_empty()
            {
                slice.flush_pure_parts(
                    start,
                    candidate.stmt_end,
                    &pending_controls,
                    true,
                    &mut accepted,
                );
            }
        }
        accepted
    };
    body.fusion_scaffold = assemble_scaffold(body, accepted, &original)?;
    body.validate_fusion_scaffold()
}

struct CandidateSlice<'slice, 'body> {
    index: &'slice FunctionFusionCertificateIndex<'body>,
    block: &'slice runmat_mir::BasicBlock,
    original_candidate: &'slice FusionCandidateGroup,
    layout: Option<&'slice FunctionPrivateSlotLayout>,
}

impl CandidateSlice<'_, '_> {
    fn flush_pure_parts(
        &self,
        start: usize,
        end: usize,
        controls: &[usize],
        include_tail: bool,
        accepted: &mut Vec<FusionCandidateGroup>,
    ) -> usize {
        let mut tail = start;
        for control in controls.iter().copied().filter(|control| *control < end) {
            self.try_add_segment(tail, control, accepted);
            tail = control + 1;
        }
        if include_tail {
            self.try_add_segment(tail, end, accepted);
        }
        tail
    }

    fn try_add_segment(
        &self,
        start: usize,
        end: usize,
        accepted: &mut Vec<FusionCandidateGroup>,
    ) -> bool {
        let Some(statements) = self.block.statements.get(start..end) else {
            return false;
        };
        let Ok(block_id) = u32::try_from(self.original_candidate.block.0) else {
            return false;
        };
        let signal_count = (start..end)
            .filter_map(|position| {
                self.index.signal_pcs(ProgramPointId {
                    function: self.index.owner(),
                    block: block_id,
                    position: u32::try_from(position).ok()?,
                })
            })
            .map(<[usize]>::len)
            .sum();
        if signal_count < 2 {
            return false;
        }
        let Some(first) = statements.first() else {
            return false;
        };
        let source_span = statements
            .iter()
            .skip(1)
            .fold(first.span, |span, statement| {
                runmat_hir::merge_span(span, statement.span)
            });
        let mut candidate = self.original_candidate.clone();
        candidate.stmt_start = start;
        candidate.stmt_end = end;
        candidate.signal_count = signal_count;
        candidate.source_span = source_span;
        let admitted = self
            .index
            .certify_candidate(&candidate, self.layout)
            .is_ok();
        if admitted {
            accepted.push(candidate);
        }
        admitted
    }
}

fn assemble_scaffold(
    body: &FunctionBytecode,
    mut candidates: Vec<FusionCandidateGroup>,
    original: &FunctionFusionScaffold,
) -> Result<Option<FunctionFusionScaffold>, String> {
    if candidates.is_empty() {
        return Ok(None);
    }
    for (id, candidate) in candidates.iter_mut().enumerate() {
        candidate.id = id;
    }
    let index = FunctionFusionCertificateIndex::new(body)?;
    let windows =
        selected_function_windows(body, &index, &candidates, original.private_layout.as_ref())?;
    Ok(Some(FunctionFusionScaffold {
        signal_count: candidates
            .iter()
            .map(|candidate| candidate.signal_count)
            .sum(),
        candidates,
        windows,
        private_layout: original.private_layout.clone(),
    }))
}

#[cfg(test)]
#[path = "function_fusion/tests.rs"]
mod tests;
