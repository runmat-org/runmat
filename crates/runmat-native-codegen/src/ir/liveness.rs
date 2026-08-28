use std::collections::{BTreeMap, BTreeSet};

use super::verify::{terminator_edges, terminator_values};
use super::{NativeAssembly, NativeBlockId, NativeEdgeArgument, NativeValueId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBlockLiveness {
    pub block: NativeBlockId,
    pub live_in: BTreeSet<NativeValueId>,
    pub live_out: BTreeSet<NativeValueId>,
}

pub fn analyze_liveness(
    assembly: &NativeAssembly,
) -> BTreeMap<runmat_types::ProgramFunctionId, Vec<NativeBlockLiveness>> {
    assembly
        .functions
        .iter()
        .map(|function| {
            let rows = function
                .blocks
                .iter()
                .map(|block| {
                    let definitions =
                        block
                            .parameters
                            .iter()
                            .map(|parameter| parameter.value)
                            .chain(block.instructions.iter().flat_map(|instruction| {
                                instruction.outputs.iter().map(|o| o.value)
                            }))
                            .chain(std::iter::once(block.side_effect_epoch))
                            .chain(
                                block
                                    .instructions
                                    .iter()
                                    .filter_map(|instruction| instruction.effect_epoch_output),
                            )
                            .collect::<BTreeSet<_>>();
                    let uses = block
                        .instructions
                        .iter()
                        .flat_map(|instruction| instruction.inputs.iter().copied())
                        .chain(block.instructions.iter().filter_map(|instruction| {
                            instruction
                                .frame_state
                                .as_ref()
                                .map(|state| state.side_effect_epoch)
                        }))
                        .chain(block.instructions.iter().flat_map(|instruction| {
                            instruction
                                .frame_state
                                .iter()
                                .flat_map(|state| state.locals.iter().map(|local| local.value))
                        }))
                        .chain(block.instructions.iter().flat_map(|instruction| {
                            instruction
                                .frame_state
                                .iter()
                                .flat_map(|state| state.operands.iter().copied())
                        }))
                        .chain(
                            block
                                .terminator
                                .frame_state
                                .locals
                                .iter()
                                .map(|local| local.value),
                        )
                        .chain(block.terminator.frame_state.operands.iter().copied())
                        .chain(std::iter::once(
                            block.terminator.frame_state.side_effect_epoch,
                        ))
                        .chain(block.region_boundaries.iter().flat_map(|boundary| {
                            boundary.live_values.iter().map(|binding| binding.ssa)
                        }))
                        .chain(block.region_boundaries.iter().flat_map(|boundary| {
                            boundary.guards.iter().filter_map(|guard| guard.value)
                        }))
                        .chain(block.region_boundaries.iter().flat_map(|boundary| {
                            boundary.frame_state.locals.iter().map(|local| local.value)
                        }))
                        .chain(
                            block
                                .region_boundaries
                                .iter()
                                .map(|boundary| boundary.frame_state.side_effect_epoch),
                        )
                        .chain(terminator_values(&block.terminator.kind))
                        .collect::<BTreeSet<_>>();
                    let live_in = uses.difference(&definitions).copied().collect();
                    let live_out = terminator_edges(&block.terminator.kind)
                        .into_iter()
                        .flat_map(|edge| edge.arguments.iter())
                        .filter_map(|argument| match argument {
                            NativeEdgeArgument::Value(value) => Some(value),
                            _ => None,
                        })
                        .copied()
                        .chain(
                            terminator_edges(&block.terminator.kind)
                                .into_iter()
                                .map(|edge| edge.side_effect_epoch),
                        )
                        .collect();
                    NativeBlockLiveness {
                        block: block.id,
                        live_in,
                        live_out,
                    }
                })
                .collect();
            (function.id, rows)
        })
        .collect()
}
