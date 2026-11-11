use super::*;
use crate::{
    NativeBlock, NativeEdge, NativeEdgeArgument, NativeFrameState, NativeFunction,
    NativeInstruction, NativeTerminatorKind,
};

fn invalid(message: &'static str) -> NativeCodegenError {
    NativeCodegenError::new("native.liveness.identity", message)
}
fn index(ids: &[NativeValueId], id: NativeValueId) -> NativeCodegenResult<usize> {
    ids.binary_search(&id)
        .map_err(|_| invalid("liveness use has no verified SSA definition"))
}
fn block_index(function: &NativeFunction, id: NativeBlockId) -> NativeCodegenResult<usize> {
    function
        .blocks
        .binary_search_by_key(&id, |block| block.id)
        .map_err(|_| invalid("liveness edge has no verified target"))
}
fn bit(words: &[u64], id: usize) -> bool {
    words[id / 64] & (1u64 << (id % 64)) != 0
}
fn set(words: &mut [u64], id: usize) {
    words[id / 64] |= 1u64 << (id % 64);
}
fn clear(words: &mut [u64], id: usize) {
    words[id / 64] &= !(1u64 << (id % 64));
}
fn push<T>(values: &mut Vec<T>, value: T) -> NativeCodegenResult<()> {
    if values.len() == values.capacity() {
        return Err(overflow());
    }
    values.push(value);
    Ok(())
}

fn frame_uses(
    frame: &NativeFrameState,
    visit: &mut impl FnMut(NativeValueId) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    for local in &frame.locals {
        visit(local.value)?;
    }
    for value in &frame.operands {
        visit(*value)?;
    }
    visit(frame.side_effect_epoch)
}

fn boundary_uses(
    block: &NativeBlock,
    position: u32,
    visit: &mut impl FnMut(NativeValueId) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    for boundary in &block.region_boundaries {
        if boundary.point.position != position {
            continue;
        }
        for value in &boundary.live_values {
            visit(value.ssa)?;
        }
        for value in boundary.guards.iter().filter_map(|guard| guard.value) {
            visit(value)?;
        }
        frame_uses(&boundary.frame_state, visit)?;
    }
    Ok(())
}

pub(super) fn instruction_uses(
    block: &NativeBlock,
    instruction: &NativeInstruction,
    mut visit: impl FnMut(NativeValueId) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    for value in &instruction.inputs {
        visit(*value)?;
    }
    if let Some(frame) = &instruction.frame_state {
        frame_uses(frame, &mut visit)?;
    }
    boundary_uses(block, instruction.site.point.position, &mut visit)
}

fn instruction_definitions(
    instruction: &NativeInstruction,
    mut visit: impl FnMut(NativeValueId) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    for output in &instruction.outputs {
        visit(output.value)?;
    }
    if let Some(value) = instruction.effect_epoch_output {
        visit(value)?;
    }
    Ok(())
}

fn edges(
    kind: &NativeTerminatorKind,
    mut visit: impl FnMut(&NativeEdge) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    match kind {
        NativeTerminatorKind::Goto { edge } => visit(edge),
        NativeTerminatorKind::Branch {
            then_edge,
            else_edge,
            ..
        }
        | NativeTerminatorKind::For {
            body: then_edge,
            exit: else_edge,
            ..
        }
        | NativeTerminatorKind::ParFor {
            body: then_edge,
            exit: else_edge,
            ..
        }
        | NativeTerminatorKind::Spmd {
            body: then_edge,
            exit: else_edge,
            ..
        }
        | NativeTerminatorKind::TryCatch {
            try_edge: then_edge,
            catch_edge: else_edge,
            ..
        } => {
            visit(then_edge)?;
            visit(else_edge)
        }
        NativeTerminatorKind::Switch {
            cases, otherwise, ..
        } => {
            for (_, edge) in cases {
                visit(edge)?;
            }
            visit(otherwise)
        }
        NativeTerminatorKind::Await { resume, .. } => visit(resume),
        NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable => Ok(()),
    }
}

fn edge_uses(
    edge: &NativeEdge,
    visit: &mut impl FnMut(NativeValueId) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    for argument in &edge.arguments {
        if let NativeEdgeArgument::Value(value) = argument {
            visit(*value)?;
        }
    }
    visit(edge.side_effect_epoch)
}

pub(super) fn terminator_uses(
    block: &NativeBlock,
    mut visit: impl FnMut(NativeValueId) -> NativeCodegenResult<()>,
) -> NativeCodegenResult<()> {
    frame_uses(&block.terminator.frame_state, &mut visit)?;
    boundary_uses(block, block.terminator.site.point.position, &mut visit)?;
    match &block.terminator.kind {
        NativeTerminatorKind::Branch { condition, .. } => visit(*condition)?,
        NativeTerminatorKind::Switch {
            discriminant,
            cases,
            ..
        } => {
            visit(*discriminant)?;
            for (value, _) in cases {
                visit(*value)?;
            }
        }
        NativeTerminatorKind::For { iterable, .. } => visit(*iterable)?,
        NativeTerminatorKind::ParFor {
            iterable,
            maximum_workers,
            ..
        } => {
            visit(*iterable)?;
            if let Some(value) = maximum_workers {
                visit(*value)?;
            }
        }
        NativeTerminatorKind::Spmd { header, .. }
        | NativeTerminatorKind::Return { values: header } => {
            for value in header {
                visit(*value)?;
            }
        }
        NativeTerminatorKind::Await { future, .. } => visit(*future)?,
        _ => {}
    }
    edges(&block.terminator.kind, |edge| edge_uses(edge, &mut visit))
}

fn reachable(
    function: &NativeFunction,
    start: NativeBlockId,
    words: &mut [u64],
) -> NativeCodegenResult<()> {
    words.fill(0);
    set(words, block_index(function, start)?);
    loop {
        let mut changed = false;
        for (source, block) in function.blocks.iter().enumerate() {
            if !bit(words, source) {
                continue;
            }
            edges(&block.terminator.kind, |edge| {
                let target = block_index(function, edge.target)?;
                if !bit(words, target) {
                    set(words, target);
                    changed = true;
                }
                Ok(())
            })?;
        }
        if !changed {
            return Ok(());
        }
    }
}

fn exception_edges(
    function: &NativeFunction,
    matrix: &mut [u64],
    reach: &mut [u64],
    graph_words: usize,
) -> NativeCodegenResult<()> {
    matrix.fill(0);
    let (try_reach, catch_reach) = reach.split_at_mut(graph_words);
    for (handler, block) in function.blocks.iter().enumerate() {
        if let NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } = &block.terminator.kind
        {
            reachable(function, try_edge.target, try_reach)?;
            reachable(function, catch_edge.target, catch_reach)?;
            // Match the validated native continuation's protected region,
            // including exceptional transfers from interior blocks.
            for source in 0..function.blocks.len() {
                if bit(try_reach, source) && !bit(catch_reach, source) {
                    set(
                        &mut matrix[source * graph_words..(source + 1) * graph_words],
                        handler,
                    );
                }
            }
        }
    }
    Ok(())
}

fn transfer_word(
    function: &NativeFunction,
    ids: &[NativeValueId],
    blocks: &[NativeBlockLiveness],
    bits: &[u64],
    edge: &NativeEdge,
    word: usize,
) -> NativeCodegenResult<u64> {
    let target = block_index(function, edge.target)?;
    let mut value = bits[blocks[target].entry + word];
    // Target block parameters are newly defined by the edge. Keep the actual
    // incoming argument identities, never a target parameter's old generation.
    for parameter in &function.blocks[target].parameters {
        let id = index(ids, parameter.value)?;
        if id / 64 == word {
            value &= !(1u64 << (id % 64));
        }
    }
    let id = index(ids, function.blocks[target].side_effect_epoch)?;
    if id / 64 == word {
        value &= !(1u64 << (id % 64));
    }
    edge_uses(edge, &mut |id| {
        let id = index(ids, id)?;
        if id / 64 == word {
            value |= 1u64 << (id % 64);
        }
        Ok(())
    })?;
    Ok(value)
}

/// Immutable CFG identities and exceptional transfers for the fixed point.
/// The evolving bitsets remain separate so each pass can update them in place.
struct FlowGraph<'a> {
    function: &'a NativeFunction,
    ids: &'a [NativeValueId],
    blocks: &'a [NativeBlockLiveness],
    exception_handlers: &'a [u64],
    graph_words: usize,
}

impl FlowGraph<'_> {
    fn out_word(&self, bits: &[u64], source: usize, word: usize) -> NativeCodegenResult<u64> {
        let Self {
            function,
            ids,
            blocks,
            exception_handlers: matrix,
            graph_words,
        } = *self;
        let mut out = 0;
        edges(&function.blocks[source].terminator.kind, |edge| {
            out |= transfer_word(function, ids, blocks, bits, edge, word)?;
            Ok(())
        })?;
        for handler in 0..function.blocks.len() {
            if !bit(
                &matrix[source * graph_words..(source + 1) * graph_words],
                handler,
            ) {
                continue;
            }
            if let NativeTerminatorKind::TryCatch { catch_edge, .. } =
                &function.blocks[handler].terminator.kind
            {
                out |= transfer_word(function, ids, blocks, bits, catch_edge, word)?;
            }
        }
        Ok(out)
    }
}

pub(super) fn function(
    function: &NativeFunction,
    data: &mut NativeProgramLiveness,
    workspace: &mut [u64],
) -> NativeCodegenResult<()> {
    let first_value = data.values.len();
    let first_block = data.blocks.len();
    let first_site = data.sites.len();
    for block in &function.blocks {
        push(&mut data.values, block.side_effect_epoch)?;
        for parameter in &block.parameters {
            push(&mut data.values, parameter.value)?;
        }
        for instruction in &block.instructions {
            instruction_definitions(instruction, |value| push(&mut data.values, value))?;
        }
    }
    data.values[first_value..].sort_unstable();
    // Verification guarantees unique definitions; retain the exact plan's
    // capacity and refuse an unexpected duplicate instead of inventing a map.
    let ids = &data.values[first_value..];
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("duplicate liveness definition"));
    }
    let values = ids.len();
    let words = add(values, 63)? / 64;
    let count = function.blocks.len();
    let block_words = mul(count, words)?;
    let graph_words = add(count, 63)? / 64;
    let matrix_words = mul(count, graph_words)?;
    let scratch_words = add(
        add(mul(block_words, 2)?, mul(words, 2)?)?,
        add(matrix_words, mul(graph_words, 2)?)?,
    )?;
    let scratch = &mut workspace[..scratch_words];
    scratch.fill(0);
    let (uses_defs, remainder) = scratch.split_at_mut(2 * block_words);
    let (uses, definitions) = uses_defs.split_at_mut(block_words);
    let (reverse, remainder) = remainder.split_at_mut(2 * words);
    let (matrix, reach) = remainder.split_at_mut(matrix_words);
    exception_edges(function, matrix, reach, graph_words)?;
    let bits_start = data.bits.len();
    let bits_end = add(bits_start, mul(block_words, 2)?)?;
    if bits_end > data.bits.capacity() {
        return Err(overflow());
    }
    data.bits.resize(bits_end, 0);
    for (number, block) in function.blocks.iter().enumerate() {
        let entry = bits_start + number * words * 2;
        push(
            &mut data.blocks,
            NativeBlockLiveness {
                block: block.id,
                entry,
                exit: entry + words,
            },
        )?;
        let block_use = &mut uses[number * words..(number + 1) * words];
        let block_def = &mut definitions[number * words..(number + 1) * words];
        for instruction in &block.instructions {
            instruction_uses(block, instruction, |value| {
                let value = index(ids, value)?;
                if !bit(block_def, value) {
                    set(block_use, value);
                }
                Ok(())
            })?;
            instruction_definitions(instruction, |value| {
                set(block_def, index(ids, value)?);
                Ok(())
            })?;
            push(
                &mut data.sites,
                NativeRetirementSite {
                    site: instruction.site.clone(),
                    first_retired: 0,
                    retired: 0,
                },
            )?;
        }
        terminator_uses(block, |value| {
            let value = index(ids, value)?;
            if !bit(block_def, value) {
                set(block_use, value);
            }
            Ok(())
        })?;
        push(
            &mut data.sites,
            NativeRetirementSite {
                site: block.terminator.site.clone(),
                first_retired: 0,
                retired: 0,
            },
        )?;
    }
    let blocks = &data.blocks[first_block..];
    let graph = FlowGraph {
        function,
        ids,
        blocks,
        exception_handlers: matrix,
        graph_words,
    };
    // Monotone CFG fixed point. Edge transfer substitutes block parameters;
    // loop/backedge and diamond uses cannot disappear through ordering.
    loop {
        data.admission.check_current()?;
        let mut changed = false;
        for source in (0..count).rev() {
            for word in 0..words {
                let out = graph.out_word(&data.bits, source, word)?;
                let entry =
                    uses[source * words + word] | (out & !definitions[source * words + word]);
                let row = blocks[source];
                changed |=
                    data.bits[row.entry + word] != entry || data.bits[row.exit + word] != out;
                data.bits[row.entry + word] = entry;
                data.bits[row.exit + word] = out;
            }
        }
        if !changed {
            break;
        }
    }
    let (live, retired_mask) = reverse.split_at_mut(words);
    let mut site = first_site;
    for (source, block) in function.blocks.iter().enumerate() {
        let row = blocks[source];
        live.copy_from_slice(&data.bits[row.exit..row.exit + words]);
        let terminator_site = site + block.instructions.len();
        retired_mask.fill(0);
        terminator_uses(block, |value| use_before(ids, live, retired_mask, value))?;
        store_retired(
            ids,
            retired_mask,
            &mut data.retired,
            &mut data.sites[terminator_site],
        )?;
        for (offset, instruction) in block.instructions.iter().enumerate().rev() {
            retired_mask.fill(0);
            instruction_definitions(instruction, |value| {
                let value = index(ids, value)?;
                if !bit(live, value) {
                    set(retired_mask, value);
                }
                clear(live, value);
                Ok(())
            })?;
            instruction_uses(block, instruction, |value| {
                use_before(ids, live, retired_mask, value)
            })?;
            store_retired(
                ids,
                retired_mask,
                &mut data.retired,
                &mut data.sites[site + offset],
            )?;
        }
        site = terminator_site + 1;
    }
    data.sites[first_site..].sort_unstable_by(|left, right| left.site.cmp(&right.site));
    push(
        &mut data.functions,
        NativeFunctionLiveness {
            function: function.id,
            first_value,
            values,
            first_block,
            blocks: count,
            first_site,
            sites: site - first_site,
            words,
        },
    )
}

fn use_before(
    ids: &[NativeValueId],
    live: &mut [u64],
    retired: &mut [u64],
    value: NativeValueId,
) -> NativeCodegenResult<()> {
    let value = index(ids, value)?;
    if !bit(live, value) {
        set(retired, value);
    }
    set(live, value);
    Ok(())
}

fn store_retired(
    ids: &[NativeValueId],
    mask: &[u64],
    values: &mut Vec<NativeValueId>,
    site: &mut NativeRetirementSite,
) -> NativeCodegenResult<()> {
    site.first_retired = values.len();
    for (word, value) in mask.iter().enumerate() {
        let mut remaining = *value;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let index = word * 64 + bit;
            if let Some(value) = ids.get(index) {
                push(values, *value)?;
            }
            remaining &= remaining - 1;
        }
    }
    site.retired = values.len() - site.first_retired;
    Ok(())
}
