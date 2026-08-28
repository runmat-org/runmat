use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_builtins::BuiltinPurity;
use runmat_hir::FunctionId;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, EffectSet, ProgramFunctionId, ProgramPointId,
    ProgramSpan, RegionContract, RegionId, RegionProvenance, RegionValueFact, RegionValueId,
    REGION_CONTRACT_SCHEMA_VERSION,
};

use crate::{BasicBlockId, MirBody, MirOutputTarget, MirPlace, MirRvalue, MirStmt, MirStmtKind};

use super::liveness::{analyze_liveness, BodyLiveness};
use super::uses::{consumer_sites, successors};
use crate::analysis::inference::{statement_contract, FunctionSummary};
use crate::analysis::{AnalysisStore, RegionAnalysis, RegionFutureConsumer};

#[derive(Debug, Clone)]
struct Run {
    operations: Vec<(BasicBlockId, usize)>,
}

pub(crate) fn discover_regions(
    body: &MirBody,
    function: ProgramFunctionId,
    source: runmat_types::ProgramSourceId,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    store: &AnalysisStore,
) -> Vec<RegionAnalysis> {
    let liveness = analyze_liveness(body);
    let mut runs = local_runs(body, summaries);
    merge_linear_runs(body, &mut runs);
    runs.sort_by_key(|run| run.operations.first().copied());
    let consumers = consumer_sites(body, function);

    runs.into_iter()
        .enumerate()
        .filter_map(|(ordinal, run)| {
            build_region(
                body,
                function,
                source,
                u32::try_from(ordinal).ok()?,
                run,
                summaries,
                store,
                &liveness,
                &consumers,
            )
        })
        .collect()
}

fn local_runs(body: &MirBody, summaries: &BTreeMap<FunctionId, FunctionSummary>) -> Vec<Run> {
    let mut runs = Vec::new();
    for block in &body.blocks {
        let mut current = Vec::new();
        for (position, statement) in block.statements.iter().enumerate() {
            if statement_is_pure(statement, summaries) {
                current.push((block.id, position));
            } else if !current.is_empty() {
                runs.push(Run {
                    operations: std::mem::take(&mut current),
                });
            }
        }
        if !current.is_empty() {
            runs.push(Run {
                operations: current,
            });
        }
    }
    runs
}

fn merge_linear_runs(body: &MirBody, runs: &mut Vec<Run>) {
    if runs.len() < 2 {
        return;
    }
    let block_index = body
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.id, index))
        .collect::<BTreeMap<_, _>>();
    let mut predecessors: BTreeMap<BasicBlockId, Vec<BasicBlockId>> = BTreeMap::new();
    for block in &body.blocks {
        for successor in successors(&block.terminator.kind) {
            predecessors.entry(successor).or_default().push(block.id);
        }
    }
    for values in predecessors.values_mut() {
        values.sort();
        values.dedup();
    }
    let mut first_run = BTreeMap::new();
    let mut last_run = BTreeMap::new();
    for (index, run) in runs.iter().enumerate() {
        let Some((first_block, first_position)) = run.operations.first().copied() else {
            continue;
        };
        let Some((last_block, last_position)) = run.operations.last().copied() else {
            continue;
        };
        if first_position == 0 {
            first_run.insert(first_block, index);
        }
        if block_index
            .get(&last_block)
            .and_then(|index| body.blocks.get(*index))
            .is_some_and(|block| last_position + 1 == block.statements.len())
        {
            last_run.insert(last_block, index);
        }
    }

    let mut next = BTreeMap::new();
    let mut previous = BTreeMap::new();
    for block in &body.blocks {
        let crate::MirTerminatorKind::Goto(successor) = block.terminator.kind else {
            continue;
        };
        if predecessors.get(&successor).map(Vec::as_slice) != Some(&[block.id])
            || path_exists(body, successor, block.id)
        {
            continue;
        }
        let (Some(&from), Some(&to)) = (last_run.get(&block.id), first_run.get(&successor)) else {
            continue;
        };
        if from != to && !next.contains_key(&from) && !previous.contains_key(&to) {
            next.insert(from, to);
            previous.insert(to, from);
        }
    }

    let mut merged = Vec::new();
    let mut consumed = BTreeSet::new();
    for root in 0..runs.len() {
        if previous.contains_key(&root) || consumed.contains(&root) {
            continue;
        }
        let mut operations = Vec::new();
        let mut current = root;
        loop {
            if !consumed.insert(current) {
                break;
            }
            operations.extend(runs[current].operations.iter().copied());
            let Some(&successor) = next.get(&current) else {
                break;
            };
            current = successor;
        }
        merged.push(Run { operations });
    }
    for (index, run) in runs.iter().enumerate() {
        if consumed.insert(index) {
            merged.push(run.clone());
        }
    }
    *runs = merged;
}

#[allow(clippy::too_many_arguments)]
fn build_region(
    body: &MirBody,
    function: ProgramFunctionId,
    source: runmat_types::ProgramSourceId,
    ordinal: u32,
    run: Run,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    store: &AnalysisStore,
    liveness: &BodyLiveness,
    consumers: &[(ProgramPointId, BTreeSet<crate::MirLocalId>)],
) -> Option<RegionAnalysis> {
    let (first_block, first_position) = *run.operations.first()?;
    let (last_block, last_position) = *run.operations.last()?;
    let entry = point(function, first_block, first_position)?;
    let exit = point(function, last_block, last_position.checked_add(1)?)?;
    let id = RegionId { function, ordinal };
    let live_in = liveness.before.get(&(first_block, first_position))?;
    let live_out = liveness.after.get(&(last_block, last_position))?;
    let live_in = region_values(function, live_in);
    let live_out = region_values(function, live_out);

    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    let mut span_start = usize::MAX;
    let mut span_end = 0usize;
    let mut operations = Vec::with_capacity(run.operations.len());
    for (block, position) in &run.operations {
        let statement = statement(body, *block, *position)?;
        let (statement_effects, statement_capabilities) = statement_contract(statement, summaries);
        effects.0.extend(statement_effects.0);
        capabilities.0.extend(statement_capabilities.0);
        span_start = span_start.min(statement.span.start);
        span_end = span_end.max(statement.span.end);
        operations.push(point(function, *block, *position)?);
    }

    let mut facts = BTreeMap::new();
    if let Some(point) = store.facts_at(entry) {
        for value in &live_in {
            if let Some(fact) = point.local(*value).and_then(|local| local.fact.clone()) {
                facts.insert(*value, fact);
            }
        }
    }
    if let Some(point) = store.facts_at(exit) {
        for value in &live_out {
            if let Some(fact) = point.local(*value).and_then(|local| local.fact.clone()) {
                facts.insert(*value, fact);
            }
        }
    }
    let value_facts = facts
        .into_iter()
        .map(|(value, fact)| RegionValueFact { value, fact })
        .collect();
    let future_consumers = future_consumers(body, &run, function, live_out.as_slice(), consumers);
    let contract = RegionContract {
        schema_version: REGION_CONTRACT_SCHEMA_VERSION,
        id,
        source,
        span: ProgramSpan {
            start: u64::try_from(span_start).ok()?,
            end: u64::try_from(span_end).ok()?,
        },
        entry,
        exits: vec![exit],
        live_in,
        live_out,
        value_facts,
        effects,
        capabilities,
        guards: Vec::new(),
        provenance: RegionProvenance::Inferred,
    };
    contract.validate().ok()?;
    Some(RegionAnalysis {
        contract,
        operations,
        future_consumers,
    })
}

fn statement_is_pure(
    statement: &MirStmt,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> bool {
    let local_targets = match &statement.kind {
        MirStmtKind::Assign { place, .. } => matches!(place, MirPlace::Local(_)),
        MirStmtKind::MultiAssign { targets, .. } => targets.targets.iter().all(|target| {
            matches!(
                target,
                MirOutputTarget::Discard | MirOutputTarget::Place(MirPlace::Local(_))
            )
        }),
        MirStmtKind::Expr(_) => true,
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => false,
    };
    if !local_targets || !statement_calls_are_pure(statement) {
        return false;
    }
    let (effects, capabilities) = statement_contract(statement, summaries);
    effects.0.is_empty()
        && !capabilities
            .0
            .contains(&CapabilityRequirement::ParallelRuntime)
        && !capabilities
            .0
            .contains(&CapabilityRequirement::DistributedRuntime)
}

fn statement_calls_are_pure(statement: &MirStmt) -> bool {
    match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::Expr(value) => rvalue_calls_are_pure(value),
        _ => false,
    }
}

fn rvalue_calls_are_pure(value: &MirRvalue) -> bool {
    match value {
        MirRvalue::Call(call) => call.purity == BuiltinPurity::Pure,
        MirRvalue::ShortCircuit { right_temps, .. } => {
            right_temps.iter().all(statement_calls_are_pure)
        }
        MirRvalue::Future { .. }
        | MirRvalue::Spawn(_)
        | MirRvalue::Distributed(_)
        | MirRvalue::Collective(_) => false,
        _ => true,
    }
}

fn future_consumers(
    body: &MirBody,
    run: &Run,
    function: ProgramFunctionId,
    live_out: &[RegionValueId],
    consumers: &[(ProgramPointId, BTreeSet<crate::MirLocalId>)],
) -> Vec<RegionFutureConsumer> {
    let Some((last_block, last_position)) = run.operations.last().copied() else {
        return Vec::new();
    };
    let reachable = reachable_blocks(body, last_block);
    let loops_to_exit_block = successors_for_block(body, last_block)
        .into_iter()
        .any(|successor| path_exists(body, successor, last_block));
    let operation_points = run
        .operations
        .iter()
        .filter_map(|(block, position)| point(function, *block, *position))
        .collect::<BTreeSet<_>>();
    let locals = live_out
        .iter()
        .filter_map(|value| usize::try_from(value.local).ok().map(crate::MirLocalId))
        .collect::<BTreeSet<_>>();
    let mut result = Vec::new();
    for (consumer, used) in consumers {
        let block = BasicBlockId(consumer.block as usize);
        if !reachable.contains(&block) || operation_points.contains(consumer) {
            continue;
        }
        if block == last_block
            && consumer.position <= u32::try_from(last_position).unwrap_or(u32::MAX)
            && !loops_to_exit_block
        {
            continue;
        }
        for local in used.intersection(&locals) {
            if let Ok(local) = u32::try_from(local.0) {
                result.push(RegionFutureConsumer {
                    value: RegionValueId { function, local },
                    point: *consumer,
                });
            }
        }
    }
    result.sort();
    result.dedup();
    result
}

fn region_values(
    function: ProgramFunctionId,
    locals: &BTreeSet<crate::MirLocalId>,
) -> Vec<RegionValueId> {
    locals
        .iter()
        .filter_map(|local| {
            u32::try_from(local.0)
                .ok()
                .map(|local| RegionValueId { function, local })
        })
        .collect()
}

fn statement(body: &MirBody, block: BasicBlockId, position: usize) -> Option<&MirStmt> {
    body.blocks
        .iter()
        .find(|candidate| candidate.id == block)?
        .statements
        .get(position)
}

fn point(
    function: ProgramFunctionId,
    block: BasicBlockId,
    position: usize,
) -> Option<ProgramPointId> {
    Some(ProgramPointId {
        function,
        block: u32::try_from(block.0).ok()?,
        position: u32::try_from(position).ok()?,
    })
}

fn path_exists(body: &MirBody, from: BasicBlockId, target: BasicBlockId) -> bool {
    if from == target {
        return true;
    }
    let mut queue = VecDeque::from([from]);
    let mut visited = BTreeSet::new();
    while let Some(block) = queue.pop_front() {
        if !visited.insert(block) {
            continue;
        }
        for successor in successors_for_block(body, block) {
            if successor == target {
                return true;
            }
            queue.push_back(successor);
        }
    }
    false
}

fn reachable_blocks(body: &MirBody, from: BasicBlockId) -> BTreeSet<BasicBlockId> {
    let mut queue = VecDeque::from([from]);
    let mut visited = BTreeSet::new();
    while let Some(block) = queue.pop_front() {
        if visited.insert(block) {
            queue.extend(successors_for_block(body, block));
        }
    }
    visited
}

fn successors_for_block(body: &MirBody, block: BasicBlockId) -> Vec<BasicBlockId> {
    body.blocks
        .iter()
        .find(|candidate| candidate.id == block)
        .map_or_else(Vec::new, |block| successors(&block.terminator.kind))
}
