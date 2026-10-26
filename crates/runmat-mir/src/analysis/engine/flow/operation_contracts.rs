use std::collections::{BTreeMap, BTreeSet};

use runmat_hir::FunctionId;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, EffectKind, EffectSet, ProgramFunctionId, ProgramPointId,
};

use crate::analysis::inference::{
    conservative_rvalue_contract, rvalue_operation_contract, statement_phase_contract,
    FunctionSummary,
};
use crate::{BasicBlock, MirBody, MirStmt, MirTerminatorKind};
use crate::{MirOperationContract, MirOperationPhase, MirOperationSite};

use super::super::state::FlowState;

pub(super) fn complete_missing(
    body: &MirBody,
    function: ProgramFunctionId,
    contracts: &mut Vec<MirOperationContract>,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) {
    let Some(expected) = crate::operation_sites(function, body) else {
        return;
    };
    let present = contracts
        .iter()
        .map(|contract| contract.site.clone())
        .collect::<BTreeSet<_>>();
    let blocks = body
        .blocks
        .iter()
        .filter_map(|block| Some((u32::try_from(block.id.0).ok()?, block)))
        .collect::<BTreeMap<_, _>>();
    for site in expected {
        if present.contains(&site) {
            continue;
        }
        let block = blocks.get(&site.point.block).copied();
        let (effects, capabilities) = match (site.phase, block) {
            (MirOperationPhase::Rvalue, Some(block)) => block
                .statements
                .get(site.point.position as usize)
                .and_then(|statement| crate::statement_rvalue(&statement.kind))
                .map(|value| conservative_rvalue_contract(value, summaries))
                .unwrap_or_default(),
            (MirOperationPhase::Statement, Some(block)) => block
                .statements
                .get(site.point.position as usize)
                .map(|statement| statement_phase_contract(statement, summaries, &[]))
                .unwrap_or_default(),
            (MirOperationPhase::TerminatorRvalue, Some(block)) => {
                crate::terminator_rvalues(&block.terminator.kind)
                    .get(site.ordinal as usize)
                    .map(|value| conservative_rvalue_contract(value, summaries))
                    .unwrap_or_default()
            }
            (MirOperationPhase::Terminator, Some(block)) => terminator(&block.terminator.kind),
            _ => Default::default(),
        };
        contracts.push(MirOperationContract {
            site,
            effects,
            capabilities,
        });
    }
}

fn point_for(
    function: ProgramFunctionId,
    block: &BasicBlock,
    position: usize,
) -> Option<ProgramPointId> {
    Some(ProgramPointId {
        function,
        block: u32::try_from(block.id.0).ok()?,
        position: u32::try_from(position).ok()?,
    })
}

pub(super) fn record_statement(
    contracts: &mut Vec<MirOperationContract>,
    function: ProgramFunctionId,
    block: &BasicBlock,
    position: usize,
    statement: &MirStmt,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) {
    let Some(point) = point_for(function, block, position) else {
        return;
    };
    if let Some(value) = crate::statement_rvalue(&statement.kind) {
        let (effects, capabilities) = rvalue_operation_contract(value, state, summaries);
        contracts.push(MirOperationContract {
            site: MirOperationSite {
                point,
                phase: MirOperationPhase::Rvalue,
                ordinal: 0,
                construct: crate::rvalue_construct_kind(value),
            },
            effects,
            capabilities,
        });
    }
    let facts = state.value_facts();
    let (effects, capabilities) = statement_phase_contract(statement, summaries, &facts);
    contracts.push(MirOperationContract {
        site: MirOperationSite {
            point,
            phase: MirOperationPhase::Statement,
            ordinal: 0,
            construct: crate::statement_construct_kind(&statement.kind),
        },
        effects,
        capabilities,
    });
}

pub(super) fn record_terminator(
    contracts: &mut Vec<MirOperationContract>,
    function: ProgramFunctionId,
    block: &BasicBlock,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) {
    let Some(point) = point_for(function, block, block.statements.len()) else {
        return;
    };
    for (ordinal, value) in crate::terminator_rvalues(&block.terminator.kind)
        .into_iter()
        .enumerate()
    {
        let Ok(ordinal) = u32::try_from(ordinal) else {
            return;
        };
        let (effects, capabilities) = rvalue_operation_contract(&value, state, summaries);
        contracts.push(MirOperationContract {
            site: MirOperationSite {
                point,
                phase: MirOperationPhase::TerminatorRvalue,
                ordinal,
                construct: crate::rvalue_construct_kind(&value),
            },
            effects,
            capabilities,
        });
    }
    let (effects, capabilities) = terminator(&block.terminator.kind);
    contracts.push(MirOperationContract {
        site: MirOperationSite {
            point,
            phase: MirOperationPhase::Terminator,
            ordinal: 0,
            construct: crate::terminator_construct_kind(&block.terminator.kind),
        },
        effects,
        capabilities,
    });
}

pub(super) fn terminator(terminator: &MirTerminatorKind) -> (EffectSet, CapabilitySet) {
    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    match terminator {
        MirTerminatorKind::Await { .. } => {
            effects.0.insert(EffectKind::MaySuspend);
            capabilities
                .0
                .insert(CapabilityRequirement::ParallelRuntime);
        }
        MirTerminatorKind::ParFor { .. } | MirTerminatorKind::Spmd { .. } => {
            capabilities
                .0
                .insert(CapabilityRequirement::ParallelRuntime);
        }
        _ => {}
    }
    (effects, capabilities)
}
