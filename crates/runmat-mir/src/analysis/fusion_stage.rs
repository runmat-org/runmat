//! Per-statement MIR evidence for a later staged fusion execution decision.
//!
//! These are analysis facts, not permission to reorder, omit, or execute a
//! statement on a provider. They use the owning body-flow facts and call
//! summaries, together with MIR region liveness.

use std::collections::{BTreeMap, BTreeSet};

use runmat_types::{
    CapabilitySet, CertaintyFact, EffectKind, EffectSet, ProgramFunctionId, ProgramPointId,
    RegionValueId, ValueKindFact,
};
use serde::{Deserialize, Serialize};

use super::inference::{statement_contract_with_facts, FunctionSummary};
use super::regions::{statement_uses_defs, BodyLiveness};
use super::{AssignmentFact, FunctionConvergence, ProgramPointFacts};
use crate::{MirBody, MirLocalId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionStageFactStatus {
    Inferred,
    UnknownInputs,
    UnknownEffects,
    MissingFacts,
    WidenedFacts,
    Nonconverged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionStageContract {
    pub point: ProgramPointId,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub uses: Vec<RegionValueId>,
    pub defs: Vec<RegionValueId>,
    pub live_after: Vec<RegionValueId>,
    pub status: FusionStageFactStatus,
}

pub(super) fn certify_body(
    body: &MirBody,
    function: ProgramFunctionId,
    points: &[ProgramPointFacts],
    summaries: &BTreeMap<runmat_hir::FunctionId, FunctionSummary>,
    liveness: &BodyLiveness,
    convergence: FunctionConvergence,
) -> Vec<FusionStageContract> {
    let facts_by_point: BTreeMap<_, _> = points.iter().map(|facts| (facts.point, facts)).collect();
    let mut contracts = Vec::new();
    for block in &body.blocks {
        let Ok(block_id) = u32::try_from(block.id.0) else {
            continue;
        };
        for (position, statement) in block.statements.iter().enumerate() {
            let Ok(position_id) = u32::try_from(position) else {
                continue;
            };
            let point = ProgramPointId {
                function,
                block: block_id,
                position: position_id,
            };
            let (uses, defs) = statement_uses_defs(statement);
            let live_after = liveness.after.get(&(block.id, position));
            let uses = typed_locals(function, &uses);
            let defs = typed_locals(function, &defs);
            let live_after = live_after.and_then(|locals| typed_locals(function, locals));
            let facts = facts_by_point.get(&point).copied();

            let (effects, capabilities, status) = match (facts, &uses, &defs, &live_after) {
                (Some(facts), Some(uses), Some(_), Some(_)) => {
                    let local_facts = facts.local_fact_vector(body.locals.len());
                    let (effects, capabilities) =
                        statement_contract_with_facts(statement, summaries, &local_facts);
                    let status = if effects.0.contains(&EffectKind::Unknown) {
                        FusionStageFactStatus::UnknownEffects
                    } else if uses.iter().any(|id| !proven_input(facts, *id)) {
                        FusionStageFactStatus::UnknownInputs
                    } else {
                        FusionStageFactStatus::Inferred
                    };
                    (effects, capabilities, status)
                }
                _ => (
                    EffectSet([EffectKind::Unknown].into_iter().collect()),
                    CapabilitySet::default(),
                    FusionStageFactStatus::MissingFacts,
                ),
            };
            let (effects, status) = match convergence {
                FunctionConvergence::Exact => (effects, status),
                FunctionConvergence::Widened | FunctionConvergence::DynamicRecursion => {
                    let mut effects = effects;
                    effects.0.insert(EffectKind::Unknown);
                    let status = if convergence == FunctionConvergence::Widened {
                        FusionStageFactStatus::WidenedFacts
                    } else {
                        FusionStageFactStatus::Nonconverged
                    };
                    (effects, status)
                }
            };
            contracts.push(FusionStageContract {
                point,
                effects,
                capabilities,
                uses: uses.unwrap_or_default(),
                defs: defs.unwrap_or_default(),
                live_after: live_after.unwrap_or_default(),
                status,
            });
        }
    }
    contracts
}

fn proven_input(facts: &ProgramPointFacts, value: RegionValueId) -> bool {
    facts.local(value).is_some_and(|local| {
        local.assignment == AssignmentFact::DefinitelyAssigned
            && local.fact.as_ref().is_some_and(|fact| {
                fact.certainty == CertaintyFact::Proven
                    && !matches!(fact.kind, ValueKindFact::Never | ValueKindFact::Unknown)
            })
    })
}

fn typed_locals(
    function: ProgramFunctionId,
    locals: &BTreeSet<MirLocalId>,
) -> Option<Vec<RegionValueId>> {
    locals
        .iter()
        .map(|local| {
            Some(RegionValueId {
                function,
                local: u32::try_from(local.0).ok()?,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "fusion_stage/tests.rs"]
mod tests;
