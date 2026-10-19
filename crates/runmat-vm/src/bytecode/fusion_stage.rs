//! Selected MIR stage facts bound once to a function-local instruction stream.
//! This is statement evidence, not provider or numerical execution admission.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_mir::analysis::{
    AnalysisRevision, AnalysisStore, FusionStageContract, FusionStageFactStatus,
};
use runmat_mir::{CatchScopeId, MirBody, MirLocalKind};
use runmat_types::{CapabilitySet, EffectSet, ProgramFunctionId, ProgramPointId, RegionValueId};
use serde::{Deserialize, Serialize};

use super::instr::Instr;
use super::program::fusion_stage_source_fingerprint;
use super::statement_pc::{project_statement_pcs, StatementPcRange};
use super::BytecodeSourceRef;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledFusionStageStatement {
    pub contract: FusionStageContract,
    pub pc: StatementPcRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompiledPrivateTemporarySlot {
    pub local: u32,
    pub slot: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionStageProjection {
    pub revision: AnalysisRevision,
    /// Immutable analyzed program-body identity, never a session function ID.
    pub portable_origin: ProgramFunctionId,
    /// Compiler-selected executable carrier, sealed with the source stream.
    /// Legacy projections without this field cannot authorize image stores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) selection_origin: Option<FusionStageSelectionOrigin>,
    pub owner: ProgramFunctionId,
    pub mir_local_count: usize,
    pub statements: Vec<CompiledFusionStageStatement>,
    /// Compiler-derived MIR Temporary roles joined to their exact VM slots.
    /// Missing in legacy artifacts means no private bridge authority.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) private_temporary_slots: Vec<CompiledPrivateTemporarySlot>,
    /// The compiler's exact function-local slot/layout owner. Entrypoint
    /// streams retain their existing assembly layout instead. This is not a
    /// provider or arithmetic certificate and cannot be supplied by a caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) selected_function_layout: Option<crate::layout::VmFunctionLayout>,
    /// Binds the selected facts to the exact serialized executable stream.
    pub source_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FusionStageSelectionOrigin {
    EntrypointStream,
    FunctionBody,
}

/// A retained definition's local statement facts without transport/session
/// owner ordinals. Source ownership is supplied by the enclosing definition.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub(crate) struct FusionStageDefinitionIdentity<'a> {
    revision: &'a AnalysisRevision,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection_origin: Option<FusionStageSelectionOrigin>,
    mir_local_count: usize,
    statements: Vec<FusionStageStatementIdentity<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    private_temporary_slots: &'a Vec<CompiledPrivateTemporarySlot>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct FusionStageStatementIdentity<'a> {
    block: u32,
    position: u32,
    pc_start: usize,
    pc_end: usize,
    effects: &'a EffectSet,
    capabilities: &'a CapabilitySet,
    uses: Vec<u32>,
    defs: Vec<u32>,
    live_after: Vec<u32>,
    status: FusionStageFactStatus,
}

pub(crate) struct FusionStageStream<'a> {
    pub instructions: &'a [Instr],
    pub catch_scopes: &'a [Option<CatchScopeId>],
    pub resume_points: &'a BTreeMap<ProgramPointId, usize>,
    pub source_id: Option<BytecodeSourceRef>,
    pub var_count: usize,
}

/// Canonical stage-index ordering checked once before projecting any selected
/// body, so a caller cannot claim that an arbitrary AnalysisStore is sorted.
#[derive(Clone, Copy)]
pub(crate) struct OrderedFusionStageAnalysis<'a> {
    analysis: &'a AnalysisStore,
}

impl<'a> OrderedFusionStageAnalysis<'a> {
    pub(crate) fn new(analysis: &'a AnalysisStore) -> Option<Self> {
        analysis.revision.validate_current().ok()?;
        analysis
            .fusion_stage_contracts
            .windows(2)
            .all(|pair| pair[0].point < pair[1].point)
            .then_some(Self { analysis })
    }
}

impl FusionStageProjection {
    pub(crate) fn project(
        body: &MirBody,
        selected: OrderedFusionStageAnalysis<'_>,
        selection_origin: FusionStageSelectionOrigin,
        layout: &crate::layout::VmFunctionLayout,
        stream: FusionStageStream<'_>,
    ) -> Option<Self> {
        let analysis = selected.analysis;
        if layout.function != body.function {
            return None;
        }
        let ranges = project_statement_pcs(stream.resume_points, stream.instructions.len()).ok()?;
        let owner = ranges.first()?.statement.function;
        if stream.catch_scopes.len() != stream.instructions.len() {
            return None;
        }
        let mut expected = BTreeSet::new();
        for block in &body.blocks {
            let block_id = u32::try_from(block.id.0).ok()?;
            for position in 0..block.statements.len() {
                expected.insert(ProgramPointId {
                    function: owner,
                    block: block_id,
                    position: u32::try_from(position).ok()?,
                });
            }
        }
        if expected.len() != ranges.len()
            || !ranges
                .iter()
                .map(|range| range.statement)
                .eq(expected.iter().copied())
        {
            return None;
        }
        let statements = ranges
            .into_iter()
            .map(|pc| {
                let contract = analysis.fusion_stage_at(pc.statement)?.clone();
                Some(CompiledFusionStageStatement { contract, pc })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut private_temporary_slots = body
            .locals
            .iter()
            .filter(|local| matches!(&local.kind, MirLocalKind::Temporary))
            .map(|local| {
                if local.binding.is_some() {
                    return None;
                }
                let slot = layout.mir_local_slots.get(&local.id)?;
                if layout.binding_slots.values().any(|bound| bound == slot) {
                    return None;
                }
                Some(CompiledPrivateTemporarySlot {
                    local: u32::try_from(local.id.0).ok()?,
                    slot: slot.0,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        private_temporary_slots.sort_unstable_by_key(|pair| pair.local);
        if private_temporary_slots
            .windows(2)
            .any(|pair| pair[0].local == pair[1].local)
            || private_temporary_slots
                .iter()
                .any(|pair| pair.slot >= stream.var_count)
            || private_temporary_slots
                .iter()
                .enumerate()
                .any(|(index, pair)| {
                    private_temporary_slots[index + 1..]
                        .iter()
                        .any(|later| later.slot == pair.slot)
                })
        {
            return None;
        }
        let mut result = Self {
            revision: analysis.revision.clone(),
            portable_origin: owner,
            selection_origin: Some(selection_origin),
            owner,
            mir_local_count: body.locals.len(),
            statements,
            private_temporary_slots,
            selected_function_layout: matches!(
                selection_origin,
                FusionStageSelectionOrigin::FunctionBody
            )
            .then(|| layout.clone()),
            source_fingerprint: String::new(),
        };
        if !result.structurally_valid(&stream) {
            return None;
        }
        result.source_fingerprint = result.fingerprint(&stream).ok()?;
        Some(result)
    }

    pub(crate) fn validate(&self, stream: FusionStageStream<'_>) -> bool {
        self.revision.validate_current().is_ok()
            && self.structurally_valid(&stream)
            && self
                .fingerprint(&stream)
                .is_ok_and(|fingerprint| fingerprint == self.source_fingerprint)
    }

    pub(crate) fn rebind_owner(
        &mut self,
        owner: ProgramFunctionId,
        stream: FusionStageStream<'_>,
    ) -> bool {
        self.rebind_owner_unsealed(owner);
        self.reseal_owned_remap(stream)
    }

    pub(crate) fn rebind_owner_unsealed(&mut self, owner: ProgramFunctionId) {
        self.owner = owner;
        for statement in &mut self.statements {
            statement.contract.point.function = owner;
            statement.pc.statement.function = owner;
            for value in statement
                .contract
                .uses
                .iter_mut()
                .chain(&mut statement.contract.defs)
                .chain(&mut statement.contract.live_after)
            {
                value.function = owner;
            }
        }
    }

    pub(crate) fn reseal_owned_remap(&mut self, stream: FusionStageStream<'_>) -> bool {
        if !self.structurally_valid(&stream) {
            return false;
        }
        let Ok(fingerprint) = self.fingerprint(&stream) else {
            return false;
        };
        self.source_fingerprint = fingerprint;
        true
    }

    /// Definition identity retains analyzed facts without compiler-unit or
    /// session-local ordinals, and omits the stream digest (whose source
    /// ordinal is envelope-local).
    /// Runtime and artifact admission still require `validate` on the exact
    /// executable stream before these facts can be consumed.
    pub(crate) fn portable_definition_identity(&self) -> FusionStageDefinitionIdentity<'_> {
        FusionStageDefinitionIdentity {
            revision: &self.revision,
            selection_origin: self.selection_origin,
            mir_local_count: self.mir_local_count,
            private_temporary_slots: &self.private_temporary_slots,
            statements: self
                .statements
                .iter()
                .map(|statement| FusionStageStatementIdentity {
                    block: statement.contract.point.block,
                    position: statement.contract.point.position,
                    pc_start: statement.pc.start,
                    pc_end: statement.pc.end,
                    effects: &statement.contract.effects,
                    capabilities: &statement.contract.capabilities,
                    uses: statement
                        .contract
                        .uses
                        .iter()
                        .map(|value| value.local)
                        .collect(),
                    defs: statement
                        .contract
                        .defs
                        .iter()
                        .map(|value| value.local)
                        .collect(),
                    live_after: statement
                        .contract
                        .live_after
                        .iter()
                        .map(|value| value.local)
                        .collect(),
                    status: statement.contract.status,
                })
                .collect(),
        }
    }

    pub(crate) fn statement_at(
        &self,
        point: ProgramPointId,
    ) -> Option<&CompiledFusionStageStatement> {
        self.statements
            .binary_search_by_key(&point, |row| row.contract.point)
            .ok()
            .map(|index| &self.statements[index])
    }

    fn structurally_valid(&self, stream: &FusionStageStream<'_>) -> bool {
        if self.statements.is_empty() || stream.catch_scopes.len() != stream.instructions.len() {
            return false;
        }
        if self
            .selected_function_layout
            .as_ref()
            .is_some_and(|layout| {
                layout.resume_points != *stream.resume_points
                    || layout.local_count != stream.var_count
            })
        {
            return false;
        }
        if self.private_temporary_slots.iter().any(|pair| {
            usize::try_from(pair.local).map_or(true, |local| local >= self.mir_local_count)
                || pair.slot >= stream.var_count
        }) || self
            .private_temporary_slots
            .windows(2)
            .any(|pair| pair[0].local >= pair[1].local)
            || self
                .private_temporary_slots
                .iter()
                .enumerate()
                .any(|(index, pair)| {
                    self.private_temporary_slots[index + 1..]
                        .iter()
                        .any(|later| later.slot == pair.slot)
                })
        {
            return false;
        }
        let Ok(ranges) = project_statement_pcs(stream.resume_points, stream.instructions.len())
        else {
            return false;
        };
        ranges.len() == self.statements.len()
            && ranges.iter().zip(&self.statements).all(|(range, row)| {
                *range == row.pc
                    && row.contract.point == range.statement
                    && range.statement.function == self.owner
                    && valid_values(&row.contract.uses, self.owner, self.mir_local_count)
                    && valid_values(&row.contract.defs, self.owner, self.mir_local_count)
                    && valid_values(&row.contract.live_after, self.owner, self.mir_local_count)
            })
    }

    fn fingerprint(&self, stream: &FusionStageStream<'_>) -> Result<String, serde_json::Error> {
        fusion_stage_source_fingerprint(self, stream)
    }
}

fn valid_values(values: &[RegionValueId], owner: ProgramFunctionId, count: usize) -> bool {
    values.iter().all(|value| {
        value.function == owner && usize::try_from(value.local).is_ok_and(|local| local < count)
    }) && values.windows(2).all(|pair| pair[0] < pair[1])
}

impl super::Bytecode {
    pub(crate) fn reseal_fusion_stage_after_owned_remap(&mut self) {
        let valid = self
            .fusion_metadata
            .stage_projection
            .as_mut()
            .is_none_or(|projection| {
                Arc::make_mut(projection).reseal_owned_remap(FusionStageStream {
                    instructions: &self.instructions,
                    catch_scopes: &self.catch_scopes,
                    resume_points: &self.resume_points,
                    source_id: self.source_id,
                    var_count: self.var_count,
                })
            });
        if !valid {
            self.fusion_metadata.stage_projection = None;
        }
    }

    pub(crate) fn validate_fusion_stage_projection(&self) -> Result<(), String> {
        let Some(projection) = self.fusion_metadata.stage_projection.as_ref() else {
            return Ok(());
        };
        projection
            .validate(FusionStageStream {
                instructions: &self.instructions,
                catch_scopes: &self.catch_scopes,
                resume_points: &self.resume_points,
                source_id: self.source_id,
                var_count: self.var_count,
            })
            .then_some(())
            .ok_or_else(|| "invalid selected MIR fusion-stage projection".to_string())
    }
}

impl super::FunctionBytecode {
    pub(crate) fn reseal_fusion_stage_after_owned_remap(&mut self) {
        let valid = self
            .fusion_stage_projection
            .as_mut()
            .is_none_or(|projection| {
                Arc::make_mut(projection).reseal_owned_remap(FusionStageStream {
                    instructions: &self.instructions,
                    catch_scopes: &self.catch_scopes,
                    resume_points: &self.resume_points,
                    source_id: self.source_id,
                    var_count: self.var_count,
                })
            });
        if !valid {
            self.fusion_stage_projection = None;
        }
    }

    pub(crate) fn validate_fusion_stage_projection(&self) -> Result<(), String> {
        let Some(projection) = self.fusion_stage_projection.as_ref() else {
            return Ok(());
        };
        (matches!(
            projection.selection_origin,
            None | Some(FusionStageSelectionOrigin::FunctionBody)
        ) && projection.portable_origin == self.portable_function
            && (projection.selection_origin != Some(FusionStageSelectionOrigin::FunctionBody)
                || projection.selected_function_layout.is_some())
            && projection
                .selected_function_layout
                .as_ref()
                .is_none_or(|layout| {
                    layout.function == self.function
                        && layout.resume_points == self.resume_points
                        && layout.local_count == self.var_count
                })
            && projection.validate(FusionStageStream {
                instructions: &self.instructions,
                catch_scopes: &self.catch_scopes,
                resume_points: &self.resume_points,
                source_id: self.source_id,
                var_count: self.var_count,
            }))
        .then_some(())
        .ok_or_else(|| "invalid selected function MIR fusion-stage projection".to_string())
    }
}

#[cfg(test)]
#[path = "fusion_stage/tests.rs"]
mod tests;
