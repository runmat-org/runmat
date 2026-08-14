use std::collections::{BTreeMap, BTreeSet};

use crate::ir::*;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_mir::{BasicBlockId, MirLocalId, MirOutputTarget, MirStmtKind};
use runmat_types::{CapabilitySet, EffectSet, ProgramFunctionId, RegionContract};

pub(super) fn lower_function(
    function: ProgramFunctionId,
    metadata: &runmat_mir::MirFunctionMetadata,
    body: &runmat_mir::MirBody,
    analysis: &runmat_mir::analysis::AnalysisStore,
    binding_names: Option<&BTreeMap<runmat_types::BindingId, String>>,
    regions: &[RegionContract],
) -> NativeCodegenResult<NativeFunction> {
    let local_count = body.locals.len();
    let _ = u32::try_from(local_count).map_err(|_| {
        NativeCodegenError::new(
            "native.lowering.local_count",
            "MIR local count exceeds the Native IR schema",
        )
        .at_function(function)
    })?;
    let locals = body
        .locals
        .iter()
        .map(|local| {
            let id = checked_local(local.id, function)?;
            let name = local
                .binding
                .map(|binding| {
                    binding_names
                        .and_then(|names| names.get(&binding))
                        .cloned()
                        .ok_or_else(|| {
                            NativeCodegenError::new(
                                "native.lowering.binding_name",
                                "bound MIR local has no canonical semantic name",
                            )
                            .at_function(function)
                        })
                })
                .transpose()?;
            Ok(NativeLocalMetadata {
                id,
                binding: local.binding,
                name,
                kind: NativeLocalKind::from(&local.kind),
            })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let mut ctx = super::context::FunctionLoweringContext::new(function, metadata.source, analysis);
    let mut parameter_sets = BTreeMap::new();
    let mut epoch_parameters = BTreeMap::new();
    for block in &body.blocks {
        let point = ctx.point(block.id, 0)?;
        let parameters = body
            .locals
            .iter()
            .map(|local| {
                Ok(NativeBlockParameter {
                    local: checked_local(local.id, function)?,
                    value: ctx.value(),
                    value_type: ctx.value_type(point, local.id),
                })
            })
            .collect::<NativeCodegenResult<Vec<_>>>()?;
        parameter_sets.insert(block.id, parameters);
        epoch_parameters.insert(block.id, ctx.value());
    }

    let mut blocks = Vec::with_capacity(body.blocks.len());
    let expected_sites = super::inventory::expected_sites(function, body)?;
    for block in &body.blocks {
        let parameters = parameter_sets.get(&block.id).cloned().ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.block_parameters",
                "missing preallocated block parameters",
            )
            .at_function(function)
        })?;
        let mut locals = parameters
            .iter()
            .map(|parameter| parameter.value)
            .collect::<Vec<_>>();
        let mut side_effect_epoch = *epoch_parameters.get(&block.id).ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.effect_epoch",
                "missing preallocated block effect epoch",
            )
            .at_function(function)
        })?;
        let mut instructions = Vec::new();
        let mut region_boundaries = lower_region_boundaries(
            &ctx,
            ctx.point(block.id, 0)?,
            &locals,
            side_effect_epoch,
            regions,
        )?;
        for (position, statement) in block.statements.iter().enumerate() {
            let before = ctx.point(block.id, position)?;
            let after = ctx.point(block.id, position + 1)?;
            let source = ctx.source_location(statement.span)?;
            let (effects, capabilities) = ctx.effect_delta(before, after);
            let analyzed_requirements = super::operation::OperationRequirements {
                effects,
                capabilities,
            };
            let mut produced = Vec::new();
            if let Some((value, output_locals)) = statement_rvalue_outputs(&statement.kind) {
                let result = rvalue_result(&statement.kind)?;
                let requirements = super::operation::merge_requirements(
                    super::operation::declared_requirements(value),
                    analyzed_requirements.clone(),
                );
                let mut instruction = super::operation::lower_rvalue(
                    &mut ctx,
                    super::operation::OperationSite {
                        point: before,
                        output_point: after,
                        phase: NativeSitePhase::Rvalue,
                        ordinal: 0,
                        source: source.clone(),
                    },
                    value,
                    &locals,
                    &output_locals,
                    result,
                    requirements,
                    &mut side_effect_epoch,
                )?;
                super::operation::install_frame_locals(&mut instruction, &locals);
                produced = instruction
                    .outputs
                    .iter()
                    .map(|output| output.value)
                    .collect();
                instructions.push(instruction);
            }

            let output_roots = statement_output_roots(&statement.kind);
            let statement_requirements = if statement_rvalue_outputs(&statement.kind).is_some() {
                super::operation::OperationRequirements {
                    effects: EffectSet::default(),
                    capabilities: CapabilitySet::default(),
                }
            } else {
                super::operation::merge_requirements(
                    super::operation::declared_statement_requirements(&statement.kind),
                    analyzed_requirements,
                )
            };
            let mut instruction = super::operation::lower_statement(
                &mut ctx,
                super::operation::OperationSite {
                    point: before,
                    output_point: after,
                    phase: NativeSitePhase::Statement,
                    ordinal: 0,
                    source,
                },
                &statement.kind,
                &locals,
                produced,
                &output_roots,
                statement_requirements,
                &mut side_effect_epoch,
            )?;
            super::operation::install_frame_locals(&mut instruction, &locals);
            for (root, output) in output_roots.iter().zip(&instruction.outputs) {
                let slot = locals.get_mut(root.0).ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.statement_output",
                        "statement updates a local outside its function",
                    )
                    .at_point(before)
                })?;
                *slot = output.value;
            }
            instructions.push(instruction);
            region_boundaries.extend(lower_region_boundaries(
                &ctx,
                after,
                &locals,
                side_effect_epoch,
                regions,
            )?);
        }

        let (terminator, mut terminator_instructions, _) = super::terminator::lower_terminator(
            &mut ctx,
            block,
            &locals,
            side_effect_epoch,
            &parameter_sets,
        )?;
        instructions.append(&mut terminator_instructions);
        region_boundaries.sort_by_key(|boundary| (boundary.point, boundary.region, boundary.kind));
        blocks.push(NativeBlock {
            id: checked_block(block.id, function)?,
            parameters,
            side_effect_epoch: *epoch_parameters.get(&block.id).expect("preallocated epoch"),
            region_boundaries,
            instructions,
            terminator,
        });
    }
    blocks.sort_by_key(|block| block.id);
    let entry = body.blocks.first().ok_or_else(|| {
        NativeCodegenError::new("native.lowering.blocks", "MIR function has no blocks")
            .at_function(function)
    })?;
    Ok(NativeFunction {
        id: function,
        source: metadata.source,
        name: metadata.name.0.clone(),
        abi: lower_abi(body, function)?,
        locals,
        index_expressions: super::index_expression::derive(body, function)?,
        entry: checked_block(entry.id, function)?,
        blocks,
        expected_sites,
    })
}

fn rvalue_result(statement: &MirStmtKind) -> NativeCodegenResult<NativeRvalueResult> {
    Ok(match statement {
        MirStmtKind::Assign { .. } => NativeRvalueResult::Assignment,
        MirStmtKind::MultiAssign { targets, .. } => NativeRvalueResult::MultiAssignment(
            u32::try_from(targets.targets.len()).map_err(|_| {
                NativeCodegenError::new(
                    "native.lowering.output_arity",
                    "MIR output arity exceeds the Native IR schema",
                )
            })?,
        ),
        MirStmtKind::Expr(_) => NativeRvalueResult::Discard,
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => {
            return Err(NativeCodegenError::new(
                "native.lowering.rvalue_role",
                "statement without an rvalue has no Native IR rvalue role",
            ))
        }
    })
}

fn statement_rvalue_outputs(
    statement: &MirStmtKind,
) -> Option<(&runmat_mir::MirRvalue, Vec<Option<MirLocalId>>)> {
    match statement {
        MirStmtKind::Assign { place, value } => {
            Some((value, vec![super::operation::root_local(place)]))
        }
        MirStmtKind::MultiAssign { targets, value } => Some((
            value,
            targets
                .targets
                .iter()
                .map(|target| match target {
                    MirOutputTarget::Place(place) => super::operation::root_local(place),
                    MirOutputTarget::Discard => None,
                })
                .collect(),
        )),
        MirStmtKind::Expr(value) => Some((value, Vec::new())),
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => None,
    }
}

fn statement_output_roots(statement: &MirStmtKind) -> Vec<MirLocalId> {
    let roots = match statement {
        MirStmtKind::Assign { place, .. } => {
            super::operation::root_local(place).into_iter().collect()
        }
        MirStmtKind::MultiAssign { targets, .. } => targets
            .targets
            .iter()
            .filter_map(|target| match target {
                MirOutputTarget::Place(place) => super::operation::root_local(place),
                MirOutputTarget::Discard => None,
            })
            .collect(),
        MirStmtKind::PlaceMutation(mutation) => super::operation::root_local(&mutation.place)
            .into_iter()
            .collect(),
        MirStmtKind::WorkspaceEffect { bindings, .. } => bindings.clone(),
        MirStmtKind::Expr(_) | MirStmtKind::EnvironmentEffect(_) => Vec::new(),
    };
    let mut seen = BTreeSet::new();
    roots
        .into_iter()
        .filter(|local| seen.insert(*local))
        .collect()
}

fn lower_abi(
    body: &runmat_mir::MirBody,
    function: ProgramFunctionId,
) -> NativeCodegenResult<NativeFunctionAbi> {
    let find = |binding| -> NativeCodegenResult<NativeLocalId> {
        body.locals
            .iter()
            .find(|local| local.binding == Some(binding))
            .map(|local| checked_local(local.id, function))
            .transpose()?
            .ok_or_else(|| {
                NativeCodegenError::new(
                    "native.lowering.function_abi",
                    "function ABI binding has no MIR local",
                )
                .at_function(function)
            })
    };
    Ok(NativeFunctionAbi {
        fixed_inputs: body
            .abi
            .fixed_inputs
            .iter()
            .copied()
            .map(find)
            .collect::<NativeCodegenResult<_>>()?,
        varargin: body.abi.varargin.map(find).transpose()?,
        fixed_outputs: body
            .abi
            .fixed_outputs
            .iter()
            .copied()
            .map(find)
            .collect::<NativeCodegenResult<_>>()?,
        varargout: body.abi.varargout.map(find).transpose()?,
        implicit_nargin: body.abi.implicit_nargin.map(find).transpose()?,
        implicit_nargout: body.abi.implicit_nargout.map(find).transpose()?,
    })
}

fn lower_region_boundaries(
    ctx: &super::context::FunctionLoweringContext<'_>,
    point: runmat_types::ProgramPointId,
    locals: &[NativeValueId],
    side_effect_epoch: NativeValueId,
    regions: &[RegionContract],
) -> NativeCodegenResult<Vec<NativeRegionBoundary>> {
    let mut boundaries = Vec::new();
    for region in regions {
        if region.id.function != ctx.function {
            continue;
        }
        if region.entry == point {
            boundaries.push(region_boundary(
                ctx,
                region,
                point,
                NativeRegionBoundaryKind::Entry,
                &region.live_in,
                locals,
                side_effect_epoch,
            )?);
        }
        if region.exits.binary_search(&point).is_ok() {
            boundaries.push(region_boundary(
                ctx,
                region,
                point,
                NativeRegionBoundaryKind::Exit,
                &region.live_out,
                locals,
                side_effect_epoch,
            )?);
        }
    }
    Ok(boundaries)
}

#[allow(clippy::too_many_arguments)]
fn region_boundary(
    ctx: &super::context::FunctionLoweringContext<'_>,
    region: &RegionContract,
    point: runmat_types::ProgramPointId,
    kind: NativeRegionBoundaryKind,
    live_values: &[runmat_types::RegionValueId],
    locals: &[NativeValueId],
    side_effect_epoch: NativeValueId,
) -> NativeCodegenResult<NativeRegionBoundary> {
    let resolve = |value: runmat_types::RegionValueId| {
        if value.function != ctx.function {
            return Err(NativeCodegenError::new(
                "native.lowering.region_value",
                "region value belongs to another function",
            )
            .at_point(point));
        }
        locals.get(value.local as usize).copied().ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.region_value",
                "region value has no MIR local at its boundary",
            )
            .at_point(point)
        })
    };
    let live_values = live_values
        .iter()
        .copied()
        .map(|value| {
            Ok(NativeRegionValueBinding {
                value,
                ssa: resolve(value)?,
            })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let guards = region
        .guards
        .iter()
        .cloned()
        .map(|contract| {
            let value = guard_value(&contract.condition).map(resolve).transpose()?;
            Ok(NativeRegionGuard { contract, value })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let source = NativeSourceLocation {
        source: region.source,
        span: region.span,
    };
    Ok(NativeRegionBoundary {
        region: region.id,
        point,
        kind,
        live_values,
        guards,
        frame_state: super::context::frame_state(
            point,
            source,
            locals,
            Vec::new(),
            side_effect_epoch,
        ),
    })
}

fn guard_value(
    condition: &runmat_types::RegionGuardCondition,
) -> Option<runmat_types::RegionValueId> {
    match condition {
        runmat_types::RegionGuardCondition::ValueFact { value, .. }
        | runmat_types::RegionGuardCondition::Shape { value, .. }
        | runmat_types::RegionGuardCondition::Class { value, .. }
        | runmat_types::RegionGuardCondition::Residency { value, .. }
        | runmat_types::RegionGuardCondition::Alias { value, .. } => Some(*value),
        runmat_types::RegionGuardCondition::RuntimeState { .. }
        | runmat_types::RegionGuardCondition::Capability { .. } => None,
    }
}

pub(super) fn checked_local(
    local: MirLocalId,
    function: ProgramFunctionId,
) -> NativeCodegenResult<NativeLocalId> {
    u32::try_from(local.0).map(NativeLocalId).map_err(|_| {
        NativeCodegenError::new(
            "native.lowering.local_identity",
            "MIR local identity exceeds the Native IR schema",
        )
        .at_function(function)
    })
}

pub(super) fn checked_block(
    block: BasicBlockId,
    function: ProgramFunctionId,
) -> NativeCodegenResult<NativeBlockId> {
    u32::try_from(block.0).map(NativeBlockId).map_err(|_| {
        NativeCodegenError::new(
            "native.lowering.block_identity",
            "MIR block identity exceeds the Native IR schema",
        )
        .at_function(function)
    })
}
