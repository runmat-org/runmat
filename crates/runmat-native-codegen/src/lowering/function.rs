use std::collections::{BTreeMap, BTreeSet};

use crate::ir::*;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_mir::{BasicBlockId, MirLocalId, MirOutputTarget, MirStmtKind};
use runmat_types::{FunctionId, RegionContract};

pub(super) fn lower_function(
    function: FunctionId,
    mir_body_digest: runmat_execution::Digest,
    input: &super::assembly::NativeLoweringInput<'_>,
    analysis: &runmat_mir::analysis::AnalysisStore,
    metadata: &runmat_mir::MirFunctionMetadata,
    retained_callable_parent: Option<runmat_hir::FunctionId>,
    body: &runmat_mir::MirBody,
) -> NativeCodegenResult<NativeFunction> {
    let assembly = input.mir;
    let binding_names = input.binding_names;
    let regions = &input.manifest.regions;
    body.validate_expression_regions().map_err(|error| {
        NativeCodegenError::new(
            "native.lowering.expression_region",
            format!("invalid MIR expression region: {error}"),
        )
        .at_function(function)
    })?;
    body.validate_statement_outputs().map_err(|error| {
        NativeCodegenError::new(
            "native.lowering.statement_output",
            format!("invalid MIR statement output: {error}"),
        )
        .at_function(function)
    })?;
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
    let mut binding_locals = locals
        .iter()
        .filter_map(|local| {
            local.binding.map(|binding| NativeBindingLocal {
                binding,
                local: local.id,
            })
        })
        .collect::<Vec<_>>();
    binding_locals.sort_by_key(|entry| entry.binding);
    let projection = assembly
        .lexical_environment_projection(function)
        .map_err(|error| {
            NativeCodegenError::new("native.lowering.lexical_environment", error)
                .at_function(function)
        })?;
    let mut lexical_bindings = locals
        .iter()
        .filter_map(|local| {
            let binding = local.binding?;
            let kind = if projection.owned.contains(&binding) {
                NativeLexicalBindingKind::Owned
            } else if projection.inherited.contains(&binding) {
                NativeLexicalBindingKind::Inherited
            } else if projection.transit.contains(&binding) {
                NativeLexicalBindingKind::Transit
            } else {
                return None;
            };
            Some(NativeLexicalBinding {
                binding,
                local: local.id,
                kind,
            })
        })
        .collect::<Vec<_>>();
    lexical_bindings.sort_by_key(|entry| entry.binding);
    if lexical_bindings.len()
        != projection.owned.len() + projection.inherited.len() + projection.transit.len()
    {
        return Err(NativeCodegenError::new(
            "native.lowering.lexical_environment",
            "shared lexical binding has no unique Native IR local",
        )
        .at_function(function));
    }
    let inherited_storage_bindings = projection
        .inherited_storage
        .iter()
        .map(|entry| {
            let local = locals
                .iter()
                .find(|local| local.binding == Some(entry.binding))
                .ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.inherited_storage",
                        "inherited canonical storage has no Native IR local",
                    )
                    .at_function(function)
                })?;
            let name = local.name.clone().ok_or_else(|| {
                NativeCodegenError::new(
                    "native.lowering.inherited_storage",
                    "inherited canonical storage has no semantic name",
                )
                .at_function(function)
            })?;
            let owner_function = assembly
                .functions
                .get(&entry.owner)
                .map(|metadata| metadata.name.0.clone())
                .ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.inherited_storage",
                        "inherited canonical storage owner is absent",
                    )
                    .at_function(function)
                })?;
            let kind = match entry.kind {
                runmat_mir::MirInheritedStorageKind::Global => NativeInheritedStorageKind::Global,
                runmat_mir::MirInheritedStorageKind::Persistent => {
                    NativeInheritedStorageKind::Persistent
                }
            };
            Ok(NativeInheritedStorageBinding {
                binding: entry.binding,
                local: local.id,
                owner: entry.owner,
                name,
                owner_function,
                kind,
            })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let closure_captures = metadata
        .captures
        .iter()
        .map(|capture| {
            let by_value = locals.iter().find(|local| {
                local.binding == Some(capture.binding) && local.kind == NativeLocalKind::Capture
            });
            let lexical = lexical_bindings.iter().find(|binding| {
                binding.binding == capture.binding
                    && matches!(
                        binding.kind,
                        NativeLexicalBindingKind::Inherited | NativeLexicalBindingKind::Transit
                    )
            });
            let storage = inherited_storage_bindings
                .iter()
                .find(|binding| binding.binding == capture.binding);
            let (local, disposition) = match (by_value, lexical, storage) {
                (Some(local), None, None) => (local.id, NativeClosureCaptureDisposition::ByValue),
                (Some(local), Some(binding), None)
                    if binding.kind == NativeLexicalBindingKind::Transit
                        && binding.local == local.id =>
                {
                    (local.id, NativeClosureCaptureDisposition::TransitSnapshot)
                }
                (local, Some(binding), None)
                    if binding.kind == NativeLexicalBindingKind::Inherited
                        && local.is_none_or(|local| local.id == binding.local) =>
                {
                    (binding.local, NativeClosureCaptureDisposition::Lexical)
                }
                (local, None, Some(binding))
                    if local.is_none_or(|local| local.id == binding.local) =>
                {
                    (
                        binding.local,
                        NativeClosureCaptureDisposition::InheritedStorage,
                    )
                }
                _ => {
                    return Err(NativeCodegenError::new(
                        "native.lowering.closure_capture",
                        "closure capture has no coherent Native frame authority",
                    )
                    .at_function(function))
                }
            };
            Ok(NativeClosureCapture {
                binding: capture.binding,
                local,
                disposition,
            })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let mut ctx =
        super::context::FunctionLoweringContext::new(function, metadata.source, analysis)?;
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
    let expected_sites = super::inventory::expected_sites(ctx.portable_function, body)?;
    let operation_contracts = analysis
        .operation_contracts
        .iter()
        .filter(|contract| contract.site.point.function == ctx.portable_function)
        .cloned()
        .collect::<Vec<_>>();
    if operation_contracts
        .iter()
        .map(|contract| contract.site.clone())
        .ne(expected_sites.iter().cloned())
    {
        return Err(NativeCodegenError::new(
            "native.lowering.operation_contract_coverage",
            "analysis operation contracts do not exactly cover canonical MIR sites",
        )
        .at_function(function));
    }
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
            let mut produced = Vec::new();
            if let Some((value, output_locals)) = statement_rvalue_outputs(&statement.kind) {
                let result = rvalue_result(&statement.kind)?;
                let operation_site = NativeMirSite {
                    point: before,
                    phase: NativeSitePhase::Rvalue,
                    ordinal: 0,
                    construct: runmat_mir::rvalue_construct_kind(value),
                };
                let requirements = ctx.operation_contract(&operation_site)?;
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
            let statement_site = NativeMirSite {
                point: before,
                phase: NativeSitePhase::Statement,
                ordinal: 0,
                construct: runmat_mir::statement_construct_kind(&statement.kind),
            };
            let statement_requirements = ctx.operation_contract(&statement_site)?;
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
            catch_scopes: body
                .catch_scopes
                .get(&block.id)
                .cloned()
                .unwrap_or_default(),
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
    let abi = lower_abi(body, function)?;
    let argument_validations = lower_argument_validations(metadata, &locals, &abi, function)?;
    let capabilities = analysis
        .function_for_local(function)
        .ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.function_analysis",
                "MIR function has no canonical analysis summary",
            )
            .at_function(function)
        })?
        .capabilities
        .clone();
    let expression_continuation_geometry =
        crate::NativeExpressionContinuationGeometry::for_blocks(&blocks).ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.expression_continuation_geometry",
                "expression continuation geometry exceeds portable capacity",
            )
            .at_function(function)
        })?;
    Ok(NativeFunction {
        spawn_safety: runmat_mir::analysis::function_spawn_safety(assembly, function).ok_or_else(
            || {
                NativeCodegenError::new(
                    "native.lowering.spawn_contract",
                    "function has no source MIR spawn contract",
                )
                .at_function(function)
            },
        )?,
        id: function,
        portable_id: ctx.portable_function,
        mir_body_digest,
        source: metadata.source,
        name: metadata.name.0.clone(),
        source_role: metadata.source_role,
        lexical_parent: retained_callable_parent,
        class_method_owner: metadata.class_method_owner.clone(),
        capabilities,
        abi,
        argument_validations,
        locals,
        binding_locals,
        closure_captures,
        lexical_bindings,
        inherited_storage_bindings,
        creates_lexical_environment: projection.creates_activation,
        entry: checked_block(entry.id, function)?,
        blocks,
        expression_continuation_geometry,
        operation_contracts,
        expected_sites,
    })
}

pub(super) fn lower_argument_validations(
    metadata: &runmat_mir::MirFunctionMetadata,
    locals: &[NativeLocalMetadata],
    abi: &NativeFunctionAbi,
    function: FunctionId,
) -> NativeCodegenResult<Vec<NativeFunctionArgumentValidation>> {
    let mut validations = metadata
        .argument_validations
        .iter()
        .map(|validation| {
            let input = locals
                .iter()
                .find(|local| local.binding == Some(validation.binding))
                .map(|local| local.id)
                .ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.argument_validation",
                        "argument validation binding has no retained Native IR local",
                    )
                    .at_function(function)
                })?;
            if !abi.fixed_inputs.contains(&input) {
                return Err(NativeCodegenError::new(
                    "native.lowering.argument_validation",
                    "argument validation binding is not a fixed function input",
                )
                .at_function(function));
            }
            Ok(NativeFunctionArgumentValidation {
                input,
                size: validation.size.clone(),
                class_name: validation.class_name.clone(),
                validators: validation.validators.clone(),
                default_value: validation.default_value.clone(),
            })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    validations.sort_by_key(|validation| validation.input);
    Ok(validations)
}

fn rvalue_result(statement: &MirStmtKind) -> NativeCodegenResult<NativeRvalueResult> {
    Ok(match statement {
        MirStmtKind::Assign { .. } | MirStmtKind::PreparedAssign { .. } => {
            NativeRvalueResult::Assignment
        }
        MirStmtKind::MultiAssign { targets, .. } => {
            let _ = u32::try_from(targets.targets.len()).map_err(|_| {
                NativeCodegenError::new(
                    "native.lowering.output_arity",
                    "MIR output arity exceeds the Native IR schema",
                )
            })?;
            NativeRvalueResult::MultiAssignment(targets.clone())
        }
        MirStmtKind::SequenceAssign { target, .. } => {
            NativeRvalueResult::SequenceAssignment(target.clone())
        }
        MirStmtKind::Expr(value) if value.has_statement_result() => NativeRvalueResult::Expression,
        MirStmtKind::Expr(_) => NativeRvalueResult::Discard,
        MirStmtKind::StatementOutput(_) => {
            return Err(NativeCodegenError::new(
                "native.lowering.rvalue_role",
                "statement output has no Native IR rvalue role",
            ))
        }
        MirStmtKind::PrepareFixedAssignment(_)
        | MirStmtKind::PlaceMutation(_)
        | MirStmtKind::CaptureSequence { .. }
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
        MirStmtKind::PreparedAssign { preparation, value } => Some((
            value,
            vec![super::operation::root_local(&preparation.mutation.place)],
        )),
        MirStmtKind::Assign { place, value } => {
            Some((value, vec![super::operation::root_local(place)]))
        }
        MirStmtKind::MultiAssign { targets, value } => {
            let outputs = if targets
                .targets
                .iter()
                .any(|target| matches!(target, MirOutputTarget::Sequence(_)))
            {
                Vec::new()
            } else {
                targets
                    .targets
                    .iter()
                    .map(|target| match target {
                        MirOutputTarget::Place(place) => super::operation::root_local(place),
                        MirOutputTarget::Sequence(_) => None,
                        MirOutputTarget::Discard => None,
                    })
                    .collect()
            };
            Some((value, outputs))
        }
        MirStmtKind::SequenceAssign { value, .. } => Some((value, Vec::new())),
        MirStmtKind::StatementOutput(_) => None,
        MirStmtKind::Expr(value) => Some((
            value,
            value
                .has_statement_result()
                .then_some(None)
                .into_iter()
                .collect(),
        )),
        MirStmtKind::PrepareFixedAssignment(_)
        | MirStmtKind::PlaceMutation(_)
        | MirStmtKind::CaptureSequence { .. }
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => None,
    }
}

fn statement_output_roots(statement: &MirStmtKind) -> Vec<MirLocalId> {
    let roots = match statement {
        MirStmtKind::PreparedAssign { preparation, .. } => {
            super::operation::root_local(&preparation.mutation.place)
                .into_iter()
                .collect()
        }
        MirStmtKind::PrepareFixedAssignment(preparation) => match preparation.output() {
            Some(runmat_mir::MirFixedSelectorDestination::Scalar(local)) => vec![local],
            Some(runmat_mir::MirFixedSelectorDestination::Sequence(_)) | None => Vec::new(),
        },
        MirStmtKind::Assign { place, .. } => {
            super::operation::root_local(place).into_iter().collect()
        }
        MirStmtKind::MultiAssign { targets, .. } => targets
            .targets
            .iter()
            .filter_map(|target| match target {
                MirOutputTarget::Place(place) => super::operation::root_local(place),
                MirOutputTarget::Sequence(target) => super::operation::root_local(target.base()),
                MirOutputTarget::Discard => None,
            })
            .collect(),
        MirStmtKind::SequenceAssign { target, .. } => {
            sequence_target_root_local(target).into_iter().collect()
        }
        MirStmtKind::PlaceMutation(mutation) => super::operation::root_local(&mutation.place)
            .into_iter()
            .collect(),
        MirStmtKind::WorkspaceEffect { bindings, .. } => bindings.clone(),
        MirStmtKind::Expr(_)
        | MirStmtKind::StatementOutput(_)
        | MirStmtKind::CaptureSequence { .. }
        | MirStmtKind::EnvironmentEffect(_) => Vec::new(),
    };
    let mut seen = BTreeSet::new();
    roots
        .into_iter()
        .filter(|local| seen.insert(*local))
        .collect()
}

fn sequence_target_root_local(target: &runmat_mir::MirSequenceTarget) -> Option<MirLocalId> {
    super::operation::root_local(target.base())
}

pub(super) fn lower_abi(
    body: &runmat_mir::MirBody,
    function: FunctionId,
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
            .filter(|binding| Some(**binding) != body.abi.varargin)
            .copied()
            .map(find)
            .collect::<NativeCodegenResult<_>>()?,
        varargin: body.abi.varargin.map(find).transpose()?,
        fixed_outputs: body
            .abi
            .fixed_output_bindings()
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
        if region.id.function != ctx.portable_function {
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
        if value.function != ctx.portable_function {
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
    function: FunctionId,
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
    function: FunctionId,
) -> NativeCodegenResult<NativeBlockId> {
    u32::try_from(block.0).map(NativeBlockId).map_err(|_| {
        NativeCodegenError::new(
            "native.lowering.block_identity",
            "MIR block identity exceeds the Native IR schema",
        )
        .at_function(function)
    })
}
