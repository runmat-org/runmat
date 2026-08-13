use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::{NativeCodegenError, NativeCodegenResult, NATIVE_IR_SCHEMA_VERSION};

impl NativeAssembly {
    pub fn verify(&self) -> NativeCodegenResult<()> {
        if self.schema_version != NATIVE_IR_SCHEMA_VERSION {
            return Err(NativeCodegenError::new(
                "native.ir.schema_version",
                "unsupported Native IR schema version",
            ));
        }
        if self.executable_identity.program != self.program {
            return Err(NativeCodegenError::new(
                "native.ir.program_identity",
                "assembly and executable identities name different programs",
            ));
        }
        self.target.validate()?;
        if self.native_cache_key != self.target.cache_key(&self.executable_cache_key)? {
            return Err(NativeCodegenError::new(
                "native.ir.cache_key",
                "native cache key does not bind the executable, target, schema, and ABI",
            ));
        }
        for region in &self.requirements.regions {
            region
                .validate()
                .map_err(|error| schema_error("native.ir.region", error))?;
            if !region
                .capabilities
                .0
                .is_subset(&self.requirements.capabilities.0)
            {
                return Err(NativeCodegenError::new(
                    "native.ir.region_capabilities",
                    "region requirement is absent from the executable capability set",
                ));
            }
        }
        self.requirements
            .interop
            .validate()
            .map_err(|error| schema_error("native.ir.interop", error))?;
        self.requirements
            .parallel
            .validate()
            .map_err(|error| schema_error("native.ir.parallel", error))?;
        if self
            .requirements
            .parallel
            .parfor_regions
            .iter()
            .map(|region| &region.capabilities)
            .chain(
                self.requirements
                    .parallel
                    .spmd_regions
                    .iter()
                    .map(|region| &region.capabilities),
            )
            .any(|required| !required.0.is_subset(&self.requirements.capabilities.0))
        {
            return Err(NativeCodegenError::new(
                "native.ir.parallel_capabilities",
                "parallel requirement is absent from the executable capability set",
            ));
        }
        ensure_sorted_unique_by("native.ir.functions", &self.functions, |function| {
            function.id
        })?;
        let functions = self
            .functions
            .iter()
            .map(|function| (function.id, function))
            .collect::<BTreeMap<_, _>>();
        if self.entrypoints.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .entrypoints
                .iter()
                .any(|entrypoint| !functions.contains_key(entrypoint))
        {
            return Err(NativeCodegenError::new(
                "native.ir.entrypoints",
                "entrypoints must be sorted, unique, and name retained functions",
            ));
        }
        for function in &self.functions {
            verify_function(
                function,
                &self.requirements.regions,
                &self.requirements.capabilities,
            )?;
        }
        Ok(())
    }
}

fn verify_function(
    function: &NativeFunction,
    regions: &[runmat_types::RegionContract],
    capabilities: &runmat_types::CapabilitySet,
) -> NativeCodegenResult<()> {
    let error = |code, message| NativeCodegenError::new(code, message).at_function(function.id);
    if function.blocks.is_empty() {
        return Err(error("native.ir.blocks", "function must contain a block"));
    }
    for local in function
        .abi
        .fixed_inputs
        .iter()
        .chain(function.abi.varargin.iter())
        .chain(&function.abi.fixed_outputs)
        .chain(function.abi.varargout.iter())
        .chain(function.abi.implicit_nargin.iter())
        .chain(function.abi.implicit_nargout.iter())
    {
        if local.0 >= function.local_count {
            return Err(error(
                "native.ir.function_abi",
                "function ABI names a local outside the function",
            ));
        }
    }
    ensure_sorted_unique_by("native.ir.blocks", &function.blocks, |block| block.id)
        .map_err(|error| error.at_function(function.id))?;
    let blocks = function
        .blocks
        .iter()
        .map(|block| (block.id, block))
        .collect::<BTreeMap<_, _>>();
    if !blocks.contains_key(&function.entry) {
        return Err(error(
            "native.ir.entry",
            "function entry does not name a retained block",
        ));
    }

    let mut all_values = BTreeSet::new();
    let mut all_instructions = BTreeSet::new();
    let mut all_safepoints = BTreeSet::new();
    let mut actual_sites = BTreeSet::new();
    for block in &function.blocks {
        if block.instructions.iter().any(|instruction| {
            instruction.site.point.block != block.id.0
                || instruction.site.point.function != function.id
                || instruction.source.source != function.source
        }) || block.terminator.site.point.block != block.id.0
            || block.terminator.site.point.function != function.id
            || block.terminator.source.source != function.source
            || block.terminator.source.span.start > block.terminator.source.span.end
        {
            return Err(error(
                "native.ir.site_block",
                "sites must name their containing function, block, and source",
            ));
        }
        if block.parameters.len() != function.local_count as usize
            || block
                .parameters
                .iter()
                .enumerate()
                .any(|(index, parameter)| parameter.local.0 as usize != index)
        {
            return Err(error(
                "native.ir.block_parameters",
                "every block must declare one ordered parameter per MIR local",
            ));
        }
        let mut available = BTreeSet::new();
        if !all_values.insert(block.side_effect_epoch) || !available.insert(block.side_effect_epoch)
        {
            return Err(error(
                "native.ir.value_identity",
                "SSA value identities must be unique within a function",
            ));
        }
        for parameter in &block.parameters {
            if !all_values.insert(parameter.value) || !available.insert(parameter.value) {
                return Err(error(
                    "native.ir.value_identity",
                    "SSA value identities must be unique within a function",
                ));
            }
        }
        let mut current_effect_epoch = block.side_effect_epoch;
        let mut boundary_states =
            BTreeMap::from([(0_u32, (available.clone(), current_effect_epoch))]);
        for instruction in &block.instructions {
            if !all_instructions.insert(instruction.id) {
                return Err(error(
                    "native.ir.instruction_identity",
                    "instruction identities must be unique within a function",
                ));
            }
            verify_instruction(
                function,
                instruction,
                &available,
                current_effect_epoch,
                &mut all_safepoints,
            )?;
            if !instruction.capabilities.0.is_subset(&capabilities.0) {
                return Err(error(
                    "native.ir.capabilities",
                    "instruction requirement is absent from the executable capability set",
                ));
            }
            actual_sites.insert(instruction.site.clone());
            for output in &instruction.outputs {
                if !all_values.insert(output.value) || !available.insert(output.value) {
                    return Err(error(
                        "native.ir.value_identity",
                        "SSA value identities must be unique within a function",
                    ));
                }
            }
            if let Some(output) = instruction.effect_epoch_output {
                if !all_values.insert(output) || !available.insert(output) {
                    return Err(error(
                        "native.ir.value_identity",
                        "SSA value identities must be unique within a function",
                    ));
                }
                current_effect_epoch = output;
            }
            if instruction.site.phase == NativeSitePhase::Statement {
                let position = instruction
                    .site
                    .point
                    .position
                    .checked_add(1)
                    .ok_or_else(|| {
                        error(
                            "native.ir.boundary_position",
                            "statement position exceeds the Native IR schema",
                        )
                    })?;
                boundary_states.insert(position, (available.clone(), current_effect_epoch));
            }
        }
        for boundary in &block.region_boundaries {
            let (boundary_available, boundary_epoch) = boundary_states
                .get(&boundary.point.position)
                .ok_or_else(|| {
                    error(
                        "native.ir.region_boundary_point",
                        "region boundary does not name a materializable MIR point",
                    )
                })?;
            verify_frame_state(function, &boundary.frame_state, boundary_available)?;
            if boundary.frame_state.point != boundary.point
                || boundary.frame_state.side_effect_epoch != *boundary_epoch
                || boundary.live_values.iter().any(|binding| {
                    !boundary_available.contains(&binding.ssa)
                        || boundary
                            .frame_state
                            .locals
                            .get(binding.value.local as usize)
                            .is_none_or(|local| local.value != binding.ssa)
                })
                || boundary.guards.iter().any(|guard| {
                    guard
                        .value
                        .is_some_and(|value| !boundary_available.contains(&value))
                })
            {
                return Err(error(
                    "native.ir.region_boundary_state",
                    "region boundary must bind guards and live values to exact available SSA state",
                ));
            }
        }
        verify_frame_state(function, &block.terminator.frame_state, &available)?;
        if block.terminator.frame_state.side_effect_epoch != current_effect_epoch
            || block.terminator.frame_state.point != block.terminator.site.point
            || block.terminator.frame_state.source != block.terminator.source
        {
            return Err(error(
                "native.ir.terminator_frame_state",
                "terminator frame state must name its exact site and current effect epoch",
            ));
        }
        if block.terminator.class != block.terminator.site.construct.native_lowering_class() {
            return Err(error(
                "native.ir.terminator_class",
                "terminator class differs from canonical MIR classification",
            ));
        }
        let terminator_requires_safepoint = block.terminator.class
            != runmat_mir::NativeLoweringClass::NativeOperation
            && block.terminator.class != runmat_mir::NativeLoweringClass::ProvenUnreachable;
        if block.terminator.safepoint.is_some() != terminator_requires_safepoint {
            return Err(error(
                "native.ir.terminator_safepoint",
                "slow or suspending terminators require an explicit safepoint",
            ));
        }
        if let Some(safepoint) = block.terminator.safepoint {
            if !all_safepoints.insert(safepoint) {
                return Err(error(
                    "native.ir.safepoint_identity",
                    "safepoint identities must be unique within a function",
                ));
            }
        }
        actual_sites.insert(block.terminator.site.clone());
        verify_terminator(
            function,
            &block.terminator.kind,
            &available,
            current_effect_epoch,
            &blocks,
        )?;
    }

    let expected = function
        .expected_sites
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if expected.len() != function.expected_sites.len() || expected != actual_sites {
        return Err(error(
            "native.ir.construct_coverage",
            "Native IR sites do not exactly cover every canonical MIR construct",
        ));
    }
    if function
        .expected_sites
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(error(
            "native.ir.expected_sites",
            "expected MIR sites must be sorted and unique",
        ));
    }
    verify_region_boundaries(function, regions)?;
    Ok(())
}

fn verify_instruction(
    function: &NativeFunction,
    instruction: &NativeInstruction,
    available: &BTreeSet<NativeValueId>,
    current_effect_epoch: NativeValueId,
    all_safepoints: &mut BTreeSet<NativeSafepointId>,
) -> NativeCodegenResult<()> {
    let error = |code, message| {
        NativeCodegenError::new(code, message)
            .at_point(instruction.site.point)
            .for_construct(instruction.site.construct)
    };
    if instruction.site.point.function != function.id {
        return Err(error(
            "native.ir.site_function",
            "instruction site belongs to another function",
        ));
    }
    if instruction.source.span.start > instruction.source.span.end {
        return Err(error(
            "native.ir.source_span",
            "instruction source span end precedes its start",
        ));
    }
    if instruction.class != instruction.site.construct.native_lowering_class() {
        return Err(error(
            "native.ir.instruction_class",
            "instruction class differs from canonical MIR classification",
        ));
    }
    let payload_construct = match &instruction.operation {
        NativeOperation::Rvalue { value, .. } => runmat_mir::rvalue_construct_kind(value),
        NativeOperation::Statement(statement) => runmat_mir::statement_construct_kind(statement),
    };
    if payload_construct != instruction.site.construct {
        return Err(error(
            "native.ir.operation_construct",
            "operation payload and declared construct differ",
        ));
    }
    if let NativeOperation::Rvalue { value, .. } = &instruction.operation {
        let inventory = runmat_mir::rvalue_construct_inventory(value);
        if instruction.embedded_constructs != inventory[1..] {
            return Err(error(
                "native.ir.embedded_constructs",
                "embedded construct inventory differs from canonical MIR",
            ));
        }
        if inventory.iter().any(|construct| {
            construct.native_lowering_class()
                == runmat_mir::NativeLoweringClass::CapabilityRejection
        }) {
            return Err(error(
                "native.capability.distributed_core_pending",
                "rvalue contains a construct requiring the R25 distributed core",
            ));
        }
        let (declared_effects, declared_capabilities) =
            runmat_mir::rvalue_declared_requirements(value);
        if !declared_effects.0.is_subset(&instruction.effects.0)
            || !declared_capabilities
                .0
                .is_subset(&instruction.capabilities.0)
        {
            return Err(error(
                "native.ir.declared_requirements",
                "instruction omits effects or capabilities declared by canonical MIR",
            ));
        }
    } else if let NativeOperation::Statement(statement) = &instruction.operation {
        if !instruction.embedded_constructs.is_empty() {
            return Err(error(
                "native.ir.embedded_constructs",
                "statement instruction cannot declare embedded constructs",
            ));
        }
        if !runmat_mir::statement_declared_effects(statement)
            .0
            .is_subset(&instruction.effects.0)
        {
            return Err(error(
                "native.ir.declared_requirements",
                "instruction omits effects declared by canonical MIR",
            ));
        }
    }
    if instruction
        .inputs
        .iter()
        .any(|value| !available.contains(value))
    {
        return Err(error(
            "native.ir.instruction_input",
            "instruction input is not defined earlier in its block",
        ));
    }
    if instruction
        .inputs
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != instruction.inputs.len()
        || instruction
            .outputs
            .iter()
            .map(|output| output.value)
            .collect::<BTreeSet<_>>()
            .len()
            != instruction.outputs.len()
    {
        return Err(error(
            "native.ir.instruction_values",
            "instruction inputs and outputs must each be unique",
        ));
    }
    let expected_outputs = expected_output_count(instruction)?;
    if instruction.outputs.len() != expected_outputs {
        return Err(error(
            "native.ir.output_arity",
            "instruction output arity differs from its canonical MIR role",
        ));
    }
    let requires_safepoint = instruction.class != runmat_mir::NativeLoweringClass::NativeOperation
        || !instruction.effects.0.is_empty();
    if requires_safepoint != instruction.safepoint.is_some()
        || instruction.safepoint.is_some() != instruction.frame_state.is_some()
    {
        return Err(error(
            "native.ir.safepoint",
            "slow, suspending, or effectful operations require exact safepoint state",
        ));
    }
    if instruction.effect_epoch_output.is_some() == instruction.effects.0.is_empty() {
        return Err(error(
            "native.ir.effect_epoch_output",
            "exactly effectful instructions must advance the side-effect epoch",
        ));
    }
    if let Some(safepoint) = instruction.safepoint {
        if !all_safepoints.insert(safepoint) {
            return Err(error(
                "native.ir.safepoint_identity",
                "safepoint identities must be unique within a function",
            ));
        }
    }
    if let Some(state) = &instruction.frame_state {
        verify_frame_state(function, state, available)?;
        if state.point != instruction.site.point
            || state.source != instruction.source
            || state.side_effect_epoch != current_effect_epoch
            || state.operands != instruction.inputs
        {
            return Err(error(
                "native.ir.frame_state_site",
                "frame state must describe the exact instruction site and current effect epoch",
            ));
        }
    }
    Ok(())
}

fn expected_output_count(instruction: &NativeInstruction) -> NativeCodegenResult<usize> {
    match (&instruction.site.phase, &instruction.operation) {
        (
            NativeSitePhase::TerminatorRvalue,
            NativeOperation::Rvalue {
                result: NativeRvalueResult::Terminator,
                ..
            },
        ) => Ok(1),
        (NativeSitePhase::Rvalue, NativeOperation::Rvalue { value, result }) => {
            let arity = match result {
                NativeRvalueResult::Assignment => 1,
                NativeRvalueResult::MultiAssignment(arity) => *arity as usize,
                NativeRvalueResult::Discard => 0,
                NativeRvalueResult::Terminator => {
                    return Err(NativeCodegenError::new(
                        "native.ir.operation_phase",
                        "terminator rvalue role appears at a statement site",
                    ))
                }
            };
            if let runmat_mir::MirRvalue::Call(call) = value {
                if call.requested_outputs.fixed_count() != arity {
                    return Err(NativeCodegenError::new(
                        "native.ir.call_output_arity",
                        "call result role differs from requested output arity",
                    ));
                }
            }
            Ok(arity)
        }
        (NativeSitePhase::Statement, NativeOperation::Statement(statement)) => {
            let mut roots = BTreeSet::new();
            match statement {
                runmat_mir::MirStmtKind::Assign { place, .. } => {
                    if let Some(local) = root_local(place) {
                        roots.insert(local);
                    }
                }
                runmat_mir::MirStmtKind::MultiAssign { targets, .. } => {
                    for target in &targets.targets {
                        if let runmat_mir::MirOutputTarget::Place(place) = target {
                            if let Some(local) = root_local(place) {
                                roots.insert(local);
                            }
                        }
                    }
                }
                runmat_mir::MirStmtKind::PlaceMutation(mutation) => {
                    if let Some(local) = root_local(&mutation.place) {
                        roots.insert(local);
                    }
                }
                runmat_mir::MirStmtKind::WorkspaceEffect { bindings, .. } => {
                    roots.extend(bindings.iter().copied());
                }
                runmat_mir::MirStmtKind::Expr(_)
                | runmat_mir::MirStmtKind::EnvironmentEffect(_) => {}
            }
            Ok(roots.len())
        }
        _ => Err(NativeCodegenError::new(
            "native.ir.operation_phase",
            "operation payload is invalid for its Native IR site phase",
        )),
    }
}

fn root_local(place: &runmat_mir::MirPlace) -> Option<runmat_mir::MirLocalId> {
    match place {
        runmat_mir::MirPlace::Local(local) => Some(*local),
        runmat_mir::MirPlace::Binding(_) => None,
        runmat_mir::MirPlace::Member(base, _)
        | runmat_mir::MirPlace::DynamicMember(base, _)
        | runmat_mir::MirPlace::Index(base, _) => root_local(base),
    }
}

fn verify_frame_state(
    function: &NativeFunction,
    state: &NativeFrameState,
    available: &BTreeSet<NativeValueId>,
) -> NativeCodegenResult<()> {
    if state.point.function != function.id
        || !available.contains(&state.side_effect_epoch)
        || state.locals.len() != function.local_count as usize
        || state.locals.iter().enumerate().any(|(index, local)| {
            local.local.0 as usize != index || !available.contains(&local.value)
        })
        || state
            .operands
            .iter()
            .any(|value| !available.contains(value))
    {
        return Err(NativeCodegenError::new(
            "native.ir.frame_state",
            "frame state must materialize every ordered local from available SSA values",
        )
        .at_function(function.id));
    }
    Ok(())
}

fn verify_terminator(
    function: &NativeFunction,
    terminator: &NativeTerminatorKind,
    available: &BTreeSet<NativeValueId>,
    current_effect_epoch: NativeValueId,
    blocks: &BTreeMap<NativeBlockId, &NativeBlock>,
) -> NativeCodegenResult<()> {
    for value in terminator_values(terminator) {
        if !available.contains(&value) {
            return Err(NativeCodegenError::new(
                "native.ir.terminator_input",
                "terminator input is not defined in its block",
            )
            .at_function(function.id));
        }
    }
    for edge in terminator_edges(terminator) {
        let target = blocks.get(&edge.target).ok_or_else(|| {
            NativeCodegenError::new(
                "native.ir.edge_target",
                "edge target does not name a retained block",
            )
            .at_function(function.id)
        })?;
        if edge.arguments.len() != target.parameters.len() {
            return Err(NativeCodegenError::new(
                "native.ir.edge_arity",
                "edge argument count differs from target block parameter count",
            )
            .at_function(function.id));
        }
        if !available.contains(&edge.side_effect_epoch)
            || edge.side_effect_epoch != current_effect_epoch
        {
            return Err(NativeCodegenError::new(
                "native.ir.edge_effect_epoch",
                "edge must transfer the current side-effect epoch from its source block",
            )
            .at_function(function.id));
        }
        for (argument, parameter) in edge.arguments.iter().zip(&target.parameters) {
            match argument {
                NativeEdgeArgument::Value(value) if available.contains(value) => {}
                NativeEdgeArgument::LoopIteration { local }
                | NativeEdgeArgument::CaughtException { local }
                | NativeEdgeArgument::AwaitResult { local }
                    if *local == parameter.local => {}
                _ => {
                    return Err(NativeCodegenError::new(
                        "native.ir.edge_argument",
                        "edge argument is unavailable or targets the wrong local",
                    )
                    .at_function(function.id));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn terminator_edges(terminator: &NativeTerminatorKind) -> Vec<&NativeEdge> {
    match terminator {
        NativeTerminatorKind::Goto { edge } => vec![edge],
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
        } => vec![then_edge, else_edge],
        NativeTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, edge)| edge)
            .chain(std::iter::once(otherwise))
            .collect(),
        NativeTerminatorKind::Await { resume, .. } => vec![resume],
        NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable => Vec::new(),
    }
}

pub(crate) fn terminator_values(terminator: &NativeTerminatorKind) -> Vec<NativeValueId> {
    let mut values = match terminator {
        NativeTerminatorKind::Goto { .. }
        | NativeTerminatorKind::TryCatch { .. }
        | NativeTerminatorKind::Unreachable => Vec::new(),
        NativeTerminatorKind::Branch { condition, .. } => vec![*condition],
        NativeTerminatorKind::Switch {
            discriminant,
            cases,
            ..
        } => std::iter::once(*discriminant)
            .chain(cases.iter().map(|(value, _)| *value))
            .collect(),
        NativeTerminatorKind::For { iterable, .. }
        | NativeTerminatorKind::ParFor { iterable, .. } => vec![*iterable],
        NativeTerminatorKind::Spmd { header, .. } => header.clone(),
        NativeTerminatorKind::Return { values } => values.clone(),
        NativeTerminatorKind::Await { future, .. } => vec![*future],
    };
    if let NativeTerminatorKind::ParFor {
        maximum_workers: Some(maximum_workers),
        ..
    } = terminator
    {
        values.push(*maximum_workers);
    }
    for edge in terminator_edges(terminator) {
        values.extend(edge.arguments.iter().filter_map(|argument| match argument {
            NativeEdgeArgument::Value(value) => Some(*value),
            _ => None,
        }));
    }
    values
}

fn verify_region_boundaries(
    function: &NativeFunction,
    regions: &[runmat_types::RegionContract],
) -> NativeCodegenResult<()> {
    let contracts = regions
        .iter()
        .filter(|region| region.id.function == function.id)
        .map(|region| (region.id, region))
        .collect::<BTreeMap<_, _>>();
    for boundary in function
        .blocks
        .iter()
        .flat_map(|block| block.region_boundaries.iter())
    {
        let Some(contract) = contracts.get(&boundary.region) else {
            return Err(NativeCodegenError::new(
                "native.ir.region_boundary_contract",
                "region boundary has no executable-manifest contract",
            )
            .at_function(function.id));
        };
        let source = NativeSourceLocation {
            source: contract.source,
            span: contract.span,
        };
        if boundary.frame_state.source != source
            || boundary.guards.iter().any(|guard| {
                match region_guard_value(&guard.contract.condition) {
                    Some(value) => {
                        guard.value
                            != boundary
                                .frame_state
                                .locals
                                .get(value.local as usize)
                                .map(|local| local.value)
                    }
                    None => guard.value.is_some(),
                }
            })
        {
            return Err(NativeCodegenError::new(
                "native.ir.region_boundary_guard",
                "region boundary source or guard SSA binding is not exact",
            )
            .at_function(function.id));
        }
    }
    let mut actual = function
        .blocks
        .iter()
        .flat_map(|block| block.region_boundaries.iter())
        .map(|boundary| {
            (
                boundary.region,
                boundary.point,
                boundary.kind,
                boundary
                    .live_values
                    .iter()
                    .map(|binding| binding.value)
                    .collect::<Vec<_>>(),
                boundary
                    .guards
                    .iter()
                    .map(|guard| guard.contract.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let mut expected = regions
        .iter()
        .filter(|region| region.id.function == function.id)
        .flat_map(|region| {
            std::iter::once((
                region.id,
                region.entry,
                NativeRegionBoundaryKind::Entry,
                region.live_in.clone(),
                region.guards.clone(),
            ))
            .chain(region.exits.iter().copied().map(|point| {
                (
                    region.id,
                    point,
                    NativeRegionBoundaryKind::Exit,
                    region.live_out.clone(),
                    region.guards.clone(),
                )
            }))
        })
        .collect::<Vec<_>>();
    actual.sort_by_key(|item| (item.1, item.0, item.2));
    expected.sort_by_key(|item| (item.1, item.0, item.2));
    if actual != expected {
        return Err(NativeCodegenError::new(
            "native.ir.region_boundaries",
            "region boundary markers differ from the executable manifest",
        )
        .at_function(function.id));
    }
    Ok(())
}

fn region_guard_value(
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

fn ensure_sorted_unique_by<T, K: Ord + Copy>(
    field: &'static str,
    values: &[T],
    key: impl Fn(&T) -> K,
) -> NativeCodegenResult<()> {
    if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        return Err(NativeCodegenError::new(
            field,
            "entries must be sorted and unique",
        ));
    }
    Ok(())
}

fn schema_error(
    code: &'static str,
    error: runmat_types::SchemaValidationError,
) -> NativeCodegenError {
    NativeCodegenError::new(code, format!("{}: {}", error.path, error.message))
}
