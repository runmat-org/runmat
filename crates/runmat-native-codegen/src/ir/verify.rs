use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::{NativeCodegenError, NativeCodegenResult, NATIVE_IR_SCHEMA_VERSION};

#[path = "verify/capture_layout.rs"]
mod capture_layout;
#[path = "verify/super_constructor.rs"]
mod super_constructor;

impl NativeAssembly {
    pub fn verify(&self) -> NativeCodegenResult<()> {
        if self.schema_version != NATIVE_IR_SCHEMA_VERSION {
            return Err(NativeCodegenError::new(
                "native.ir.schema_version",
                format!(
                    "unsupported Native IR schema version {}; expected {}. Rebuild the program with this RunMat version",
                    self.schema_version, NATIVE_IR_SCHEMA_VERSION
                ),
            ));
        }
        if self.executable_identity.program != self.program {
            return Err(NativeCodegenError::new(
                "native.ir.program_identity",
                "assembly and executable identities name different programs",
            ));
        }
        if self.executable_identity.entrypoint_function != Some(self.selected_portable_entrypoint)
            || self.executable_identity.entrypoint_kind
                == runmat_execution::ExecutableEntrypointKind::DeclarationOnly
        {
            return Err(NativeCodegenError::new(
                "native.ir.entrypoint_binding",
                "selected portable entrypoint differs from the executable identity",
            ));
        }
        self.target.validate()?;
        if self.native_cache_key
            != self.target.cache_key(
                &self.executable_cache_key,
                self.selected_entrypoint,
                &self.selected_entrypoint_body,
            )?
        {
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
        ensure_sorted_unique_by("native.ir.entrypoints", &self.entrypoints, |entrypoint| {
            *entrypoint
        })?;
        if self.entrypoints.is_empty()
            || self
                .entrypoints
                .iter()
                .any(|entrypoint| !functions.contains_key(entrypoint))
            || !self.entrypoints.contains(&self.selected_entrypoint)
        {
            return Err(NativeCodegenError::new(
                "native.ir.entrypoints",
                "Native IR entrypoints must exist and contain the selected compiled entrypoint",
            ));
        }
        if functions
            .get(&self.selected_entrypoint)
            .is_none_or(|function| {
                function.mir_body_digest != self.selected_entrypoint_body
                    || function.portable_id != self.selected_portable_entrypoint
            })
        {
            return Err(NativeCodegenError::new(
                "native.ir.entrypoint_body",
                "selected entrypoint body differs from its Native IR function identity",
            ));
        }
        let portable_functions = self
            .functions
            .iter()
            .map(|function| function.portable_id)
            .collect::<std::collections::BTreeSet<_>>();
        if portable_functions.len() != self.functions.len() {
            return Err(NativeCodegenError::new(
                "native.ir.function_coordinates",
                "Native IR functions repeat a portable program identity",
            ));
        }
        for function in &self.functions {
            verify_function(
                function,
                &self.requirements.regions,
                &self.requirements.capabilities,
            )?;
            for binding in &function.inherited_storage_bindings {
                let owner = functions.get(&binding.owner).ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.ir.inherited_storage",
                        "canonical storage owner is absent from the admitted program",
                    )
                    .at_function(function.id)
                })?;
                let mut cursor = function.lexical_parent;
                let mut visible = false;
                for _ in 0..self.functions.len() {
                    let Some(parent) = cursor else { break };
                    let ancestor = functions.get(&parent).ok_or_else(|| {
                        NativeCodegenError::new(
                            "native.ir.inherited_storage",
                            "canonical storage ancestry is incomplete",
                        )
                        .at_function(function.id)
                    })?;
                    if ancestor.source != function.source {
                        return Err(NativeCodegenError::new(
                            "native.ir.inherited_storage",
                            "canonical storage ancestry crosses a source owner",
                        )
                        .at_function(function.id));
                    }
                    if parent == binding.owner {
                        visible = true;
                        break;
                    }
                    cursor = ancestor.lexical_parent;
                }
                if !visible
                    || owner.source != function.source
                    || owner.name != binding.owner_function
                {
                    return Err(NativeCodegenError::new(
                        "native.ir.inherited_storage",
                        "canonical storage owner is outside the child's lexical ancestry",
                    )
                    .at_function(function.id));
                }
                let expected_effect = match binding.kind {
                    NativeInheritedStorageKind::Global => {
                        runmat_types::WorkspaceEffect::MutatesGlobal
                    }
                    NativeInheritedStorageKind::Persistent => {
                        runmat_types::WorkspaceEffect::MutatesPersistent
                    }
                };
                let declared = owner.locals.iter().any(|local| {
                    local.binding == Some(binding.binding)
                        && local.name.as_deref() == Some(binding.name.as_str())
                        && owner.blocks.iter().flat_map(|block| &block.instructions).any(|instruction| {
                            matches!(&instruction.operation,
                                NativeOperation::Statement(runmat_mir::MirStmtKind::WorkspaceEffect { effect, bindings })
                                    if *effect == expected_effect
                                        && bindings.contains(&runmat_mir::MirLocalId(local.id.0 as usize)))
                        })
                });
                if !declared {
                    return Err(NativeCodegenError::new(
                        "native.ir.inherited_storage",
                        "canonical storage declaration is absent from its owner",
                    )
                    .at_function(function.id));
                }
            }
        }
        super_constructor::verify(self, &functions)?;
        Ok(())
    }
}

fn verify_function(
    function: &NativeFunction,
    regions: &[runmat_types::RegionContract],
    capabilities: &runmat_types::CapabilitySet,
) -> NativeCodegenResult<()> {
    let error = |code, message| NativeCodegenError::new(code, message).at_function(function.id);
    if function
        .operation_contracts
        .windows(2)
        .any(|pair| pair[0].site >= pair[1].site)
        || function
            .operation_contracts
            .iter()
            .any(|contract| contract.site.point.function != function.portable_id)
    {
        return Err(error(
            "native.ir.operation_contracts",
            "operation contracts must be sorted, unique, and owned by this function",
        ));
    }
    if function.blocks.is_empty() {
        return Err(error("native.ir.blocks", "function must contain a block"));
    }
    if NativeExpressionContinuationGeometry::for_blocks(&function.blocks)
        != Some(function.expression_continuation_geometry)
    {
        return Err(error(
            "native.ir.expression_continuation_geometry",
            "expression continuation backing does not match canonical MIR site geometry",
        ));
    }
    if !function.capabilities.0.is_subset(&capabilities.0) {
        return Err(error(
            "native.ir.function_capabilities",
            "function requirements exceed the executable capability envelope",
        ));
    }
    if function.locals.len() > u32::MAX as usize
        || function.locals.iter().enumerate().any(|(index, local)| {
            local.id.0 as usize != index
                || local.binding.is_some() != local.name.is_some()
                || local.name.as_ref().is_some_and(|name| name.is_empty())
        })
    {
        return Err(error(
            "native.ir.function_locals",
            "locals must be ordered and bound locals must retain non-empty canonical names",
        ));
    }
    ensure_sorted_unique_by(
        "native.ir.lexical_bindings",
        &function.lexical_bindings,
        |binding| binding.binding,
    )
    .map_err(|error| error.at_function(function.id))?;
    if has_duplicates(
        &function
            .lexical_bindings
            .iter()
            .map(|binding| binding.local)
            .collect::<Vec<_>>(),
    ) || function.lexical_bindings.iter().any(|binding| {
        function
            .local(binding.local)
            .is_none_or(|local| local.binding != Some(binding.binding))
            || (binding.kind == NativeLexicalBindingKind::Owned
                && !function.creates_lexical_environment)
            || (binding.kind == NativeLexicalBindingKind::Transit
                && function.source_role != runmat_types::FunctionSourceRole::Anonymous)
            || (binding.kind == NativeLexicalBindingKind::Inherited
                && function.source_role == runmat_types::FunctionSourceRole::Anonymous)
    }) {
        return Err(error(
            "native.ir.lexical_bindings",
            "shared lexical bindings must name unique owned locals and activations",
        ));
    }
    ensure_sorted_unique_by(
        "native.ir.inherited_storage",
        &function.inherited_storage_bindings,
        |binding| binding.binding,
    )
    .map_err(|error| error.at_function(function.id))?;
    if function.inherited_storage_bindings.iter().any(|binding| {
        function.local(binding.local).is_none_or(|local| {
            local.binding != Some(binding.binding)
                || local.name.as_deref() != Some(binding.name.as_str())
        }) || binding.name.is_empty()
            || binding.owner_function.is_empty()
            || function.source_role != runmat_types::FunctionSourceRole::Nested
            || function
                .lexical_bindings
                .iter()
                .any(|shared| shared.binding == binding.binding)
    }) {
        return Err(error(
            "native.ir.inherited_storage",
            "inherited canonical storage must name a nested capture and its owner",
        ));
    }
    capture_layout::verify(function)?;
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
        if local.0 as usize >= function.local_count() {
            return Err(error(
                "native.ir.function_abi",
                "function ABI names a local outside the function",
            ));
        }
    }
    if has_duplicates(&function.abi.fixed_inputs)
        || has_duplicates(&function.abi.fixed_outputs)
        || function
            .abi
            .varargin
            .is_some_and(|local| function.abi.fixed_inputs.contains(&local))
        || function
            .abi
            .varargout
            .is_some_and(|local| function.abi.fixed_outputs.contains(&local))
    {
        return Err(error(
            "native.ir.function_abi",
            "fixed function ABI locals must be unique and exclude variadic locals",
        ));
    }
    ensure_sorted_unique_by(
        "native.ir.argument_validations",
        &function.argument_validations,
        |validation| validation.input,
    )
    .map_err(|error| error.at_function(function.id))?;
    for validation in &function.argument_validations {
        if !function.abi.fixed_inputs.contains(&validation.input)
            || validation
                .class_name
                .as_ref()
                .is_some_and(|name| name.is_empty())
            || validation
                .validators
                .iter()
                .any(|validator| !argument_validator_is_valid(validator))
        {
            return Err(error(
                "native.ir.argument_validations",
                "argument validations must be canonical, finite, and attached to fixed inputs",
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
    let catch_scope_declarations: BTreeMap<_, _> = function
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                &block.terminator.kind,
                NativeTerminatorKind::TryCatch { .. }
            )
        })
        .map(|block| (runmat_mir::CatchScopeId(block.id.0 as usize), block))
        .collect();
    let mut all_instructions = BTreeSet::new();
    let mut all_safepoints = BTreeSet::new();
    let mut actual_sites = BTreeSet::new();
    for block in &function.blocks {
        let mut seen_scopes = BTreeSet::new();
        if block.catch_scopes.iter().enumerate().any(|(depth, scope)| {
            !seen_scopes.insert(*scope)
                || catch_scope_declarations
                    .get(scope)
                    .is_none_or(|declaration| {
                        declaration.catch_scopes.as_slice() != &block.catch_scopes[..depth]
                    })
        }) {
            return Err(error(
                "native.ir.catch_scope",
                "block catch membership names a missing, repeated, or misparented lexical scope",
            ));
        }
        if let NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } = &block.terminator.kind
        {
            let expected = runmat_mir::CatchScopeId(block.id.0 as usize);
            let mut catch_ancestry = block.catch_scopes.clone();
            catch_ancestry.push(expected);
            if blocks
                .get(&try_edge.target)
                .is_none_or(|candidate| candidate.catch_scopes != block.catch_scopes)
                || blocks
                    .get(&catch_edge.target)
                    .is_none_or(|candidate| candidate.catch_scopes != catch_ancestry)
            {
                return Err(error(
                    "native.ir.catch_scope",
                    "try and catch targets do not carry the declared MIR lexical ancestry",
                ));
            }
        }
        if block.instructions.iter().any(|instruction| {
            instruction.site.point.block != block.id.0
                || instruction.site.point.function != function.portable_id
                || instruction.source.source != function.source
        }) || block.terminator.site.point.block != block.id.0
            || block.terminator.site.point.function != function.portable_id
            || block.terminator.source.source != function.source
            || block.terminator.source.span.start > block.terminator.source.span.end
        {
            return Err(error(
                "native.ir.site_block",
                "sites must name their containing function, block, and source",
            ));
        }
        if block.parameters.len() != function.local_count()
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
        // Validate the transient sequence channel before per-instruction output
        // reconstruction. A missing consumer also removes the statement that
        // gives its producer an output role; the channel invariant is the
        // primary structural error in that hostile shape.
        verify_sequence_register_flow(function.id, &block.instructions)?;
        for instruction in &block.instructions {
            let containing_statement = block.instructions.iter().find_map(|candidate| {
                (candidate.site.point == instruction.site.point
                    && candidate.site.phase == NativeSitePhase::Statement)
                    .then_some(&candidate.operation)
                    .and_then(|operation| match operation {
                        NativeOperation::Statement(statement) => Some(statement),
                        NativeOperation::Rvalue { .. } => None,
                    })
            });
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
                containing_statement,
            )?;
            if !instruction
                .capabilities
                .0
                .is_subset(&function.capabilities.0)
            {
                return Err(error(
                    "native.ir.capabilities",
                    "instruction requirement is absent from the function capability set",
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
        let terminator_contract = function
            .operation_contracts
            .binary_search_by(|contract| contract.site.cmp(&block.terminator.site))
            .ok()
            .map(|index| &function.operation_contracts[index])
            .ok_or_else(|| {
                error(
                    "native.ir.operation_contract",
                    "terminator has no bound analysis operation contract",
                )
            })?;
        if block.terminator.effects != terminator_contract.effects
            || block.terminator.capabilities != terminator_contract.capabilities
        {
            return Err(error(
                "native.ir.operation_contract",
                "terminator requirements differ from its bound analysis contract",
            ));
        }
        if !block
            .terminator
            .capabilities
            .0
            .is_subset(&function.capabilities.0)
        {
            return Err(error(
                "native.ir.capabilities",
                "terminator requirement is absent from the function capability set",
            ));
        }
        if block.terminator.class
            != runmat_mir::effective_native_lowering_class(
                block.terminator.site.construct,
                &block.terminator.effects,
            )
        {
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
    verify_sequence_capture_flow(function, &blocks)?;
    verify_fixed_assignment_flow(function, &blocks)?;

    let contracted = function
        .operation_contracts
        .iter()
        .map(|contract| contract.site.clone())
        .collect::<BTreeSet<_>>();
    if contracted.len() != function.operation_contracts.len() || contracted != actual_sites {
        return Err(error(
            "native.ir.operation_contract_coverage",
            "operation contracts do not exactly cover lowered MIR sites",
        ));
    }
    if function
        .operation_contracts
        .iter()
        .map(|contract| &contract.site)
        .ne(function.expected_sites.iter())
    {
        return Err(error(
            "native.ir.expected_sites",
            "resume-site projection differs from bound operation contracts",
        ));
    }
    verify_region_boundaries(function, regions)?;
    Ok(())
}

fn verify_fixed_assignment_flow(
    function: &NativeFunction,
    blocks: &BTreeMap<NativeBlockId, &NativeBlock>,
) -> NativeCodegenResult<()> {
    let error = |message: &str| {
        NativeCodegenError::new("native.ir.prepared_assignment_owner", message)
            .at_function(function.id)
    };
    let mut producers = BTreeMap::new();
    let mut consumers = BTreeMap::new();
    let mut phase_contracts = Vec::new();
    let scalar_receipts = function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| match &instruction.operation {
            NativeOperation::Statement(runmat_mir::MirStmtKind::PrepareFixedAssignment(phase)) => {
                match phase.output() {
                    Some(runmat_mir::MirFixedSelectorDestination::Scalar(local)) => Some(local),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for block in &function.blocks {
        for instruction in &block.instructions {
            if !matches!(
                instruction.operation,
                NativeOperation::Statement(runmat_mir::MirStmtKind::PrepareFixedAssignment(_))
            ) && instruction
                .outputs
                .iter()
                .filter_map(|output| output.local)
                .any(|local| scalar_receipts.contains(&runmat_mir::MirLocalId(local.0 as usize)))
            {
                return Err(error(
                    "ordinary Native work overwrites a reserved fixed selector receipt",
                ));
            }
        }
        if matches!(&block.terminator.kind, NativeTerminatorKind::Await { result: Some(runmat_mir::MirPlace::Local(local)), .. } if scalar_receipts.contains(local))
        {
            return Err(error(
                "Native Await overwrites a reserved fixed selector receipt",
            ));
        }
    }
    for instruction in function.blocks.iter().flat_map(|block| &block.instructions) {
        match &instruction.operation {
            NativeOperation::Statement(runmat_mir::MirStmtKind::PrepareFixedAssignment(owner)) => {
                owner
                    .validate()
                    .map_err(|message| error(message.as_str()))?;
                if let Some(runmat_mir::MirFixedSelectorDestination::Scalar(local)) = owner.output()
                {
                    if !function.locals.iter().any(|metadata| {
                        Some(local.0) == usize::try_from(metadata.id.0).ok()
                            && metadata.kind == super::NativeLocalKind::Temporary
                            && metadata.binding.is_none()
                    }) {
                        return Err(error(
                            "fixed selector output is not a private temporary frame local",
                        ));
                    }
                }
                phase_contracts.push((&owner.assignment, &instruction.source));
                if matches!(
                    owner.phase,
                    runmat_mir::MirFixedPreparationPhase::CaptureReceiver
                ) && producers
                    .insert(
                        owner.assignment.source_statement,
                        (&owner.assignment, &instruction.source),
                    )
                    .is_some()
                {
                    return Err(error(
                        "fixed source occurrence has multiple preparation instructions",
                    ));
                }
            }
            NativeOperation::Statement(runmat_mir::MirStmtKind::PreparedAssign {
                preparation,
                ..
            }) => {
                if consumers
                    .insert(
                        preparation.source_statement,
                        (preparation, &instruction.source),
                    )
                    .is_some()
                {
                    return Err(error(
                        "fixed source occurrence has multiple consumer instructions",
                    ));
                }
            }
            NativeOperation::Statement(runmat_mir::MirStmtKind::Assign { place, .. })
                if place.requires_prepared_assignment() =>
            {
                return Err(error(
                    "expanding fixed assignment lacks its typed preparation/consumer roles",
                ));
            }
            _ => {}
        }
    }
    if producers != consumers
        || phase_contracts.iter().any(|&(assignment, source)| {
            consumers.get(&assignment.source_statement) != Some(&(assignment, source))
        })
    {
        return Err(error(
            "fixed preparation and consumer original source contracts disagree",
        ));
    }
    let mut entries =
        BTreeMap::<NativeBlockId, Option<runmat_mir::MirFixedPreparationState<'_>>>::new();
    let mut pending = std::collections::VecDeque::from([(function.entry, None)]);
    while let Some((id, mut live)) = pending.pop_front() {
        if let Some(prior) = entries.get(&id) {
            live = runmat_mir::MirFixedPreparationState::join(prior.clone(), live)
                .map_err(|message| error(&message))?;
            if *prior == live {
                continue;
            }
        }
        entries.insert(id, live.clone());
        let block = blocks
            .get(&id)
            .ok_or_else(|| error("fixed destination lifetime reaches a missing Native block"))?;
        for instruction in &block.instructions {
            match &instruction.operation {
                NativeOperation::Statement(runmat_mir::MirStmtKind::PrepareFixedAssignment(
                    owner,
                )) => {
                    live = Some(
                        runmat_mir::MirFixedPreparationState::advance(
                            live.take(),
                            owner,
                            &scalar_receipts,
                        )
                        .map_err(|message| error(&message))?,
                    );
                }
                NativeOperation::Statement(runmat_mir::MirStmtKind::PreparedAssign {
                    preparation,
                    ..
                }) => {
                    live.take()
                        .ok_or_else(|| error("fixed consumer lacks a live source occurrence"))?
                        .consume(preparation)
                        .map_err(|message| error(&message))?;
                }
                NativeOperation::Statement(
                    runmat_mir::MirStmtKind::MultiAssign { .. }
                    | runmat_mir::MirStmtKind::SequenceAssign { .. },
                ) if live.is_some() => {
                    return Err(error(
                        "another prepared output owner overlaps a live fixed destination",
                    ));
                }
                _ if live.as_ref().is_some_and(|state| !state.is_ready()) => {
                    return Err(error(
                        "ordinary Native work executes before the fixed destination is ready",
                    ));
                }
                _ => {}
            }
        }
        if let NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } = &block.terminator.kind
        {
            pending.push_back((try_edge.target, live));
            pending.push_back((catch_edge.target, None));
        } else {
            let edges = terminator_edges(&block.terminator.kind);
            if edges.is_empty() && live.is_some() {
                return Err(error(
                    "fixed destination remains live at a Native function boundary",
                ));
            }
            pending.extend(edges.into_iter().map(|edge| (edge.target, live.clone())));
        }
    }
    Ok(())
}

fn verify_sequence_capture_flow(
    function: &NativeFunction,
    blocks: &BTreeMap<NativeBlockId, &NativeBlock>,
) -> NativeCodegenResult<()> {
    let error = |message| {
        NativeCodegenError::new("native.ir.sequence_capture_flow", message).at_function(function.id)
    };
    let mut entries = BTreeMap::<NativeBlockId, BTreeSet<runmat_mir::MirSequenceLocalId>>::new();
    let mut pending = std::collections::VecDeque::from([(function.entry, BTreeSet::new())]);
    while let Some((block_id, incoming)) = pending.pop_front() {
        if let Some(existing) = entries.get(&block_id) {
            if existing != &incoming {
                return Err(error(format!(
                    "captured sequence state disagrees at block {}: {existing:?} versus {incoming:?}",
                    block_id.0
                )));
            }
            continue;
        }
        entries.insert(block_id, incoming.clone());
        let block = blocks.get(&block_id).ok_or_else(|| {
            error(format!(
                "captured sequence flow reaches missing block {}",
                block_id.0
            ))
        })?;
        let mut live = incoming;
        for instruction in &block.instructions {
            match &instruction.operation {
                NativeOperation::Statement(runmat_mir::MirStmtKind::CaptureSequence {
                    destination,
                    ..
                }) => {
                    if !live.insert(*destination) {
                        return Err(error(format!(
                            "captured sequence local {} is overwritten at instruction {}",
                            destination.0, instruction.id.0
                        )));
                    }
                }
                NativeOperation::Statement(runmat_mir::MirStmtKind::PrepareFixedAssignment(
                    phase,
                )) => {
                    let mut consumed = BTreeSet::new();
                    let mut result = Ok(());
                    phase.visit_sequence_locals(|sequence| {
                        if result.is_ok() {
                            result = consume_sequence_local(*sequence, &mut live, &mut consumed);
                        }
                    });
                    result.map_err(error)?;
                    if let Some(runmat_mir::MirFixedSelectorDestination::Sequence(destination)) =
                        phase.output()
                    {
                        if !live.insert(destination) {
                            return Err(error(format!(
                                "fixed selector overwrites captured sequence {}",
                                destination.0
                            )));
                        }
                    }
                }
                NativeOperation::Rvalue { value, .. } => {
                    consume_rvalue_sequence_locals(value, &mut live).map_err(|message| {
                        error(format!("instruction {}: {message}", instruction.id.0))
                    })?;
                }
                NativeOperation::Statement(_) => {}
            }
        }
        match &block.terminator.kind {
            NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable
                if !live.is_empty() =>
            {
                return Err(error(format!(
                    "captured sequence locals {live:?} remain live at terminator in block {}",
                    block_id.0
                )));
            }
            NativeTerminatorKind::TryCatch {
                try_edge,
                catch_edge,
                ..
            } => {
                pending.push_back((try_edge.target, live));
                pending.push_back((catch_edge.target, BTreeSet::new()));
                continue;
            }
            _ => {}
        }
        for edge in terminator_edges(&block.terminator.kind) {
            pending.push_back((edge.target, live.clone()));
        }
    }
    Ok(())
}

fn consume_rvalue_sequence_locals(
    value: &runmat_mir::MirRvalue,
    live: &mut BTreeSet<runmat_mir::MirSequenceLocalId>,
) -> Result<(), String> {
    let mut consumed = BTreeSet::new();
    match value {
        runmat_mir::MirRvalue::Call(call) => {
            for sequence in call.args.iter().filter_map(|argument| match argument {
                runmat_mir::MirCallArg::CapturedSequence(sequence) => Some(*sequence),
                _ => None,
            }) {
                consume_sequence_local(sequence, live, &mut consumed)?;
            }
        }
        runmat_mir::MirRvalue::Aggregate { elements, .. } => {
            for sequence in elements.iter().filter_map(|element| match element {
                runmat_mir::MirAggregateElement::CapturedSequence(sequence) => Some(*sequence),
                runmat_mir::MirAggregateElement::Single(_) => None,
            }) {
                consume_sequence_local(sequence, live, &mut consumed)?;
            }
        }
        runmat_mir::MirRvalue::ShortCircuit { right_temps, .. } => {
            for statement in right_temps {
                match &statement.kind {
                    runmat_mir::MirStmtKind::CaptureSequence { destination, .. } => {
                        if !live.insert(*destination) {
                            return Err(format!(
                                "captured sequence local {} is overwritten in a short-circuit expression",
                                destination.0
                            ));
                        }
                    }
                    runmat_mir::MirStmtKind::Assign { value, .. }
                    | runmat_mir::MirStmtKind::MultiAssign { value, .. }
                    | runmat_mir::MirStmtKind::SequenceAssign { value, .. }
                    | runmat_mir::MirStmtKind::Expr(value) => {
                        consume_rvalue_sequence_locals(value, live)?;
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn consume_sequence_local(
    sequence: runmat_mir::MirSequenceLocalId,
    live: &mut BTreeSet<runmat_mir::MirSequenceLocalId>,
    consumed: &mut BTreeSet<runmat_mir::MirSequenceLocalId>,
) -> Result<(), String> {
    if !consumed.insert(sequence) {
        return Err(format!(
            "captured sequence local {} is consumed more than once by one operation",
            sequence.0
        ));
    }
    if !live.remove(&sequence) {
        return Err(format!(
            "captured sequence local {} is consumed without a live capture",
            sequence.0
        ));
    }
    Ok(())
}

fn verify_sequence_register_flow(
    function: runmat_types::FunctionId,
    instructions: &[NativeInstruction],
) -> NativeCodegenResult<()> {
    let error = |message| {
        NativeCodegenError::new("native.ir.sequence_register", message).at_function(function)
    };
    let mut producer: Option<(usize, &runmat_mir::MirSequenceTarget)> = None;
    for (index, instruction) in instructions.iter().enumerate() {
        match &instruction.operation {
            NativeOperation::Rvalue {
                result: NativeRvalueResult::SequenceAssignment(target),
                ..
            } => {
                if producer.is_some() {
                    return Err(error(
                        "comma-separated sequence register is overwritten before consumption",
                    ));
                }
                producer = Some((index, target));
            }
            NativeOperation::Statement(runmat_mir::MirStmtKind::SequenceAssign {
                target, ..
            }) => {
                let Some((producer_index, produced_target)) = producer.take() else {
                    return Err(error(
                        "comma-separated sequence register is consumed without a producer",
                    ));
                };
                if producer_index + 1 != index
                    || produced_target != target
                    || instructions[producer_index].site.point != instruction.site.point
                {
                    return Err(error(
                        "comma-separated sequence producer and consumer must be adjacent and describe the same MIR statement",
                    ));
                }
            }
            _ if producer.is_some() => {
                return Err(error(
                    "comma-separated sequence register crosses an instruction boundary before consumption",
                ));
            }
            _ => {}
        }
    }
    if producer.is_some() {
        return Err(error(
            "comma-separated sequence register remains live at a block boundary",
        ));
    }
    Ok(())
}

fn has_duplicates<T: Ord + Copy>(values: &[T]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(*value))
}

fn argument_validator_is_valid(validator: &runmat_types::FunctionArgValidator) -> bool {
    use runmat_types::{FunctionArgValidationLiteral, FunctionArgValidator};

    match validator {
        FunctionArgValidator::A(names) | FunctionArgValidator::UnderlyingType(names) => {
            !names.is_empty() && names.iter().all(|name| !name.is_empty())
        }
        FunctionArgValidator::Member(values) => values.iter().all(|value| match value {
            FunctionArgValidationLiteral::Number(value) => value.is_finite(),
            FunctionArgValidationLiteral::Integer(_) => true,
            FunctionArgValidationLiteral::Text(_) | FunctionArgValidationLiteral::Bool(_) => true,
        }),
        FunctionArgValidator::InRange(lower, upper, _) => {
            lower.is_finite() && upper.is_finite() && lower <= upper
        }
        FunctionArgValidator::GreaterThanOrEqual(value)
        | FunctionArgValidator::LessThanOrEqual(value)
        | FunctionArgValidator::GreaterThan(value)
        | FunctionArgValidator::LessThan(value) => value.is_finite(),
        _ => true,
    }
}

fn verify_instruction(
    function: &NativeFunction,
    instruction: &NativeInstruction,
    available: &BTreeSet<NativeValueId>,
    current_effect_epoch: NativeValueId,
    all_safepoints: &mut BTreeSet<NativeSafepointId>,
    containing_statement: Option<&runmat_mir::MirStmtKind>,
) -> NativeCodegenResult<()> {
    let error = |code, message| {
        NativeCodegenError::new(code, message)
            .at_point(instruction.site.point)
            .for_construct(instruction.site.construct)
    };
    let contract = function
        .operation_contracts
        .binary_search_by(|contract| contract.site.cmp(&instruction.site))
        .ok()
        .map(|index| &function.operation_contracts[index])
        .ok_or_else(|| {
            error(
                "native.ir.operation_contract",
                "instruction has no bound analysis operation contract",
            )
        })?;
    if instruction.effects != contract.effects || instruction.capabilities != contract.capabilities
    {
        return Err(error(
            "native.ir.operation_contract",
            "instruction requirements differ from its bound analysis contract",
        ));
    }
    if instruction.site.point.function != function.portable_id {
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
    if instruction.class
        != runmat_mir::effective_native_lowering_class(
            instruction.site.construct,
            &instruction.effects,
        )
    {
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
        if !rvalue_selector_arity_valid(value) {
            return Err(error(
                "native.capability.selector_dimension_limit",
                "index selector count exceeds the shared 32-dimension mask contract",
            ));
        }
        let mut regions_valid = true;
        value.visit_expression_regions(|region| regions_valid &= region.validate().is_ok());
        if !regions_valid {
            return Err(error(
                "native.ir.expression_region",
                "embedded contextual or prepared expression region is invalid",
            ));
        }
        if rvalue_contains_legacy_binding_place(value) {
            return Err(error(
                "native.ir.legacy_binding_place",
                "legacy MIR binding places must be canonicalized to locals before Native IR",
            ));
        }
        if let NativeOperation::Rvalue {
            result: NativeRvalueResult::MultiAssignment(targets),
            ..
        } = &instruction.operation
        {
            let Some(runmat_mir::MirStmtKind::MultiAssign {
                targets: statement_targets,
                ..
            }) = containing_statement
            else {
                return Err(error(
                    "native.ir.multi_assignment_owner",
                    "multi-assignment rvalue is not owned by a multi-assignment statement",
                ));
            };
            if targets != statement_targets {
                return Err(error(
                    "native.ir.multi_assignment_owner",
                    "multi-assignment rvalue targets differ from its owning statement",
                ));
            }
        }
    } else if let NativeOperation::Statement(statement) = &instruction.operation {
        let expected_constructs = runmat_mir::statement_construct_inventory(statement)
            .into_iter()
            .skip(1)
            .collect::<Vec<_>>();
        if instruction.embedded_constructs != expected_constructs {
            return Err(error(
                "native.ir.embedded_constructs",
                "statement embedded-construct inventory disagrees with canonical MIR",
            ));
        }
        if runmat_mir::statement_construct_inventory(statement)
            .iter()
            .any(|construct| {
                construct.native_lowering_class()
                    == runmat_mir::NativeLoweringClass::CapabilityRejection
            })
        {
            return Err(error(
                "native.capability.distributed_core_pending",
                "statement contains a construct requiring the R25 distributed core",
            ));
        }
        if !statement_selector_arity_valid(statement) {
            return Err(error(
                "native.capability.selector_dimension_limit",
                "index selector count exceeds the shared 32-dimension mask contract",
            ));
        }
        let mut regions_valid = true;
        statement.visit_expression_regions(|region| regions_valid &= region.validate().is_ok());
        if !regions_valid {
            return Err(error(
                "native.ir.expression_region",
                "embedded contextual or prepared expression region is invalid",
            ));
        }
        if statement_contains_legacy_binding_place(statement) {
            return Err(error(
                "native.ir.legacy_binding_place",
                "legacy MIR binding places must be canonicalized to locals before Native IR",
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
    if instruction.outputs.iter().any(|output| {
        output
            .local
            .is_some_and(|local| local.0 as usize >= function.local_count())
    }) {
        return Err(error(
            "native.ir.output_local",
            "instruction output names a local outside the function",
        ));
    }
    let expected_output_locals = expected_output_locals(instruction, containing_statement)?;
    if instruction
        .outputs
        .iter()
        .map(|output| output.local)
        .ne(expected_output_locals)
    {
        return Err(error(
            "native.ir.output_local_identity",
            "instruction output locals differ from the retained canonical statement role",
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

fn rvalue_selector_arity_valid(value: &runmat_mir::MirRvalue) -> bool {
    let mut nested_valid = true;
    value.visit_direct_expression_steps(|steps| {
        nested_valid &= steps.iter().all(|step| match step {
            runmat_mir::MirExpressionStep::Let { value, .. } => rvalue_selector_arity_valid(value),
            runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                expansion_source_selector_arity_valid(source)
            }
        });
    });
    nested_valid && rvalue_selector_arity_valid_without_regions(value)
}

fn rvalue_selector_arity_valid_without_regions(value: &runmat_mir::MirRvalue) -> bool {
    match value {
        runmat_mir::MirRvalue::Index { indexing, .. } => {
            indexing.components.len() <= u32::BITS as usize
        }
        runmat_mir::MirRvalue::ShortCircuit { right_temps, .. } => {
            right_temps.iter().all(|statement| {
                !matches!(
                    &statement.kind,
                    runmat_mir::MirStmtKind::PrepareFixedAssignment(_)
                        | runmat_mir::MirStmtKind::PreparedAssign { .. }
                ) && statement_selector_arity_valid(&statement.kind)
            })
        }
        _ => true,
    }
}

fn rvalue_contains_legacy_binding_place(value: &runmat_mir::MirRvalue) -> bool {
    let mut nested_legacy = false;
    value.visit_direct_expression_steps(|steps| {
        nested_legacy |= steps.iter().any(|step| match step {
            runmat_mir::MirExpressionStep::Let { value, .. } => {
                rvalue_contains_legacy_binding_place(value)
            }
            runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                expansion_source_contains_legacy_binding_place(source)
            }
        });
    });
    nested_legacy || rvalue_contains_legacy_binding_place_without_regions(value)
}

fn rvalue_contains_legacy_binding_place_without_regions(value: &runmat_mir::MirRvalue) -> bool {
    matches!(value, runmat_mir::MirRvalue::ShortCircuit { right_temps, .. }
        if right_temps.iter().any(|statement| statement_contains_legacy_binding_place(&statement.kind)))
}

fn expansion_source_selector_arity_valid(source: &runmat_mir::MirExpansionSource) -> bool {
    let own_valid = expansion_source_selector_arity_valid_without_regions(source);
    let mut nested_valid = true;
    source.visit_direct_expression_steps(|steps| {
        nested_valid &= steps.iter().all(|step| match step {
            runmat_mir::MirExpressionStep::Let { value, .. } => rvalue_selector_arity_valid(value),
            runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                expansion_source_selector_arity_valid(source)
            }
        });
    });
    own_valid && nested_valid
}

fn expansion_source_selector_arity_valid_without_regions(
    source: &runmat_mir::MirExpansionSource,
) -> bool {
    match source {
        runmat_mir::MirExpansionSource::SubscriptChain(chain) => chain
            .steps
            .iter()
            .filter_map(|step| match step {
                runmat_mir::MirSubscriptStep::Index(indexing) => Some(indexing),
                runmat_mir::MirSubscriptStep::DottedInvoke { .. } => None,
                runmat_mir::MirSubscriptStep::Member(_)
                | runmat_mir::MirSubscriptStep::DynamicMember(_)
                | runmat_mir::MirSubscriptStep::PreparedDynamicMember(_) => None,
            })
            .all(|indexing| indexing.components.len() <= u32::BITS as usize),
        runmat_mir::MirExpansionSource::CellContents { indexing, .. } => {
            indexing.components.len() <= u32::BITS as usize
        }
        runmat_mir::MirExpansionSource::ReturnedOutputs(_)
        | runmat_mir::MirExpansionSource::Member { .. }
        | runmat_mir::MirExpansionSource::DynamicMember { .. } => true,
    }
}

fn expansion_source_contains_legacy_binding_place(source: &runmat_mir::MirExpansionSource) -> bool {
    let mut contains = false;
    source.visit_direct_expression_steps(|steps| {
        contains |= steps.iter().any(|step| match step {
            runmat_mir::MirExpressionStep::Let { value, .. } => {
                rvalue_contains_legacy_binding_place(value)
            }
            runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                expansion_source_contains_legacy_binding_place(source)
            }
        });
    });
    contains || expansion_source_contains_legacy_binding_place_without_regions(source)
}

fn expansion_source_contains_legacy_binding_place_without_regions(
    _source: &runmat_mir::MirExpansionSource,
) -> bool {
    false
}

fn statement_selector_arity_valid(statement: &runmat_mir::MirStmtKind) -> bool {
    match statement {
        runmat_mir::MirStmtKind::PrepareFixedAssignment(owner) => {
            owner.validate().is_ok()
                && place_selector_arity_valid(&owner.assignment.mutation.place)
                && owner
                    .selector_region()
                    .is_none_or(|region| match &region.argument {
                        runmat_mir::MirCallArg::Expansion(source) => {
                            expansion_source_selector_arity_valid(source)
                        }
                        _ => true,
                    })
                && owner.steps().iter().all(|step| match step {
                    runmat_mir::MirExpressionStep::Let { value, .. } => {
                        rvalue_selector_arity_valid(value)
                    }
                    runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                        expansion_source_selector_arity_valid(source)
                    }
                })
        }
        runmat_mir::MirStmtKind::PreparedAssign { preparation, value } => {
            preparation.validate_phased().is_ok() && rvalue_selector_arity_valid(value)
        }
        runmat_mir::MirStmtKind::Assign { place, value } => {
            place_selector_arity_valid(place) && rvalue_selector_arity_valid(value)
        }
        runmat_mir::MirStmtKind::MultiAssign { targets, value } => {
            targets.targets.iter().all(|target| match target {
                runmat_mir::MirOutputTarget::Place(place) => place_selector_arity_valid(place),
                runmat_mir::MirOutputTarget::Sequence(target) => {
                    place_selector_arity_valid(target.base())
                        && sequence_target_selector_arity_valid(target)
                }
                runmat_mir::MirOutputTarget::Discard => true,
            }) && rvalue_selector_arity_valid(value)
        }
        runmat_mir::MirStmtKind::SequenceAssign { target, value } => {
            let base_valid = match target {
                runmat_mir::MirSequenceTarget::Member { base, .. }
                | runmat_mir::MirSequenceTarget::DynamicMember { base, .. }
                | runmat_mir::MirSequenceTarget::CellContents { base, .. } => {
                    place_selector_arity_valid(base)
                }
            };
            base_valid && rvalue_selector_arity_valid(value)
        }
        runmat_mir::MirStmtKind::CaptureSequence { source, .. } => {
            expansion_source_selector_arity_valid(source)
        }
        runmat_mir::MirStmtKind::Expr(value) => rvalue_selector_arity_valid(value),
        runmat_mir::MirStmtKind::PlaceMutation(mutation) => {
            place_selector_arity_valid(&mutation.place)
        }
        runmat_mir::MirStmtKind::WorkspaceEffect { .. }
        | runmat_mir::MirStmtKind::StatementOutput(_)
        | runmat_mir::MirStmtKind::EnvironmentEffect(_) => true,
    }
}

fn statement_contains_legacy_binding_place(statement: &runmat_mir::MirStmtKind) -> bool {
    match statement {
        runmat_mir::MirStmtKind::PrepareFixedAssignment(owner) => {
            place_contains_legacy_binding(&owner.assignment.mutation.place)
                || owner.steps().iter().any(|step| match step {
                    runmat_mir::MirExpressionStep::Let { value, .. } => {
                        rvalue_contains_legacy_binding_place(value)
                    }
                    runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                        expansion_source_contains_legacy_binding_place(source)
                    }
                })
        }
        runmat_mir::MirStmtKind::PreparedAssign { preparation, value } => {
            place_contains_legacy_binding(&preparation.mutation.place)
                || rvalue_contains_legacy_binding_place(value)
        }
        runmat_mir::MirStmtKind::Assign { place, value } => {
            place_contains_legacy_binding(place) || rvalue_contains_legacy_binding_place(value)
        }
        runmat_mir::MirStmtKind::MultiAssign { targets, value } => {
            targets.targets.iter().any(|target| match target {
                runmat_mir::MirOutputTarget::Place(place) => place_contains_legacy_binding(place),
                runmat_mir::MirOutputTarget::Sequence(target) => {
                    place_contains_legacy_binding(target.base())
                }
                runmat_mir::MirOutputTarget::Discard => false,
            }) || rvalue_contains_legacy_binding_place(value)
        }
        runmat_mir::MirStmtKind::SequenceAssign { target, value } => {
            let base = match target {
                runmat_mir::MirSequenceTarget::Member { base, .. }
                | runmat_mir::MirSequenceTarget::DynamicMember { base, .. }
                | runmat_mir::MirSequenceTarget::CellContents { base, .. } => base,
            };
            place_contains_legacy_binding(base) || rvalue_contains_legacy_binding_place(value)
        }
        runmat_mir::MirStmtKind::CaptureSequence { source, .. } => {
            expansion_source_contains_legacy_binding_place(source)
        }
        runmat_mir::MirStmtKind::Expr(value) => rvalue_contains_legacy_binding_place(value),
        runmat_mir::MirStmtKind::PlaceMutation(mutation) => {
            place_contains_legacy_binding(&mutation.place)
        }
        runmat_mir::MirStmtKind::WorkspaceEffect { .. }
        | runmat_mir::MirStmtKind::StatementOutput(_)
        | runmat_mir::MirStmtKind::EnvironmentEffect(_) => false,
    }
}

fn place_contains_legacy_binding(place: &runmat_mir::MirPlace) -> bool {
    match place {
        runmat_mir::MirPlace::Binding(_) => true,
        runmat_mir::MirPlace::Local(_) => false,
        runmat_mir::MirPlace::Member(base, _)
        | runmat_mir::MirPlace::DynamicMember(base, _)
        | runmat_mir::MirPlace::Index(base, _) => place_contains_legacy_binding(base),
    }
}

fn place_selector_arity_valid(place: &runmat_mir::MirPlace) -> bool {
    match place {
        runmat_mir::MirPlace::Local(_) | runmat_mir::MirPlace::Binding(_) => true,
        runmat_mir::MirPlace::Member(base, _) => place_selector_arity_valid(base),
        runmat_mir::MirPlace::DynamicMember(base, _) => place_selector_arity_valid(base),
        runmat_mir::MirPlace::Index(base, indexing) => {
            place_selector_arity_valid(base) && indexing.components.len() <= u32::BITS as usize
        }
    }
}

fn sequence_target_selector_arity_valid(target: &runmat_mir::MirSequenceTarget) -> bool {
    match target {
        runmat_mir::MirSequenceTarget::Member { .. }
        | runmat_mir::MirSequenceTarget::DynamicMember { .. } => true,
        runmat_mir::MirSequenceTarget::CellContents { indexing, .. } => {
            indexing.components.len() <= u32::BITS as usize
        }
    }
}

fn expected_output_locals(
    instruction: &NativeInstruction,
    containing_statement: Option<&runmat_mir::MirStmtKind>,
) -> NativeCodegenResult<impl Iterator<Item = Option<NativeLocalId>>> {
    let locals = match (&instruction.site.phase, &instruction.operation) {
        (NativeSitePhase::Rvalue, NativeOperation::Rvalue { .. }) => {
            let statement = containing_statement.ok_or_else(|| {
                NativeCodegenError::new(
                    "native.ir.output_local_identity",
                    "rvalue output has no containing statement",
                )
                .at_point(instruction.site.point)
            })?;
            rvalue_output_locals(statement)?
        }
        (NativeSitePhase::TerminatorRvalue, NativeOperation::Rvalue { .. }) => vec![None],
        (NativeSitePhase::Statement, NativeOperation::Statement(statement)) => {
            statement_output_locals(statement)?
                .into_iter()
                .map(Some)
                .collect()
        }
        _ => Vec::new(),
    };
    Ok(locals.into_iter())
}

fn rvalue_output_locals(
    statement: &runmat_mir::MirStmtKind,
) -> NativeCodegenResult<Vec<Option<NativeLocalId>>> {
    use runmat_mir::{MirOutputTarget, MirStmtKind};
    match statement {
        MirStmtKind::PreparedAssign {
            preparation:
                runmat_mir::MirPreparedAssignment {
                    mutation: runmat_mir::MirPlaceMutation { place, .. },
                    ..
                },
            ..
        }
        | MirStmtKind::Assign { place, .. } => root_local(place)
            .map(checked_native_local)
            .transpose()
            .map(|local| vec![local]),
        MirStmtKind::MultiAssign { targets, .. }
            if targets
                .targets
                .iter()
                .any(|target| matches!(target, MirOutputTarget::Sequence(_))) =>
        {
            Ok(Vec::new())
        }
        MirStmtKind::MultiAssign { targets, .. } => targets
            .targets
            .iter()
            .map(|target| match target {
                MirOutputTarget::Place(place) => {
                    root_local(place).map(checked_native_local).transpose()
                }
                MirOutputTarget::Sequence(_) => Ok(None),
                MirOutputTarget::Discard => Ok(None),
            })
            .collect(),
        MirStmtKind::SequenceAssign { .. } => Ok(Vec::new()),
        MirStmtKind::CaptureSequence { .. } => Ok(Vec::new()),
        MirStmtKind::Expr(value) => Ok(value
            .has_statement_result()
            .then_some(None)
            .into_iter()
            .collect()),
        MirStmtKind::PrepareFixedAssignment(_)
        | MirStmtKind::PlaceMutation(_)
        | MirStmtKind::StatementOutput(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => Err(NativeCodegenError::new(
            "native.ir.output_local_identity",
            "statement without an rvalue contains an rvalue instruction",
        )),
    }
}

fn statement_output_locals(
    statement: &runmat_mir::MirStmtKind,
) -> NativeCodegenResult<Vec<NativeLocalId>> {
    use runmat_mir::{MirOutputTarget, MirStmtKind};
    let roots = match statement {
        MirStmtKind::PreparedAssign {
            preparation:
                runmat_mir::MirPreparedAssignment {
                    mutation: runmat_mir::MirPlaceMutation { place, .. },
                    ..
                },
            ..
        }
        | MirStmtKind::Assign { place, .. } => root_local(place).into_iter().collect(),
        MirStmtKind::MultiAssign { targets, .. } => targets
            .targets
            .iter()
            .filter_map(|target| match target {
                MirOutputTarget::Place(place) => root_local(place),
                MirOutputTarget::Sequence(target) => root_local(target.base()),
                MirOutputTarget::Discard => None,
            })
            .collect(),
        MirStmtKind::SequenceAssign { target, .. } => {
            root_local(target.base()).into_iter().collect()
        }
        MirStmtKind::PrepareFixedAssignment(preparation) => match preparation.output() {
            Some(runmat_mir::MirFixedSelectorDestination::Scalar(local)) => vec![local],
            Some(runmat_mir::MirFixedSelectorDestination::Sequence(_)) | None => Vec::new(),
        },
        MirStmtKind::CaptureSequence { .. } => Vec::new(),
        MirStmtKind::PlaceMutation(mutation) => root_local(&mutation.place).into_iter().collect(),
        MirStmtKind::WorkspaceEffect { bindings, .. } => bindings.clone(),
        MirStmtKind::Expr(_)
        | MirStmtKind::StatementOutput(_)
        | MirStmtKind::EnvironmentEffect(_) => Vec::new(),
    };
    let mut seen = BTreeSet::new();
    roots
        .into_iter()
        .filter(|local| seen.insert(*local))
        .map(checked_native_local)
        .collect()
}

fn checked_native_local(local: runmat_mir::MirLocalId) -> NativeCodegenResult<NativeLocalId> {
    u32::try_from(local.0).map(NativeLocalId).map_err(|_| {
        NativeCodegenError::new(
            "native.ir.output_local_identity",
            "MIR output local exceeds the Native IR schema",
        )
    })
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
            let arity =
                match result {
                    NativeRvalueResult::Assignment => 1,
                    NativeRvalueResult::MultiAssignment(targets) => {
                        if targets.targets.iter().any(|target| {
                            matches!(target, runmat_mir::MirOutputTarget::Sequence(_))
                        }) {
                            0
                        } else {
                            targets.targets.len()
                        }
                    }
                    NativeRvalueResult::SequenceAssignment(_) => 0,
                    NativeRvalueResult::Expression => 1,
                    NativeRvalueResult::Discard => 0,
                    NativeRvalueResult::Terminator => {
                        return Err(NativeCodegenError::new(
                            "native.ir.operation_phase",
                            "terminator rvalue role appears at a statement site",
                        ))
                    }
                };
            let runtime_cardinality = matches!(result, NativeRvalueResult::SequenceAssignment(_))
                || matches!(
                    result,
                    NativeRvalueResult::MultiAssignment(targets)
                        if targets.targets.iter().any(|target| matches!(
                            target,
                            runmat_mir::MirOutputTarget::Sequence(_)
                        ))
                );
            if !runtime_cardinality {
                if let runmat_mir::MirRvalue::Call(call) = value {
                    let Some(requested) = call.requested_outputs.known_count() else {
                        return Err(NativeCodegenError::new(
                            "native.ir.call_output_arity",
                            "runtime-determined call output count requires a dynamic native result role",
                        ));
                    };
                    if requested != arity {
                        return Err(NativeCodegenError::new(
                            "native.ir.call_output_arity",
                            "call result role differs from requested output arity",
                        ));
                    }
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
                        let place = match target {
                            runmat_mir::MirOutputTarget::Place(place) => Some(place),
                            runmat_mir::MirOutputTarget::Sequence(target) => Some(target.base()),
                            runmat_mir::MirOutputTarget::Discard => None,
                        };
                        if let Some(local) = place.and_then(root_local) {
                            roots.insert(local);
                        }
                    }
                }
                runmat_mir::MirStmtKind::PrepareFixedAssignment(preparation) => {
                    if let Some(runmat_mir::MirFixedSelectorDestination::Scalar(local)) =
                        preparation.output()
                    {
                        roots.insert(local);
                    }
                }
                runmat_mir::MirStmtKind::PreparedAssign { preparation, .. } => {
                    if let Some(local) = root_local(&preparation.mutation.place) {
                        roots.insert(local);
                    }
                }
                runmat_mir::MirStmtKind::CaptureSequence { .. } => {}
                runmat_mir::MirStmtKind::SequenceAssign { target, .. } => {
                    if let Some(local) = root_local(target.base()) {
                        roots.insert(local);
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
                | runmat_mir::MirStmtKind::StatementOutput(_)
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
    if state.point.function != function.portable_id
        || !available.contains(&state.side_effect_epoch)
        || state.locals.len() != function.local_count()
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
        .filter(|region| region.id.function == function.portable_id)
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
        .filter(|region| region.id.function == function.portable_id)
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
