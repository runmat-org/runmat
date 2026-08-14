use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use runmat_native_codegen::{
    NativeBlockId, NativeEdge, NativeFunction, NativeLocalKind, NativeTerminatorKind, NativeValueId,
};
use runmat_runtime::call::function_abi::{prepare_function_inputs, FunctionInputSpec};
use runmat_runtime::native::{NativeRoot, NativeRootKind, NativeRootSet, NativeValueRef};
use runmat_runtime::{context::RuntimeContext, RuntimeError};

use crate::deopt::{
    DeoptimizationPolicy, FaultInjection, MaterializedFrame, NativeMaterializationContext,
};
use crate::memory::{ScopedValueRoots, ValueArena};
use crate::specialization::{GuardFailure, GuardFailureKind};
use crate::{JitError, JitResult};

pub(super) struct ActiveForLoop {
    pub iterator: runmat_runtime::iteration::ForColumnIterator,
    body_blocks: BTreeSet<NativeBlockId>,
}

pub(super) struct ActiveExceptionHandler {
    pub catch_edge: NativeEdge,
    protected_blocks: BTreeSet<NativeBlockId>,
}

pub(super) struct HostStateInput {
    pub function: NativeFunction,
    pub arguments: Vec<runmat_value::Value>,
    pub requested_outputs: usize,
    pub runtime: RuntimeContext,
    pub program_capture: Option<Vec<u8>>,
    pub functions: Rc<Vec<NativeFunction>>,
    pub captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    pub deoptimization: DeoptimizationPolicy,
    pub interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
}

pub(super) struct HostState {
    pub function: NativeFunction,
    pub functions: Rc<Vec<NativeFunction>>,
    pub runtime: RuntimeContext,
    pub arena: ValueArena,
    pub locals: Vec<NativeValueRef>,
    pub values: BTreeMap<NativeValueId, NativeValueRef>,
    pub roots: Vec<NativeRoot>,
    pub last_error: Option<RuntimeError>,
    pub host_failure: Option<JitError>,
    pub current_source: runmat_runtime::native::NativeSourceLocation,
    pub pending_place_mutation: Option<runmat_mir::MirPlaceMutation>,
    pub program_capture: Option<Vec<u8>>,
    pub pending_await: Option<super::awaiting::PendingAwait>,
    resume_target: Option<runmat_runtime::native::NativeSiteRequest>,
    current_block: Option<NativeBlockId>,
    global_bindings: BTreeMap<usize, String>,
    persistent_bindings: BTreeMap<usize, String>,
    active_for_loops: BTreeMap<NativeBlockId, ActiveForLoop>,
    active_exception_handlers: Vec<ActiveExceptionHandler>,
    next_await_continuation: u64,
    deoptimization: DeoptimizationPolicy,
    retired_guards: BTreeSet<runmat_types::RegionGuardId>,
    pending_deoptimization: Option<MaterializedFrame>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    supplied_inputs: usize,
    requested_outputs: usize,
    missing_input_locals: Vec<runmat_native_codegen::NativeLocalId>,
}

impl HostState {
    pub fn new(input: HostStateInput) -> JitResult<(Self, Vec<NativeValueRef>)> {
        let HostStateInput {
            function,
            arguments,
            requested_outputs,
            runtime,
            program_capture,
            functions,
            captures,
            deoptimization,
            interpreter_resume_points,
        } = input;
        let construction_values = arguments
            .iter()
            .cloned()
            .chain(captures.iter().map(|capture| capture.value.clone()))
            .collect();
        let _construction_roots =
            ScopedValueRoots::register(construction_values, "native_invocation_construction")?;
        let input_specs = function
            .argument_validations
            .iter()
            .map(|validation| {
                let input_index = function
                    .abi
                    .fixed_inputs
                    .iter()
                    .position(|local| *local == validation.input)
                    .ok_or_else(|| {
                        JitError::Host(
                            "native argument validation does not name a fixed input".into(),
                        )
                    })?;
                Ok(FunctionInputSpec {
                    input_index,
                    size: validation.size.as_ref(),
                    class_name: validation.class_name.as_deref(),
                    validators: &validation.validators,
                    default_value: validation.default_value.as_ref(),
                })
            })
            .collect::<JitResult<Vec<_>>>()?;
        let supplied_inputs = arguments.len();
        let prepared = prepare_function_inputs(
            &function.name,
            &arguments,
            function.abi.fixed_inputs.len(),
            function.abi.varargin.is_some(),
            &input_specs,
        )?;
        let missing_input_locals = function
            .abi
            .fixed_inputs
            .iter()
            .zip(&prepared.fixed)
            .filter_map(|(local, value)| value.is_none().then_some(*local))
            .collect();
        let mut arena = ValueArena::new()?;
        let argument_refs = arguments
            .into_iter()
            .map(|value| arena.insert(value))
            .collect::<Vec<_>>();
        let mut locals = vec![NativeValueRef::NULL; function.local_count()];
        let capture_locals = function
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
            .collect::<Vec<_>>();
        if capture_locals.len() != captures.len() {
            return Err(JitError::Host(format!(
                "native lexical entry expected {} captures but received {}",
                capture_locals.len(),
                captures.len()
            )));
        }
        for local in capture_locals {
            let binding = local.binding.ok_or_else(|| {
                JitError::Host("native capture local has no semantic binding".into())
            })?;
            let capture = captures
                .iter()
                .find(|capture| capture.binding == binding)
                .ok_or_else(|| {
                    JitError::Host("native lexical capture binding is missing".into())
                })?;
            locals[local.id.0 as usize] = arena.insert(capture.value.clone());
        }
        for (local, value) in function.abi.fixed_inputs.iter().zip(&prepared.fixed) {
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                JitError::Host("function ABI input local is out of bounds".into())
            })?;
            if let Some(value) = value {
                *slot = arena.insert(value.clone());
            }
        }
        if let (Some(local), Some(varargin)) = (function.abi.varargin, prepared.varargin) {
            locals[local.0 as usize] = arena.insert(runmat_value::Value::Cell(varargin));
        }
        if let Some(local) = function.abi.varargout {
            let empty = runmat_value::CellArray::new(Vec::new(), 1, 0)
                .map_err(|error| JitError::Host(format!("varargout: {error}")))?;
            locals[local.0 as usize] = arena.insert(runmat_value::Value::Cell(empty));
        }
        if let Some(local) = function.abi.implicit_nargin {
            locals[local.0 as usize] =
                arena.insert(runmat_value::Value::Num(prepared.nargin as f64));
        }
        if let Some(local) = function.abi.implicit_nargout {
            locals[local.0 as usize] =
                arena.insert(runmat_value::Value::Num(requested_outputs as f64));
        }
        let roots = locals
            .iter()
            .enumerate()
            .map(|(slot, value)| NativeRoot {
                value: *value,
                kind: NativeRootKind::LOCAL,
                slot: slot as u32,
            })
            .collect::<Vec<_>>();
        let source = function.source.0;
        let mut state = Self {
            function,
            functions,
            runtime,
            arena,
            locals,
            values: BTreeMap::new(),
            roots,
            last_error: None,
            host_failure: None,
            current_source: runmat_runtime::native::NativeSourceLocation {
                source,
                ..runmat_runtime::native::NativeSourceLocation::default()
            },
            active_for_loops: BTreeMap::new(),
            pending_place_mutation: None,
            program_capture,
            pending_await: None,
            resume_target: None,
            current_block: None,
            global_bindings: BTreeMap::new(),
            persistent_bindings: BTreeMap::new(),
            active_exception_handlers: Vec::new(),
            next_await_continuation: 1,
            deoptimization,
            retired_guards: BTreeSet::new(),
            pending_deoptimization: None,
            interpreter_resume_points,
            supplied_inputs,
            requested_outputs,
            missing_input_locals,
        };
        state.enter_block(state.function.entry)?;
        Ok((state, argument_refs))
    }

    pub fn lexical_captures(
        &self,
        function: runmat_types::ProgramFunctionId,
    ) -> JitResult<Option<Vec<runmat_runtime::call::lexical::LexicalCapture>>> {
        let target = self
            .functions
            .iter()
            .find(|candidate| candidate.id == function);
        let Some(target) = target else {
            return Ok(None);
        };
        let mut captures = Vec::new();
        for target_local in target
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
        {
            let binding = target_local.binding.ok_or_else(|| {
                JitError::Host("native capture local has no semantic binding".into())
            })?;
            let source = self
                .function
                .locals
                .iter()
                .find(|local| local.binding == Some(binding))
                .ok_or_else(|| {
                    JitError::Host("nested native capture is not visible in its caller".into())
                })?;
            let value = self
                .locals
                .get(source.id.0 as usize)
                .copied()
                .ok_or_else(|| JitError::Host("native capture local is out of bounds".into()))?;
            captures.push(runmat_runtime::call::lexical::LexicalCapture {
                binding,
                value: self.arena.get(value)?.clone(),
            });
        }
        Ok((!captures.is_empty()).then_some(captures))
    }

    pub fn program_function(
        &self,
        function: runmat_types::ProgramFunctionId,
    ) -> Option<&NativeFunction> {
        self.functions
            .iter()
            .find(|candidate| candidate.id == function)
    }

    pub fn apply_lexical_captures(
        &mut self,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    ) -> JitResult<()> {
        for capture in captures {
            let local = self
                .function
                .locals
                .iter()
                .find(|local| local.binding == Some(capture.binding))
                .ok_or_else(|| JitError::Host("returned lexical binding is not visible".into()))?;
            self.locals[local.id.0 as usize] = self.arena.insert(capture.value);
        }
        Ok(())
    }

    pub fn capture_results(&self) -> JitResult<Vec<runmat_runtime::call::lexical::LexicalCapture>> {
        self.function
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
            .map(|local| {
                let binding = local.binding.ok_or_else(|| {
                    JitError::Host("native capture local has no semantic binding".into())
                })?;
                let value = self.locals[local.id.0 as usize];
                Ok(runmat_runtime::call::lexical::LexicalCapture {
                    binding,
                    value: self.arena.get(value)?.clone(),
                })
            })
            .collect()
    }

    pub fn enter_block(&mut self, block: runmat_native_codegen::NativeBlockId) -> JitResult<()> {
        let parameters = self
            .function
            .blocks
            .iter()
            .find(|candidate| candidate.id == block)
            .ok_or_else(|| JitError::Host(format!("native block {} is unavailable", block.0)))?
            .parameters
            .clone();
        for parameter in parameters {
            let value = self
                .locals
                .get(parameter.local.0 as usize)
                .copied()
                .ok_or_else(|| JitError::Host("block parameter local is out of bounds".into()))?;
            self.values.insert(parameter.value, value);
        }
        Ok(())
    }

    pub fn refresh_roots(&mut self) -> NativeRootSet {
        for (root, value) in self.roots.iter_mut().zip(&self.locals) {
            root.value = *value;
        }
        NativeRootSet {
            roots: self.roots.as_ptr(),
            count: self.roots.len(),
        }
    }

    pub fn prepare_resume(
        &mut self,
        resume: runmat_runtime::native::NativeResumeState,
    ) -> JitResult<()> {
        let target = runmat_runtime::native::NativeSiteRequest {
            function: resume.function,
            block: resume.block,
            position: resume.position,
            phase: runmat_runtime::native::NativeSitePhase(resume.phase),
            ordinal: resume.ordinal,
            reserved: 0,
        };
        target
            .validate()
            .map_err(|error| JitError::Host(error.to_string()))?;
        if target.function != self.function.id.0
            || !self
                .function
                .expected_sites
                .iter()
                .any(|site| native_site_matches(site, target))
        {
            return Err(JitError::Host(
                "native resume target is not a verified site in this function".into(),
            ));
        }
        self.resume_target = Some(target);
        Ok(())
    }

    pub fn evaluate_guard(
        &self,
        guard: &runmat_native_codegen::NativeRegionGuard,
    ) -> Result<(), GuardFailure> {
        if self.retired_guards.contains(&guard.contract.id) {
            return Ok(());
        }
        let value = guard
            .value
            .and_then(|value| self.values.get(&value).copied())
            .and_then(|value| (!value.is_null()).then_some(value))
            .and_then(|value| self.arena.get(value).ok());
        self.deoptimization.guards.evaluate(&guard.contract, value)
    }

    pub fn should_inject_guard(&mut self, guard: runmat_types::RegionGuardId) -> bool {
        if self.deoptimization.inject == Some(FaultInjection::Guard(guard)) {
            self.deoptimization.inject = None;
            true
        } else {
            false
        }
    }

    pub fn should_inject_safepoint(
        &mut self,
        safepoint: runmat_native_codegen::NativeSafepointId,
    ) -> bool {
        if self.deoptimization.inject == Some(FaultInjection::Safepoint(safepoint)) {
            self.deoptimization.inject = None;
            true
        } else {
            false
        }
    }

    pub fn materialize_deoptimization(
        &mut self,
        frame: &runmat_native_codegen::NativeFrameState,
        site: &runmat_native_codegen::NativeMirSite,
    ) -> JitResult<MaterializedFrame> {
        let bytecode_pc = self
            .interpreter_resume_supported(site)
            .then(|| self.interpreter_resume_points.get(&frame.point).copied())
            .flatten();
        let materialized = MaterializedFrame::from_native(
            frame,
            NativeMaterializationContext {
                phase: site.phase,
                ordinal: site.ordinal,
                bytecode_pc,
                supplied_inputs: self.supplied_inputs,
                requested_outputs: self.requested_outputs,
                missing_input_locals: self.missing_input_locals.clone(),
                global_bindings: self.global_bindings.clone(),
                persistent_bindings: self.persistent_bindings.clone(),
            },
            |value| {
                let Some(reference) = self.values.get(&value).copied() else {
                    return Err(JitError::Host(format!(
                        "native frame references unavailable SSA value {}",
                        value.0
                    )));
                };
                if reference.is_null() {
                    Ok(None)
                } else {
                    self.arena.get(reference).cloned().map(Some)
                }
            },
        )?;
        self.pending_deoptimization = Some(materialized.clone());
        Ok(materialized)
    }

    pub fn take_deoptimization(&mut self) -> JitResult<MaterializedFrame> {
        self.pending_deoptimization
            .take()
            .ok_or_else(|| JitError::Host("native deoptimization has no materialized frame".into()))
    }

    pub fn retire_guard(&mut self, guard: runmat_types::RegionGuardId) {
        self.retired_guards.insert(guard);
    }

    pub fn deoptimization_target(&self) -> runmat_runtime::native::NativeResumeKind {
        self.deoptimization.target.native()
    }

    pub fn effective_deoptimization_target(
        &self,
        frame: &MaterializedFrame,
    ) -> runmat_runtime::native::NativeResumeKind {
        let requested = self.deoptimization_target();
        if requested == runmat_runtime::native::NativeResumeKind::INTERPRETER
            && frame.site.bytecode_pc.is_none()
        {
            runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE
        } else {
            requested
        }
    }

    fn interpreter_resume_supported(&self, site: &runmat_native_codegen::NativeMirSite) -> bool {
        if self.pending_await.is_some()
            || self.pending_place_mutation.is_some()
            || self.last_error.is_some()
            || !self.active_for_loops.is_empty()
            || !self.active_exception_handlers.is_empty()
        {
            return false;
        }
        match site.phase {
            runmat_native_codegen::NativeSitePhase::Rvalue => true,
            runmat_native_codegen::NativeSitePhase::Statement => {
                !self.function.expected_sites.iter().any(|candidate| {
                    candidate.point == site.point
                        && candidate.phase == runmat_native_codegen::NativeSitePhase::Rvalue
                })
            }
            runmat_native_codegen::NativeSitePhase::TerminatorRvalue => site.ordinal == 0,
            runmat_native_codegen::NativeSitePhase::Terminator => {
                !self.function.expected_sites.iter().any(|candidate| {
                    candidate.point == site.point
                        && candidate.phase
                            == runmat_native_codegen::NativeSitePhase::TerminatorRvalue
                })
            }
        }
    }

    pub fn deoptimization_reason(
        failure: GuardFailureKind,
    ) -> runmat_runtime::native::NativeDeoptReason {
        match failure {
            GuardFailureKind::Representation | GuardFailureKind::Capability => {
                runmat_runtime::native::NativeDeoptReason::REPRESENTATION
            }
            GuardFailureKind::RuntimeState => {
                runmat_runtime::native::NativeDeoptReason::RUNTIME_STATE
            }
        }
    }

    pub fn enter_site_block(&mut self, block: NativeBlockId) {
        self.current_block = Some(block);
    }

    pub fn next_await_identity(&mut self) -> JitResult<(u64, u64)> {
        if self.pending_await.is_some() {
            return Err(JitError::Host(
                "native invocation already has a pending await".into(),
            ));
        }
        let continuation = self.next_await_continuation;
        self.next_await_continuation = self
            .next_await_continuation
            .checked_add(1)
            .ok_or_else(|| JitError::Host("native await identity exhausted".into()))?;
        Ok((continuation, 1))
    }

    pub fn enter_exception_handler(
        &mut self,
        try_edge: &NativeEdge,
        catch_edge: &NativeEdge,
    ) -> JitResult<()> {
        let try_reachable = self.reachable_blocks(try_edge.target)?;
        let catch_reachable = self.reachable_blocks(catch_edge.target)?;
        let protected_blocks = try_reachable
            .difference(&catch_reachable)
            .copied()
            .collect::<BTreeSet<_>>();
        if !protected_blocks.contains(&try_edge.target) {
            return Err(JitError::Host(
                "native try region has no protected entry block".into(),
            ));
        }
        self.active_exception_handlers.push(ActiveExceptionHandler {
            catch_edge: catch_edge.clone(),
            protected_blocks,
        });
        Ok(())
    }

    pub fn take_exception_handler(&mut self) -> Option<ActiveExceptionHandler> {
        let current = self.current_block?;
        let index = self
            .active_exception_handlers
            .iter()
            .rposition(|handler| handler.protected_blocks.contains(&current))?;
        let handler = self.active_exception_handlers.remove(index);
        self.active_exception_handlers.truncate(index);
        Some(handler)
    }

    pub fn resume_request_for_block(
        &self,
        block: NativeBlockId,
    ) -> JitResult<runmat_runtime::native::NativeSiteRequest> {
        let site = self
            .function
            .expected_sites
            .iter()
            .find(|site| site.point.block == block.0)
            .ok_or_else(|| JitError::Host("native resume block has no verified site".into()))?;
        Ok(runmat_runtime::native::NativeSiteRequest {
            function: self.function.id.0,
            block: site.point.block,
            position: site.point.position,
            phase: native_site_phase(site.phase),
            ordinal: site.ordinal,
            reserved: 0,
        })
    }

    pub fn skip_before_resume(
        &mut self,
        request: runmat_runtime::native::NativeSiteRequest,
    ) -> JitResult<bool> {
        let Some(target) = self.resume_target else {
            return Ok(false);
        };
        if request == target {
            self.resume_target = None;
            return Ok(false);
        }
        skip_before_target(&self.function.expected_sites, target, request)
    }

    pub fn annotate_error(&self, mut error: RuntimeError) -> RuntimeError {
        if error.span.is_none() {
            let start = self.current_source.start as usize;
            let length = self
                .current_source
                .end
                .saturating_sub(self.current_source.start)
                .max(1) as usize;
            error.span = Some((start, length).into());
        }
        if error.context.call_frames.is_empty() && error.context.call_stack.is_empty() {
            let span = error.span.as_ref().map(|span| {
                let start = span.offset();
                (start, start + span.len())
            });
            error.context.call_frames.push(runmat_runtime::CallFrame {
                function: self.function.name.clone(),
                source_id: Some(self.function.source.0 as usize),
                span,
            });
        }
        error
    }

    pub fn set_local(&mut self, slot: usize, value: NativeValueRef) -> JitResult<()> {
        let local = self
            .locals
            .get_mut(slot)
            .ok_or_else(|| JitError::Host("native local is out of bounds".into()))?;
        *local = value;
        self.synchronize_session_binding(slot, value)
    }

    pub fn declare_global(&mut self, slot: usize, name: String) -> JitResult<()> {
        if self.persistent_bindings.contains_key(&slot) {
            return Err(JitError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        self.global_bindings.insert(slot, name.clone());
        let value = runmat_runtime::workspace::session::global_value(&name)
            .unwrap_or_else(empty_workspace_value);
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
    }

    pub fn declare_persistent(&mut self, slot: usize, name: String) -> JitResult<()> {
        if self.global_bindings.contains_key(&slot) {
            return Err(JitError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        self.persistent_bindings.insert(slot, name.clone());
        let value =
            runmat_runtime::workspace::session::persistent_named_value(&self.function.name, &name)
                .unwrap_or_else(empty_workspace_value);
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
    }

    fn synchronize_session_binding(
        &mut self,
        slot: usize,
        reference: NativeValueRef,
    ) -> JitResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let value = self.arena.get(reference)?.clone();
        if let Some(name) = self.global_bindings.get(&slot) {
            runmat_runtime::workspace::session::store_global_named(name, value.clone());
        }
        if let Some(name) = self.persistent_bindings.get(&slot) {
            runmat_runtime::workspace::session::store_persistent_named(
                &self.function.name,
                name,
                value,
            );
        }
        Ok(())
    }

    pub fn has_for_loop(&self, header: NativeBlockId) -> bool {
        self.active_for_loops.contains_key(&header)
    }

    pub fn start_for_loop(
        &mut self,
        header: NativeBlockId,
        body: NativeBlockId,
        iterator: runmat_runtime::iteration::ForColumnIterator,
    ) {
        let body_blocks = self.blocks_reaching_header(header, body);
        self.active_for_loops.insert(
            header,
            ActiveForLoop {
                iterator,
                body_blocks,
            },
        );
    }

    pub fn for_loop_mut(&mut self, header: NativeBlockId) -> JitResult<&mut ActiveForLoop> {
        self.active_for_loops
            .get_mut(&header)
            .ok_or_else(|| JitError::Host("native for-loop state is unavailable".into()))
    }

    /// Retire loop snapshots when control leaves their natural loop body.
    /// This covers `break` and exceptional structured exits as well as normal
    /// exhaustion, while retaining state across body backedges and `continue`.
    pub fn take_control_edge(&mut self, target: NativeBlockId) {
        self.active_for_loops
            .retain(|header, active| target == *header || active.body_blocks.contains(&target));
        self.active_exception_handlers
            .retain(|handler| handler.protected_blocks.contains(&target));
    }

    fn blocks_reaching_header(
        &self,
        header: NativeBlockId,
        body: NativeBlockId,
    ) -> BTreeSet<NativeBlockId> {
        let mut reverse = BTreeMap::<NativeBlockId, Vec<NativeBlockId>>::new();
        for block in &self.function.blocks {
            for target in successor_blocks(&block.terminator.kind) {
                reverse.entry(target).or_default().push(block.id);
            }
        }
        let mut reaches_header = BTreeSet::from([header]);
        let mut pending = vec![header];
        while let Some(target) = pending.pop() {
            for predecessor in reverse.get(&target).into_iter().flatten() {
                if reaches_header.insert(*predecessor) {
                    pending.push(*predecessor);
                }
            }
        }
        let mut reachable_from_body = BTreeSet::new();
        let mut pending = vec![body];
        while let Some(block) = pending.pop() {
            if block == header
                || !reaches_header.contains(&block)
                || !reachable_from_body.insert(block)
            {
                continue;
            }
            if let Some(block) = self
                .function
                .blocks
                .iter()
                .find(|candidate| candidate.id == block)
            {
                pending.extend(successor_blocks(&block.terminator.kind));
            }
        }
        reachable_from_body
    }

    fn reachable_blocks(&self, start: NativeBlockId) -> JitResult<BTreeSet<NativeBlockId>> {
        if !self.function.blocks.iter().any(|block| block.id == start) {
            return Err(JitError::Host(
                "native exception edge targets an unavailable block".into(),
            ));
        }
        let mut reachable = BTreeSet::new();
        let mut pending = vec![start];
        while let Some(block) = pending.pop() {
            if !reachable.insert(block) {
                continue;
            }
            let block = self
                .function
                .blocks
                .iter()
                .find(|candidate| candidate.id == block)
                .ok_or_else(|| JitError::Host("native CFG block is unavailable".into()))?;
            pending.extend(successor_blocks(&block.terminator.kind));
        }
        Ok(reachable)
    }
}

fn skip_before_target(
    expected_sites: &[runmat_native_codegen::NativeMirSite],
    target: runmat_runtime::native::NativeSiteRequest,
    request: runmat_runtime::native::NativeSiteRequest,
) -> JitResult<bool> {
    if request.block != target.block {
        return Err(JitError::Host(
            "generated re-entry reached a block before its resume target".into(),
        ));
    }
    let current = expected_sites
        .iter()
        .position(|site| native_site_matches(site, request));
    let target = expected_sites
        .iter()
        .position(|site| native_site_matches(site, target));
    match (current, target) {
        (Some(current), Some(target)) if current < target => Ok(true),
        (Some(_), Some(_)) => Err(JitError::Host(
            "generated re-entry passed its exact resume target".into(),
        )),
        _ => Err(JitError::Host(
            "generated re-entry produced an unverified site".into(),
        )),
    }
}

fn native_site_matches(
    site: &runmat_native_codegen::NativeMirSite,
    request: runmat_runtime::native::NativeSiteRequest,
) -> bool {
    site.point.block == request.block
        && site.point.position == request.position
        && native_site_phase(site.phase) == request.phase
        && site.ordinal == request.ordinal
}

fn native_site_phase(
    phase: runmat_native_codegen::NativeSitePhase,
) -> runmat_runtime::native::NativeSitePhase {
    match phase {
        runmat_native_codegen::NativeSitePhase::Rvalue => {
            runmat_runtime::native::NativeSitePhase::RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Statement => {
            runmat_runtime::native::NativeSitePhase::STATEMENT
        }
        runmat_native_codegen::NativeSitePhase::TerminatorRvalue => {
            runmat_runtime::native::NativeSitePhase::TERMINATOR_RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Terminator => {
            runmat_runtime::native::NativeSitePhase::TERMINATOR
        }
    }
}

fn empty_workspace_value() -> runmat_value::Value {
    runmat_value::Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0])
            .expect("the canonical empty workspace shape is valid"),
    )
}

fn successor_blocks(kind: &NativeTerminatorKind) -> Vec<NativeBlockId> {
    match kind {
        NativeTerminatorKind::Goto { edge } => vec![edge.target],
        NativeTerminatorKind::Branch {
            then_edge,
            else_edge,
            ..
        } => vec![then_edge.target, else_edge.target],
        NativeTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, edge)| edge.target)
            .chain(std::iter::once(otherwise.target))
            .collect(),
        NativeTerminatorKind::For { body, exit, .. }
        | NativeTerminatorKind::ParFor { body, exit, .. }
        | NativeTerminatorKind::Spmd { body, exit, .. } => vec![body.target, exit.target],
        NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } => vec![try_edge.target, catch_edge.target],
        NativeTerminatorKind::Await { resume, .. } => vec![resume.target],
        NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_native_codegen::{NativeMirSite, NativeSitePhase};
    use runmat_runtime::native::{NativeSitePhase as RuntimePhase, NativeSiteRequest};
    use runmat_types::{ProgramFunctionId, ProgramPointId};

    fn site(position: u32, phase: NativeSitePhase, ordinal: u32) -> NativeMirSite {
        NativeMirSite {
            point: ProgramPointId {
                function: ProgramFunctionId(7),
                block: 4,
                position,
            },
            phase,
            ordinal,
            construct: runmat_mir::MirConstructKind::Use,
        }
    }

    fn request(position: u32, phase: RuntimePhase, ordinal: u32) -> NativeSiteRequest {
        NativeSiteRequest {
            function: 7,
            block: 4,
            position,
            phase,
            ordinal,
            reserved: 0,
        }
    }

    #[test]
    fn exact_resume_skips_only_verified_predecessor_sites() {
        let expected = vec![
            site(0, NativeSitePhase::Rvalue, 0),
            site(0, NativeSitePhase::Statement, 1),
            site(1, NativeSitePhase::Terminator, 0),
        ];
        let target = request(0, RuntimePhase::STATEMENT, 1);
        assert!(
            skip_before_target(&expected, target, request(0, RuntimePhase::RVALUE, 0)).unwrap()
        );
        assert!(
            skip_before_target(&expected, target, request(1, RuntimePhase::TERMINATOR, 0)).is_err()
        );
        let wrong_block = NativeSiteRequest {
            block: 3,
            ..request(0, RuntimePhase::RVALUE, 0)
        };
        assert!(skip_before_target(&expected, target, wrong_block).is_err());
    }
}
