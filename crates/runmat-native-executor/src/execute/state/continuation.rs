use super::*;

impl HostState {
    /// Remove only SSA tokens proven dead by the admitted canonical analysis.
    /// Locals, captures, outputs and explicit arena pins remain separate roots.
    pub(in crate::execute) fn retire_site_values(
        &mut self,
        site: &runmat_native_codegen::NativeMirSite,
    ) -> NativeExecutorResult<()> {
        if !self.runtime.has_cleanup_candidates() {
            return Ok(());
        }
        let liveness = self.liveness.as_ref().ok_or_else(|| {
            NativeExecutorError::Host("native retirement has no admitted liveness metadata".into())
        })?;
        for value in liveness.get().retired_after(self.function.id, site) {
            self.values.remove(value);
        }
        Ok(())
    }

    /// Edge operands have already been rebound to the target parameters.
    /// Prune against the target entry, not the predecessor kill set: a loop
    /// parameter may have just replaced its own earlier dynamic generation.
    pub(in crate::execute) fn retire_edge_values(
        &mut self,
        target: NativeBlockId,
    ) -> NativeExecutorResult<()> {
        if !self.runtime.has_cleanup_candidates() {
            return Ok(());
        }
        let liveness = self.liveness.as_ref().ok_or_else(|| {
            NativeExecutorError::Host("native retirement has no admitted liveness metadata".into())
        })?;
        let function = self.function.id;
        self.values
            .retain(|value, _| liveness.get().live_on_entry(function, target, *value));
        Ok(())
    }

    /// Reclaim obsolete physical generations, never logical SSA definitions.
    /// Call only between settled native sites, with no arena Value borrow live.
    pub(in crate::execute) fn retire_obsolete_generations(
        &mut self,
        call: &mut runmat_runtime::native::NativeCall,
    ) -> NativeExecutorResult<()> {
        if !self.arena.has_retirement_debt() && !self.runtime.has_cleanup_candidates() {
            return Ok(());
        }
        // Use the original allocation policy, not a guessed next-operation
        // quote or ledger-pressure fraction. Finite mode pays a full census
        // after any insertion or final-pin release so obsolete residents cannot
        // consume the next operation's headroom. Current SSA/pins can still exhaust it.
        // A policy error grants nothing and is left to its original operation
        // owner; this maintenance check must not newly deny a pure operation.
        let managed = matches!(
            runmat_runtime::context::host_allocation::HostAllocationReceipt::policy(Some(
                &self.runtime
            )),
            Ok(runmat_runtime::context::host_allocation::HostAllocationPolicy::Managed)
        );
        if !managed && !self.runtime.has_cleanup_candidates() && !self.arena.retirement_due() {
            return Ok(());
        }
        if call.frame.is_null() {
            return Err(NativeExecutorError::Host("native call has no frame".into()));
        }
        // Reuse the canonical typed-owner root census, without mirroring SSA
        // tokens into another variable-width root buffer. Refresh may relocate
        // that buffer, so republish its borrowed ABI view before proceeding.
        let roots = self.refresh_roots()?;
        // SAFETY: the checked frame is borrowed from ActiveInvocation for the
        // complete synchronous machine entry.
        unsafe { (*call.frame).roots = roots };
        self.arena.begin_retirement();
        for root in self.roots.iter() {
            self.arena.mark_resident(root.value)?;
        }
        // Frame materialization and evaluated operands may still require an
        // old local generation. NativeFrameState is not a last-use analysis.
        for reference in self.values.values() {
            self.arena.mark_resident(*reference)?;
        }
        // ActiveInvocation owns initialized argument/result windows throughout
        // each machine entry. Generated code retains no additional opaque
        // Value tokens outside those windows and HostState's typed owners.
        if call.argument_count != 0 {
            if call.arguments.is_null() {
                return Err(NativeExecutorError::Host(
                    "native arguments are missing".into(),
                ));
            }
            // SAFETY: ActiveInvocation::enter_once publishes its initialized
            // argument_refs vector and its exact length for this entry.
            for reference in
                unsafe { std::slice::from_raw_parts(call.arguments, call.argument_count) }
            {
                self.arena.mark_resident(*reference)?;
            }
        }
        if call.result_capacity != 0 {
            if call.results.is_null() {
                return Err(NativeExecutorError::Host(
                    "native results are missing".into(),
                ));
            }
            // SAFETY: ActiveInvocation allocates results as vec![NULL; count],
            // clears it before entry, and publishes that vector's exact length.
            for reference in
                unsafe { std::slice::from_raw_parts(call.results, call.result_capacity) }
            {
                self.arena.mark_resident(*reference)?;
            }
        }
        self.arena.finish_retirement()
    }

    /// A catch edge abandons the throwing native site, including every
    /// partially prepared mutation. None of these cursors may be replayed at
    /// the handler's next site, even when that site has the same MIR shape.
    pub fn abandon_site_continuations(&mut self) -> NativeExecutorResult<()> {
        let mut reference_error = None;
        self.pending_place_mutation = None;
        self.pending_assign = None;
        self.pending_place_prepare = None;
        self.pending_plain_multi = None;
        self.pending_sequence_assign = None;
        self.pending_sequence_rvalue = None;
        if let Some(mut pending) = self.pending_multi_preparation.take() {
            if let Err(error) = pending.release_fixed_original(&mut self.arena) {
                reference_error = Some(error);
            }
        }
        self.sequence_assignment_register = None;
        if let Some(mut prepared) = self.prepared_multi_assignment.take() {
            if let Err(error) = prepared.release_fixed_original(&mut self.arena) {
                if reference_error.is_none() {
                    reference_error = Some(error);
                }
            }
        }
        self.pending_call = None;
        self.completed_call = None;
        self.retirement_failure = None;
        self.pending_await = None;
        self.abandon_captured_sequences();
        self.contextual_index_extents.clear();
        self.subscript_end_receivers.clear();
        let finished = self.finish_contextual_site();
        reference_error.map_or(finished, Err)
    }

    pub fn refresh_roots(&mut self) -> NativeExecutorResult<NativeRootSet> {
        // Count the actual live owner projection, not an arena highwater or a
        // copied workspace. No root token escapes this borrowed census.
        let mut count = 0usize;
        self.visit_roots(|_| count += 1);
        u32::try_from(count)
            .map_err(|_| NativeExecutorError::Host("native root cardinality overflow".into()))?;
        self.roots.reserve(&self.runtime, count)?;
        // Keep the original buffer and claim together while borrowing the
        // typed owners. Capacity is admitted before any existing ABI view can
        // be displaced; filling is allocation-free and cannot fail.
        let mut roots = std::mem::take(&mut self.roots);
        roots.values.clear();
        self.visit_roots(|root| roots.values.push(root));
        self.roots = roots;
        Ok(self.roots.abi_view())
    }

    fn visit_roots(&self, mut visit: impl FnMut(NativeRoot)) {
        for (slot, value) in self.locals.iter().enumerate() {
            visit(NativeRoot {
                value: *value,
                kind: NativeRootKind::LOCAL,
                slot: slot as u32,
            });
        }
        let mut next_slot = self.locals.len();
        if let Some(value) = self.last_expression {
            let slot = next_slot as u32;
            visit(NativeRoot {
                value,
                kind: NativeRootKind::TEMPORARY,
                slot,
            });
            next_slot += 1;
        }
        if let Some(sequence) = &self.sequence_assignment_register {
            for (slot, value) in sequence.root_references() {
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        if let Some(sequence) = &self.pending_sequence_rvalue {
            for value in sequence.root_references() {
                let slot = next_slot as u32;
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        if let Some(sequence) = &self.pending_sequence_assign {
            for value in sequence.root_references() {
                let slot = next_slot as u32;
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        if let Some(assignment) = &self.prepared_multi_assignment {
            for (slot, value) in assignment.root_references() {
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        if let Some(preparation) = &self.pending_multi_preparation {
            for value in preparation.root_references() {
                let slot = next_slot as u32;
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        if let Some(assignment) = &self.pending_assign {
            for value in assignment.root_references() {
                let slot = next_slot as u32;
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        if let Some(preparation) = &self.pending_place_prepare {
            for value in preparation.root_references() {
                let slot = next_slot as u32;
                visit(NativeRoot {
                    value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
        for value in self.expression_region_roots() {
            let slot = next_slot as u32;
            visit(NativeRoot {
                value,
                kind: NativeRootKind::TEMPORARY,
                slot,
            });
            next_slot += 1;
        }
        // Borrow only the completed owner so the canonical root buffer can
        // be updated without copying the numeric continuation's inline slots.
        for value in
            super::super::call_suspension::completed_roots_from(self.completed_call.as_ref())
        {
            let slot = next_slot as u32;
            visit(NativeRoot {
                value,
                kind: NativeRootKind::TEMPORARY,
                slot,
            });
            next_slot += 1;
        }
        if let Some(pending) = &self.pending_call {
            for value in &pending.retained_roots {
                let slot = next_slot as u32;
                visit(NativeRoot {
                    value: *value,
                    kind: NativeRootKind::TEMPORARY,
                    slot,
                });
                next_slot += 1;
            }
        }
    }

    pub fn prepare_resume(
        &mut self,
        resume: runmat_runtime::native::NativeResumeState,
    ) -> NativeExecutorResult<()> {
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
            .map_err(|error| NativeExecutorError::Host(error.to_string()))?;
        if target.function != self.function.portable_id.0
            || !self
                .function
                .expected_sites
                .iter()
                .any(|site| native_site_matches(site, target))
        {
            return Err(NativeExecutorError::Host(
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

    pub fn arm_deoptimization_fault(&mut self, fault: FaultInjection) {
        self.deoptimization.inject = Some(fault);
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

    pub fn deoptimization_transients_live(&self) -> bool {
        self.pending_await.is_some()
            || self.pending_call.is_some()
            || self.completed_call.is_some()
            || self.pending_place_mutation.is_some()
            || self.pending_assign.is_some()
            || self.pending_place_prepare.is_some()
            || self.pending_plain_multi.is_some()
            || self.pending_sequence_assign.is_some()
            || self.pending_sequence_rvalue.is_some()
            || self.sequence_assignment_register.is_some()
            || self.prepared_multi_assignment.is_some()
            || self.pending_multi_preparation.is_some()
            || self.captured_sequences_live()
            || self.has_contextual_progress()
    }

    pub fn materialize_deoptimization(
        &mut self,
        frame: &runmat_native_codegen::NativeFrameState,
        site: &runmat_native_codegen::NativeMirSite,
    ) -> NativeExecutorResult<MaterializedFrame> {
        if frame.point.function != self.function.portable_id
            || site.point.function != self.function.portable_id
        {
            return Err(NativeExecutorError::Host(
                "native deoptimization frame differs from the active function's portable owner"
                    .into(),
            ));
        }
        if self.deoptimization_transients_live() {
            return Err(NativeExecutorError::Host(
                "native deoptimization cannot materialize while contextual execution state is live"
                    .into(),
            ));
        }
        let bytecode_pc = self
            .interpreter_resume_supported(site)
            .then(|| self.interpreter_resume_points.get(&frame.point).copied())
            .flatten();
        let mut materialized = MaterializedFrame::from_native(
            frame,
            NativeMaterializationContext {
                compiled_function: self.function.id,
                phase: site.phase,
                ordinal: site.ordinal,
                bytecode_pc,
                supplied_inputs: self.supplied_inputs,
                requested_outputs: self.requested_outputs,
                missing_input_locals: self.missing_input_locals.clone(),
                global_bindings: self.global_bindings.snapshot().into_iter().collect(),
                // Filled once after canonical local refresh below.
                persistent_bindings: Default::default(),
            },
            |value| {
                let Some(reference) = self.values.get(&value).copied() else {
                    return Err(NativeExecutorError::Host(format!(
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
        // A nested invocation can change a global or persistent value after
        // this frame's last SSA definition. Override only its materialized local, leaving any other
        // local or already-evaluated operand sharing that SSA value untouched.
        for local in &mut materialized.locals {
            if let Some(reference) = self.declared_session_reference(local.local.0 as usize)? {
                local.value = if reference.is_null() {
                    None
                } else {
                    Some(self.arena.get(reference)?.clone())
                };
            }
        }
        // Refreshing a declared local also observes function invalidation.
        // Capture that same binding state alongside the refreshed locals;
        // evaluated operands above remain their original SSA snapshots.
        materialized.persistent_bindings = self.persistent_bindings.clone();
        self.pending_deoptimization = Some(materialized.clone());
        Ok(materialized)
    }

    pub fn take_deoptimization(&mut self) -> NativeExecutorResult<MaterializedFrame> {
        self.pending_deoptimization.take().ok_or_else(|| {
            NativeExecutorError::Host("native deoptimization has no materialized frame".into())
        })
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
        if self.deoptimization_transients_live()
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

    pub fn next_suspension_identity(&mut self) -> NativeExecutorResult<(u64, u64)> {
        if self.pending_await.is_some() || self.pending_call.is_some() {
            return Err(NativeExecutorError::Host(
                "native invocation already has a pending suspension".into(),
            ));
        }
        let continuation = self.next_await_continuation;
        self.next_await_continuation =
            self.next_await_continuation.checked_add(1).ok_or_else(|| {
                NativeExecutorError::Host("native suspension identity exhausted".into())
            })?;
        Ok((continuation, 1))
    }

    pub fn enter_exception_handler(
        &mut self,
        try_edge: &NativeEdge,
        catch_edge: &NativeEdge,
    ) -> NativeExecutorResult<()> {
        let try_reachable = self.reachable_blocks(try_edge.target)?;
        let catch_reachable = self.reachable_blocks(catch_edge.target)?;
        let protected_blocks = try_reachable
            .difference(&catch_reachable)
            .copied()
            .collect::<BTreeSet<_>>();
        if !protected_blocks.contains(&try_edge.target) {
            return Err(NativeExecutorError::Host(
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
    ) -> NativeExecutorResult<runmat_runtime::native::NativeSiteRequest> {
        let site = self
            .function
            .expected_sites
            .iter()
            .find(|site| site.point.block == block.0)
            .ok_or_else(|| {
                NativeExecutorError::Host("native resume block has no verified site".into())
            })?;
        Ok(runmat_runtime::native::NativeSiteRequest {
            function: self.function.portable_id.0,
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
    ) -> NativeExecutorResult<bool> {
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
            let source = self.runtime.program_source_id(self.function.source);
            error.context.call_frames.push(runmat_runtime::CallFrame {
                function: self.function.name.clone(),
                source_id: source.map(|source| source.0),
                span,
                source_owner: source.and_then(|_| {
                    self.runtime
                        .call_frame_source_owner_for_program_source(self.function.source)
                }),
                diagnostic_source: source
                    .and_then(|source| {
                        runmat_runtime::source_context::source_info_in_context(
                            &self.runtime,
                            source,
                        )
                    })
                    .map(|info| {
                        runmat_runtime::CallFrameDiagnosticSource::new(info.name, info.text)
                    }),
            });
        }
        error
    }
}
