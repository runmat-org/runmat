use super::*;

impl HostState {
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
        if target.function != self.function.id.0
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
    ) -> NativeExecutorResult<MaterializedFrame> {
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

    pub fn next_await_identity(&mut self) -> NativeExecutorResult<(u64, u64)> {
        if self.pending_await.is_some() {
            return Err(NativeExecutorError::Host(
                "native invocation already has a pending await".into(),
            ));
        }
        let continuation = self.next_await_continuation;
        self.next_await_continuation = self
            .next_await_continuation
            .checked_add(1)
            .ok_or_else(|| NativeExecutorError::Host("native await identity exhausted".into()))?;
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
            error.context.call_frames.push(runmat_runtime::CallFrame {
                function: self.function.name.clone(),
                source_id: Some(self.function.source.0 as usize),
                span,
            });
        }
        error
    }
}
