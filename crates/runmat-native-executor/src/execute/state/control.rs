use super::*;

impl HostState {
    /// Resolve catch authority from the shared MIR-authored NativeBlock scope.
    pub fn active_catch_exception(&self) -> Option<std::rc::Rc<runmat_value::MException>> {
        self.current_block
            .and_then(|current| {
                self.function
                    .blocks
                    .iter()
                    .find(|block| block.id == current)
            })
            .and_then(|block| block.catch_scopes.last())
            .and_then(|scope| self.catch_exceptions.get(scope).cloned())
            .or_else(|| self.inherited_catch.clone())
    }

    pub fn bind_catch_exception(
        &mut self,
        block: NativeBlockId,
        exception: runmat_value::MException,
    ) -> NativeExecutorResult<()> {
        let scope = self
            .function
            .blocks
            .iter()
            .find(|candidate| candidate.id == block)
            .and_then(|block| block.catch_scopes.last())
            .copied()
            .ok_or_else(|| {
                NativeExecutorError::Host(
                    "native catch edge has no MIR-authored lexical catch scope".into(),
                )
            })?;
        self.catch_exceptions
            .insert(scope, std::rc::Rc::new(exception));
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

    pub fn take_for_loop(&mut self, header: NativeBlockId) -> NativeExecutorResult<ActiveForLoop> {
        self.active_for_loops
            .remove(&header)
            .ok_or_else(|| NativeExecutorError::Host("native for-loop state is unavailable".into()))
    }

    pub fn restore_for_loop(
        &mut self,
        header: NativeBlockId,
        active: ActiveForLoop,
    ) -> NativeExecutorResult<()> {
        if self.active_for_loops.contains_key(&header) {
            return Err(NativeExecutorError::Host(
                "native for-loop state was restored twice".into(),
            ));
        }
        self.active_for_loops.insert(header, active);
        Ok(())
    }

    pub fn observe_loop_backedge(&mut self, point: runmat_types::ProgramPointId) {
        let count = self.loop_backedges.entry(point).or_default();
        *count = count.saturating_add(1);
    }

    pub fn loop_backedges(&self) -> BTreeMap<runmat_types::ProgramPointId, u64> {
        self.loop_backedges.clone()
    }

    pub fn take_osr_request(&mut self, point: runmat_types::ProgramPointId) -> bool {
        if self.osr_point == Some(point) {
            self.osr_point = None;
            self.osr_entry = Some(point);
            true
        } else {
            false
        }
    }

    pub fn osr_entry(&self) -> Option<runmat_types::ProgramPointId> {
        self.osr_entry
    }

    pub fn optimized_region(
        &self,
        request: runmat_runtime::native::NativeSiteRequest,
    ) -> Option<crate::region::OptimizedRegionPlanLease> {
        self.optimized_regions
            .get(&optimized_site_identity(request))
            .cloned()
    }

    pub fn optimized_region_inputs(
        &mut self,
        plan: &crate::region::OptimizedRegionPlan,
    ) -> NativeExecutorResult<Option<crate::region::NumericRegionInputViews<'_>>> {
        // Session bindings may replace local references while refreshing. Finish
        // every mutable authority lookup before borrowing values from the arena.
        let references = self.optimized_region_input_references(plan)?;
        let mut unavailable = false;
        for reference in references.iter() {
            if reference.is_null() {
                unavailable = true;
            }
        }
        if unavailable {
            return Ok(None);
        }
        crate::region::NumericRegionInputViews::try_from_iter(
            references
                .iter()
                .map(|reference| self.arena.get(*reference)),
        )
        .map(Some)
    }

    pub fn optimized_region_input_references(
        &mut self,
        plan: &crate::region::OptimizedRegionPlan,
    ) -> NativeExecutorResult<crate::region::NumericRegionReferences> {
        if plan.inputs.len() > runmat_runtime::numeric_region::MAX_NODES {
            return Err(NativeExecutorError::Host(
                "numeric region input count exceeds validated program limit".into(),
            ));
        }
        let references = crate::region::NumericRegionReferences::try_from_iter(
            plan.inputs
                .iter()
                .map(|value| self.local_reference(value.local as usize)),
        )?;
        for reference in references.iter() {
            if !reference.is_null() {
                // A stale non-NULL reference remains a real execution error,
                // even when another required input is still unassigned.
                self.arena.get(*reference)?;
            }
        }
        Ok(references)
    }

    pub fn publish_optimized_region<I>(
        &mut self,
        plan: &crate::region::OptimizedRegionPlan,
        computed_outputs: I,
    ) -> NativeExecutorResult<()>
    where
        I: IntoIterator<Item = runmat_value::Value>,
        I::IntoIter: ExactSizeIterator,
    {
        let input_references = crate::region::NumericRegionReferences::try_from_iter(
            plan.inputs.iter().map(|input| {
                self.locals
                    .get(input.local as usize)
                    .copied()
                    .ok_or_else(|| {
                        NativeExecutorError::Host("numeric region input local is invalid".into())
                    })
            }),
        )?;
        self.publish_optimized_region_from_inputs(plan, &input_references, computed_outputs)
    }

    /// An asynchronous region publishes aliases from its original input
    /// generations, never from locals refreshed by workspace reconciliation.
    pub fn publish_optimized_region_from_inputs<I>(
        &mut self,
        plan: &crate::region::OptimizedRegionPlan,
        input_references: &[NativeValueRef],
        computed_outputs: I,
    ) -> NativeExecutorResult<()>
    where
        I: IntoIterator<Item = runmat_value::Value>,
        I::IntoIter: ExactSizeIterator,
    {
        let computed_outputs = computed_outputs.into_iter();
        if computed_outputs.len() != plan.program.outputs.len()
            || input_references.len() != plan.inputs.len()
            || computed_outputs.len() > runmat_runtime::numeric_region::MAX_OUTPUTS
        {
            return Err(NativeExecutorError::Host(
                "numeric region input/output count is inconsistent".into(),
            ));
        }
        // Validate the complete alias vector before mutating any local/SSA
        // output or marking any original instruction as skipped.
        for reference in input_references {
            self.arena.get(*reference)?;
        }
        // Validate every index before allocating/publishing any output slot.
        for output in &plan.outputs {
            let valid = match output.source {
                crate::region::RegionOutputSource::Input(index) => index < input_references.len(),
                crate::region::RegionOutputSource::Computed(index) => {
                    index < computed_outputs.len()
                }
            };
            if !valid {
                return Err(NativeExecutorError::Host(
                    "numeric region publication index is invalid".into(),
                ));
            }
        }
        let mut computed_references =
            [NativeValueRef::NULL; runmat_runtime::numeric_region::MAX_OUTPUTS];
        for (slot, value) in computed_references.iter_mut().zip(computed_outputs) {
            *slot = self.arena.insert(value, Some(&self.runtime))?;
        }
        for output in &plan.outputs {
            let reference = match output.source {
                crate::region::RegionOutputSource::Input(index) => {
                    *input_references.get(index).ok_or_else(|| {
                        NativeExecutorError::Host("numeric region input is invalid".into())
                    })?
                }
                crate::region::RegionOutputSource::Computed(index) => {
                    *computed_references.get(index).ok_or_else(|| {
                        NativeExecutorError::Host("numeric region result is invalid".into())
                    })?
                }
            };
            if let Some(local) = output.value {
                self.set_local(local.local as usize, reference)?;
            }
            self.values.insert(output.ssa, reference);
        }
        self.skipped_optimized_sites
            .extend(plan.skipped_sites.iter().copied());
        self.vectorized_regions = self.vectorized_regions.saturating_add(1);
        Ok(())
    }

    pub fn vectorized_regions(&self) -> u64 {
        self.vectorized_regions
    }

    pub fn skip_optimized_site(
        &mut self,
        request: runmat_runtime::native::NativeSiteRequest,
    ) -> bool {
        self.skipped_optimized_sites
            .remove(&optimized_site_identity(request))
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

    pub(super) fn reachable_blocks(
        &self,
        start: NativeBlockId,
    ) -> NativeExecutorResult<BTreeSet<NativeBlockId>> {
        if !self.function.blocks.iter().any(|block| block.id == start) {
            return Err(NativeExecutorError::Host(
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
                .ok_or_else(|| {
                    NativeExecutorError::Host("native CFG block is unavailable".into())
                })?;
            pending.extend(successor_blocks(&block.terminator.kind));
        }
        Ok(reachable)
    }
}
