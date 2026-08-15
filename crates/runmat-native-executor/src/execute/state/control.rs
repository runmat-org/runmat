use super::*;

impl HostState {
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

    pub fn for_loop_mut(
        &mut self,
        header: NativeBlockId,
    ) -> NativeExecutorResult<&mut ActiveForLoop> {
        self.active_for_loops
            .get_mut(&header)
            .ok_or_else(|| NativeExecutorError::Host("native for-loop state is unavailable".into()))
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
    ) -> Option<crate::region::OptimizedRegionPlan> {
        self.optimized_regions
            .get(&optimized_site_identity(request))
            .filter(|plan| !self.disabled_optimized_regions.contains(&plan.region))
            .cloned()
    }

    pub fn disable_optimized_region(&mut self, region: runmat_types::RegionId) {
        self.disabled_optimized_regions.insert(region);
    }

    pub fn optimized_region_inputs(
        &self,
        plan: &crate::region::OptimizedRegionPlan,
    ) -> NativeExecutorResult<Vec<&runmat_value::Value>> {
        plan.inputs
            .iter()
            .map(|value| {
                let reference =
                    self.locals
                        .get(value.local as usize)
                        .copied()
                        .ok_or_else(|| {
                            NativeExecutorError::Host(
                                "numeric region input local is invalid".into(),
                            )
                        })?;
                self.arena.get(reference)
            })
            .collect()
    }

    pub fn publish_optimized_region(
        &mut self,
        plan: &crate::region::OptimizedRegionPlan,
        computed_outputs: Vec<runmat_value::Value>,
    ) -> NativeExecutorResult<()> {
        if computed_outputs.len() != plan.program.outputs.len() {
            return Err(NativeExecutorError::Host(
                "numeric region output count is inconsistent".into(),
            ));
        }
        let input_references = plan
            .inputs
            .iter()
            .map(|input| {
                self.locals
                    .get(input.local as usize)
                    .copied()
                    .ok_or_else(|| {
                        NativeExecutorError::Host("numeric region input local is invalid".into())
                    })
            })
            .collect::<NativeExecutorResult<Vec<_>>>()?;
        let computed_references = computed_outputs
            .into_iter()
            .map(|value| self.arena.insert(value))
            .collect::<Vec<_>>();
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
