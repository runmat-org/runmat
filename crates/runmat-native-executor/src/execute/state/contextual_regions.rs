use std::collections::BTreeMap;

use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::HostState;

#[path = "contextual_regions/calls.rs"]
mod calls;
#[path = "contextual_regions/subscripts.rs"]
mod subscripts;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EmbeddedOperationIdentity {
    path: Vec<EmbeddedOperationSegment>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum EmbeddedOperationSegment {
    RegionStep { region: usize, step: usize },
    Call { ordinal: usize },
}

pub(super) struct OperationScope {
    pub(super) segment: EmbeddedOperationSegment,
    pub(super) next_call: usize,
}

#[derive(Default)]
pub(super) struct ContextualRegionState {
    site: Option<NativeSiteRequest>,
    next_region: usize,
    regions: Vec<RegionProgress>,
    pub(super) operation_path: Vec<OperationScope>,
    pub(super) completed_operations: BTreeMap<EmbeddedOperationIdentity, Vec<NativeValueRef>>,
    pub(super) prepared_subscript_receivers: BTreeMap<
        (Option<EmbeddedOperationIdentity>, usize),
        (
            runmat_runtime::object::protocol::PreparedSubscriptReceiver,
            NativeValueRef,
        ),
    >,
}

#[derive(Default)]
struct RegionProgress {
    next_step: usize,
    result: Option<NativeValueRef>,
}

impl HostState {
    pub fn has_contextual_progress(&self) -> bool {
        !self
            .contextual_regions
            .prepared_subscript_receivers
            .is_empty()
            || !self.contextual_regions.completed_operations.is_empty()
            || self
                .contextual_regions
                .regions
                .iter()
                .any(|progress| progress.next_step != 0 || progress.result.is_some())
    }

    pub fn begin_contextual_site(&mut self, site: NativeSiteRequest) -> NativeExecutorResult<()> {
        match self.contextual_regions.site {
            Some(active) if active != site => {
                return Err(NativeExecutorError::Host(
                    format!(
                        "contextual selector state escaped its owning native site: active {active:?}, entered {site:?}"
                    ),
                ));
            }
            Some(_) => {
                self.contextual_regions.next_region = 0;
                self.contextual_regions.operation_path.clear();
            }
            None => self.contextual_regions.site = Some(site),
        }
        Ok(())
    }

    pub fn enter_contextual_region(&mut self) -> (usize, usize, Option<NativeValueRef>) {
        let ordinal = self.contextual_regions.next_region;
        self.contextual_regions.next_region += 1;
        if self.contextual_regions.regions.len() == ordinal {
            self.contextual_regions
                .regions
                .push(RegionProgress::default());
        }
        let progress = &self.contextual_regions.regions[ordinal];
        (ordinal, progress.next_step, progress.result)
    }

    pub fn enter_contextual_step(&mut self, region: usize, step: usize) {
        self.contextual_regions.operation_path.push(OperationScope {
            segment: EmbeddedOperationSegment::RegionStep { region, step },
            next_call: 0,
        });
    }

    pub fn finish_contextual_step(
        &mut self,
        region: usize,
        step: usize,
        next_step: usize,
    ) -> NativeExecutorResult<()> {
        self.pop_operation(EmbeddedOperationSegment::RegionStep { region, step })?;
        self.contextual_regions.regions[region].next_step = next_step;
        Ok(())
    }

    pub fn finish_contextual_region(&mut self, region: usize, result: NativeValueRef) {
        self.contextual_regions.regions[region].result = Some(result);
    }

    pub fn contextual_region_roots(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.contextual_regions
            .regions
            .iter()
            .filter_map(|progress| progress.result)
            .chain(
                self.contextual_regions
                    .completed_operations
                    .values()
                    .flat_map(|values| values.iter().copied()),
            )
            .chain(
                self.contextual_regions
                    .prepared_subscript_receivers
                    .values()
                    .map(|(_, receiver)| *receiver),
            )
    }

    pub fn finish_contextual_site(&mut self) {
        self.contextual_regions = ContextualRegionState::default();
    }

    pub fn abandon_contextual_region(
        &mut self,
        sequences: impl Iterator<Item = runmat_mir::MirSequenceLocalId>,
    ) {
        for sequence in sequences {
            self.captured_sequences.remove(&sequence);
        }
        self.contextual_regions = ContextualRegionState::default();
    }

    pub(super) fn pop_operation(
        &mut self,
        expected: EmbeddedOperationSegment,
    ) -> NativeExecutorResult<()> {
        let actual = self
            .contextual_regions
            .operation_path
            .pop()
            .map(|scope| scope.segment);
        if actual.as_ref() != Some(&expected) {
            self.contextual_regions.operation_path.clear();
            return Err(NativeExecutorError::Host(
                "contextual selector operation nesting is invalid".into(),
            ));
        }
        Ok(())
    }
}
