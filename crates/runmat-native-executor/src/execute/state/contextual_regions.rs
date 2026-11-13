use allocator_api2::vec::Vec as AllocatedVec;

use runmat_runtime::context::host_allocation::{
    FrameCollectionGate, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    OperationCollectionAllocator,
};
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::HostState;

#[path = "contextual_regions/calls.rs"]
mod calls;
#[path = "contextual_regions/captures.rs"]
mod captures;
#[path = "contextual_regions/identity.rs"]
mod identity;
#[path = "contextual_regions/ordered_cache.rs"]
mod ordered_cache;
#[path = "contextual_regions/references.rs"]
mod references;
#[path = "contextual_regions/subscripts.rs"]
mod subscripts;

pub(crate) use identity::EmbeddedOperationIdentity;
use ordered_cache::{prepare_vector_push, OrderedCache};
use references::{CachedRefs, CachedRefsBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum EmbeddedOperationSegment {
    RegionStep { region: usize, step: usize },
    Call { ordinal: usize },
}

pub(super) struct OperationScope {
    pub(super) segment: EmbeddedOperationSegment,
    pub(super) next_call: usize,
}

struct ContinuationMetadata {
    gate: FrameCollectionGate,
    operation_path: AllocatedVec<OperationScope, OperationCollectionAllocator>,
    completed_operations: OrderedCache<EmbeddedOperationIdentity, CachedRefs>,
    operation_region_ends: OrderedCache<EmbeddedOperationIdentity, usize>,
    prepared_subscript_receivers: OrderedCache<
        (Option<EmbeddedOperationIdentity>, usize),
        (
            runmat_runtime::object::protocol::PreparedSubscriptReceiver,
            NativeValueRef,
        ),
    >,
    captured_sequences:
        OrderedCache<runmat_mir::MirSequenceLocalId, captures::CapturedSequenceSlot>,
}

impl ContinuationMetadata {
    fn new(gate: FrameCollectionGate) -> Self {
        let allocator = gate.allocator();
        Self {
            operation_path: AllocatedVec::new_in(allocator.clone()),
            completed_operations: OrderedCache::new(allocator.clone()),
            operation_region_ends: OrderedCache::new(allocator.clone()),
            prepared_subscript_receivers: OrderedCache::new(allocator.clone()),
            captured_sequences: OrderedCache::new(allocator),
            gate,
        }
    }
}

#[derive(Default)]
pub(super) struct ExpressionRegionState {
    site: Option<NativeSiteRequest>,
    next_region: usize,
    regions: Vec<RegionProgress>,
    retained_inputs: Vec<NativeValueRef>,
    retained_region_results: Vec<NativeValueRef>,
    // Declared after the exact-capacity buffers so their physical blocks are
    // freed before the managed ledger is refunded.
    _allocation_claim: Option<ExactHostAllocationClaim>,
    next_retained_input: usize,
    metadata: Option<ContinuationMetadata>,
}

#[derive(Default)]
struct RegionProgress {
    next_step: usize,
    result: Option<CachedRefs>,
    owns_result_references: bool,
    next_region: usize,
}

impl ExpressionRegionState {
    fn metadata(&self) -> NativeExecutorResult<&ContinuationMetadata> {
        self.metadata.as_ref().ok_or_else(|| {
            NativeExecutorError::Host(
                "pure native function entered an expression continuation".into(),
            )
        })
    }

    fn metadata_mut(&mut self) -> NativeExecutorResult<&mut ContinuationMetadata> {
        self.metadata.as_mut().ok_or_else(|| {
            NativeExecutorError::Host(
                "pure native function entered an expression continuation".into(),
            )
        })
    }

    pub(super) fn prepare(
        runtime: &runmat_runtime::context::RuntimeContext,
        geometry: runmat_native_codegen::NativeExpressionContinuationGeometry,
    ) -> NativeExecutorResult<Self> {
        let region_capacity = usize::try_from(geometry.region_capacity).map_err(|_| {
            NativeExecutorError::Host("native expression-region capacity overflow".into())
        })?;
        let retained_input_capacity =
            usize::try_from(geometry.retained_input_capacity).map_err(|_| {
                NativeExecutorError::Host("native retained-input capacity overflow".into())
            })?;
        let bytes = std::mem::size_of::<RegionProgress>()
            .checked_mul(region_capacity)
            .and_then(|regions| {
                std::mem::size_of::<NativeValueRef>()
                    .checked_mul(retained_input_capacity)
                    .and_then(|inputs| regions.checked_add(inputs))
            })
            .and_then(|bytes| {
                std::mem::size_of::<NativeValueRef>()
                    .checked_mul(region_capacity)
                    .and_then(|results| bytes.checked_add(results))
            })
            .ok_or_else(|| {
                NativeExecutorError::Host("native expression continuation size overflow".into())
            })?;
        if bytes == 0 {
            return Ok(Self::default());
        }
        let buffers_retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| {
                NativeExecutorError::Host("native expression continuation size overflow".into())
            })?,
            1,
        )
        .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let retained = buffers_retained
            .checked_add(
                FrameCollectionGate::retained_bytes()
                    .map_err(|cause| continuation_admission_error(runtime, cause))?,
            )
            .ok_or_else(|| continuation_admission_error(runtime, HostAllocationError::Overflow))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        match HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|cause| continuation_admission_error(runtime, cause))?
        {
            Some(receipt) => {
                let gate = FrameCollectionGate::prepare(runtime, &receipt)
                    .map_err(|cause| continuation_admission_error(runtime, cause))?;
                receipt
                    .with_output_allocation(runtime, |permit| {
                        Self::allocate(
                            region_capacity,
                            retained_input_capacity,
                            bytes,
                            Some(permit),
                            gate,
                        )
                    })
                    .map_err(|cause| continuation_admission_error(runtime, cause))
            }
            None => Self::allocate(
                region_capacity,
                retained_input_capacity,
                bytes,
                None,
                FrameCollectionGate::unbounded(),
            )
            .map_err(|cause| {
                NativeExecutorError::Host(format!(
                    "native expression continuation allocation failed: {cause}"
                ))
            }),
        }
    }

    fn allocate(
        region_capacity: usize,
        retained_input_capacity: usize,
        bytes: usize,
        permit: Option<&dyn ExactHostAllocationPermit>,
        gate: FrameCollectionGate,
    ) -> Result<Self, runmat_value::ExactHostNumericAllocationError> {
        let allocation_claim = match (bytes, permit) {
            (0, _) | (_, None) => None,
            (_, Some(permit)) => Some(permit.reserve_exact_claim(bytes)?),
        };
        let regions = exact_capacity_vec(region_capacity)?;
        let retained_inputs = exact_capacity_vec(retained_input_capacity)?;
        let retained_region_results = exact_capacity_vec(region_capacity)?;
        Ok(Self {
            regions,
            retained_inputs,
            retained_region_results,
            _allocation_claim: allocation_claim,
            metadata: Some(ContinuationMetadata::new(gate)),
            ..Self::default()
        })
    }
}

fn continuation_admission_error(
    runtime: &runmat_runtime::context::RuntimeContext,
    cause: HostAllocationError,
) -> NativeExecutorError {
    NativeExecutorError::from(crate::error::allocation_error(
        Some(runtime),
        format!("native expression continuation could not be admitted: {cause}"),
        cause,
    ))
}

impl HostState {
    pub fn has_contextual_progress(&self) -> bool {
        self.contextual_regions
            .metadata
            .as_ref()
            .is_some_and(|metadata| {
                !metadata.prepared_subscript_receivers.is_empty()
                    || !metadata.completed_operations.is_empty()
            })
            || !self.contextual_regions.retained_inputs.is_empty()
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
                self.contextual_regions.next_retained_input = 0;
                if let Some(metadata) = self.contextual_regions.metadata.as_mut() {
                    metadata.operation_path.clear();
                }
            }
            None => {
                self.contextual_regions.site = Some(site);
                self.contextual_regions.next_retained_input = 0;
            }
        }
        Ok(())
    }

    /// Returns the next source-ordered input retained by an earlier attempt at
    /// this exact native site. Callers must consume these at the same MIR
    /// positions at which they were first recorded.
    pub fn take_contextual_input(&mut self) -> Option<NativeValueRef> {
        let input = self
            .contextual_regions
            .retained_inputs
            .get(self.contextual_regions.next_retained_input)
            .copied()?;
        self.contextual_regions.next_retained_input += 1;
        Some(input)
    }

    /// Retains a newly evaluated source input before a prepared expression can
    /// suspend or mutate its source binding.
    pub fn retain_contextual_input(
        &mut self,
        input: NativeValueRef,
    ) -> NativeExecutorResult<NativeValueRef> {
        if self.contextual_regions.next_retained_input
            != self.contextual_regions.retained_inputs.len()
        {
            return Err(NativeExecutorError::Host(
                "native contextual-input traversal diverged from its retained prefix".into(),
            ));
        }
        if self.contextual_regions.retained_inputs.len()
            == self.contextual_regions.retained_inputs.capacity()
        {
            return Err(NativeExecutorError::Host(
                "verified native contextual-input capacity was exhausted".into(),
            ));
        }
        self.arena.retain(input)?;
        self.contextual_regions.retained_inputs.push(input);
        self.contextual_regions.next_retained_input += 1;
        Ok(input)
    }

    pub fn enter_contextual_region(
        &mut self,
    ) -> NativeExecutorResult<(usize, usize, Option<CachedRefs>)> {
        let ordinal = self.contextual_regions.next_region;
        self.contextual_regions.next_region += 1;
        if self.contextual_regions.regions.len() == ordinal {
            if self.contextual_regions.regions.len() == self.contextual_regions.regions.capacity() {
                return Err(NativeExecutorError::Host(
                    "verified native expression-region capacity was exhausted".into(),
                ));
            }
            self.contextual_regions.regions.push(RegionProgress {
                next_region: ordinal + 1,
                ..RegionProgress::default()
            });
        }
        let progress = &self.contextual_regions.regions[ordinal];
        self.contextual_regions.next_region = progress.next_region;
        Ok((ordinal, progress.next_step, progress.result.clone()))
    }

    pub fn enter_contextual_step(
        &mut self,
        region: usize,
        step: usize,
    ) -> NativeExecutorResult<()> {
        let metadata = self.contextual_regions.metadata_mut()?;
        prepare_vector_push(&mut metadata.operation_path, &self.runtime, &metadata.gate)?;
        metadata.operation_path.push(OperationScope {
            segment: EmbeddedOperationSegment::RegionStep { region, step },
            next_call: 0,
        });
        Ok(())
    }

    pub fn finish_contextual_step(
        &mut self,
        region: usize,
        step: usize,
        next_step: usize,
    ) -> NativeExecutorResult<()> {
        self.pop_operation(EmbeddedOperationSegment::RegionStep { region, step })?;
        self.contextual_regions.regions[region].next_step = next_step;
        self.contextual_regions.regions[region].next_region = self.contextual_regions.next_region;
        Ok(())
    }

    pub fn finish_contextual_region(&mut self, region: usize, result: NativeValueRef) {
        self.contextual_regions.regions[region].result = Some(CachedRefs::One(result));
        self.contextual_regions.regions[region].owns_result_references = false;
        self.contextual_regions.regions[region].next_region = self.contextual_regions.next_region;
    }

    /// Completes a region whose result references were freshly inserted into
    /// the arena solely for resumable region replay. Unlike borrowed local or
    /// mutation results, these references have no owner after the site ends.
    pub fn finish_owned_contextual_region(&mut self, region: usize, result: CachedRefs) {
        self.contextual_regions.regions[region].result = Some(result);
        self.contextual_regions.regions[region].owns_result_references = true;
        self.contextual_regions.regions[region].next_region = self.contextual_regions.next_region;
    }

    pub fn finish_owned_contextual_values(
        &mut self,
        region: usize,
        values: &[runmat_value::Value],
    ) -> NativeExecutorResult<()> {
        let mut references = CachedRefsBuilder::prepare(&self.runtime, values.len())?;
        for value in values {
            let reference = self.arena.insert(value.clone(), Some(&self.runtime))?;
            if let Err(error) = references.push_initial(reference) {
                let _ = self.arena.release(reference);
                for prior in references.as_slice() {
                    let _ = self.arena.release(*prior);
                }
                return Err(error);
            }
        }
        let references = match references.finish() {
            Ok(references) => references,
            Err(error) => {
                for reference in references.as_slice() {
                    let _ = self.arena.release(*reference);
                }
                return Err(error);
            }
        };
        self.finish_owned_contextual_region(region, references);
        Ok(())
    }

    #[cfg(test)]
    pub fn finish_owned_contextual_reference(&mut self, region: usize, reference: NativeValueRef) {
        self.finish_owned_contextual_region(region, CachedRefs::One(reference));
    }

    pub fn retain_expression_region_result(
        &mut self,
        result: NativeValueRef,
    ) -> NativeExecutorResult<()> {
        if self.contextual_regions.retained_region_results.len()
            == self.contextual_regions.retained_region_results.capacity()
        {
            return Err(NativeExecutorError::Host(
                "verified native expression-region result capacity was exhausted".into(),
            ));
        }
        self.arena.retain(result)?;
        self.contextual_regions.retained_region_results.push(result);
        Ok(())
    }

    pub fn expression_region_roots(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.contextual_regions
            .regions
            .iter()
            .flat_map(|progress| {
                progress
                    .result
                    .iter()
                    .flat_map(|values| values.as_slice().iter().copied())
            })
            .chain(
                self.contextual_regions
                    .metadata
                    .iter()
                    .flat_map(|metadata| metadata.completed_operations.values())
                    .flat_map(|values| values.as_slice().iter().copied()),
            )
            .chain(
                self.contextual_regions
                    .metadata
                    .iter()
                    .flat_map(|metadata| metadata.prepared_subscript_receivers.values())
                    .map(|(_, receiver)| *receiver),
            )
            .chain(self.contextual_regions.retained_inputs.iter().copied())
    }

    pub fn finish_contextual_site(&mut self) -> NativeExecutorResult<()> {
        let mut first_error = None;
        for input in self.contextual_regions.retained_inputs.drain(..) {
            if let Err(error) = self.arena.release(input) {
                first_error.get_or_insert(error);
            }
        }
        for result in self.contextual_regions.retained_region_results.drain(..) {
            if let Err(error) = self.arena.release(result) {
                first_error.get_or_insert(error);
            }
        }
        for progress in &self.contextual_regions.regions {
            if progress.owns_result_references {
                for result in progress.result.iter().flat_map(CachedRefs::as_slice) {
                    if let Err(error) = self.arena.release(*result) {
                        first_error.get_or_insert(error);
                    }
                }
            }
        }
        if let Some(metadata) = self.contextual_regions.metadata.as_ref() {
            for references in metadata.completed_operations.values() {
                for reference in references.as_slice() {
                    if let Err(error) = self.arena.release(*reference) {
                        first_error.get_or_insert(error);
                    }
                }
            }
            for (_, receiver) in metadata.prepared_subscript_receivers.values() {
                if let Err(error) = self.arena.release(*receiver) {
                    first_error.get_or_insert(error);
                }
            }
        }
        self.contextual_regions.site = None;
        self.contextual_regions.next_region = 0;
        self.contextual_regions.next_retained_input = 0;
        self.contextual_regions.regions.clear();
        if let Some(metadata) = self.contextual_regions.metadata.as_mut() {
            metadata.operation_path.clear();
            metadata.completed_operations.clear_in_source_order();
            metadata.operation_region_ends.clear_in_source_order();
            metadata
                .prepared_subscript_receivers
                .clear_in_source_order();
        }
        first_error.map_or(Ok(()), Err)
    }

    pub fn abandon_contextual_region(
        &mut self,
        sequences: impl Iterator<Item = runmat_mir::MirSequenceLocalId>,
    ) -> NativeExecutorResult<()> {
        for sequence in sequences {
            self.discard_captured_sequence(sequence);
        }
        self.finish_contextual_site()
    }

    pub(super) fn pop_operation(
        &mut self,
        expected: EmbeddedOperationSegment,
    ) -> NativeExecutorResult<()> {
        let actual = self
            .contextual_regions
            .metadata_mut()?
            .operation_path
            .pop()
            .map(|scope| scope.segment);
        if actual.as_ref() != Some(&expected) {
            self.contextual_regions
                .metadata_mut()?
                .operation_path
                .clear();
            return Err(NativeExecutorError::Host(
                "contextual selector operation nesting is invalid".into(),
            ));
        }
        Ok(())
    }
}
