//! Physical host preparation shared by VM fusion and admitted builtin domains.
//! Domain owners separately certify exact occurrences and bound host operations.
use runmat_accelerate_api::{
    checked_numeric_download_host_payload_bytes, handle_provenance, same_provider_instance,
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, ProviderCapabilitySnapshot,
};
use runmat_execution::{CandidateOutputPresentation, CandidateResourceDemand, ResourceByteBound};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::Value;
use std::borrow::Cow;

#[derive(Debug)]
pub enum CpuReadbackFailure {
    BeforeEffect(RuntimeError),
    AfterEffect(RuntimeError),
}

mod managed_readback;
mod resources;
pub(crate) use resources::resources;

pub(crate) struct CpuHostResources {
    pub resources: CandidateResourceDemand,
    pub presentation: CandidateOutputPresentation,
    pub scratch_bytes: u64,
    pub download_bytes: u64,
    pub readback_count: u64,
}

pub(super) enum CpuHostPlacement<'a> {
    Native {
        cpu: Option<&'a runmat_runtime::numeric_region::service::NativeNumericRegionCpuCandidates>,
        right: &'a runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        source_host: Option<&'a CpuHostResources>,
    },
    Legacy,
    Unavailable,
    Readbacks(&'a CpuHostResources),
}

/// Host quota adds retained output to extra scratch. Preparation finishes
/// before those outputs are produced, so normalize the larger complete phase
/// peak back to extra-only demand rather than charging retained output twice.
pub(crate) fn checked_extra_host_scratch(
    retained: u64,
    preparation_peak: u64,
    body_extra: u64,
) -> Option<u64> {
    preparation_peak
        .max(retained.checked_add(body_extra)?)
        .checked_sub(retained)
}

pub(crate) struct Source {
    pub(crate) handle: GpuTensorHandle,
    pub(crate) owner: &'static dyn AccelProvider,
    pub(crate) offsets: Vec<usize>,
    pub(crate) loads: Vec<super::VmHostLoadUse>,
    pub(crate) capabilities: Cow<'static, ProviderCapabilitySnapshot>,
    pub(crate) payload_bytes: u64,
    pub(crate) host_workspace: u64,
    pub(crate) provider_scratch: ResourceByteBound,
    pub(crate) queue_slots: u32,
}

pub(crate) struct CpuReadbackPlan {
    pub(crate) sources: Vec<Source>,
    pub(crate) stack_len: usize,
    pub(crate) resident_offsets: Vec<usize>,
}

/// Checked upper bound for the request-local metadata built while deciding a
/// host candidate. Download payload and execution scratch have separate
/// physical owners; this counts only the preparation vectors that coexist
/// with the retained semantic programs. Production resident sources borrow
/// their immutable provider capability snapshots.
pub(crate) fn candidate_preparation_bytes(
    resident_inputs: usize,
    entry_uses: usize,
    load_uses: usize,
    operations: usize,
    operand_occurrences: usize,
    liveness_value_ids_upper: usize,
    max_rank: usize,
) -> Option<u64> {
    fn vector<T>(count: usize) -> Option<u64> {
        count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }
    let mut bytes = 0_u64;
    let mut add = |more: u64| {
        bytes = bytes.checked_add(more)?;
        Some(())
    };
    // The two duplicate guards, final resident offset table and one source
    // vector all exist before selection. Each source starts with full transcript
    // capacity so later same-handle extension needs no larger allocation.
    add(vector::<usize>(entry_uses.checked_mul(2)?)?)?;
    add(vector::<usize>(load_uses)?)?;
    add(vector::<Source>(resident_inputs)?)?;
    add(vector::<usize>(entry_uses.checked_mul(resident_inputs)?)?)?;
    add(vector::<super::VmHostLoadUse>(
        load_uses.checked_mul(resident_inputs)?,
    )?)?;
    add(vector::<runmat_execution::ProviderReadbackDemand>(
        resident_inputs,
    )?)?;
    add(u64::try_from(std::mem::size_of::<runmat_execution::ProviderReadbackDemands>()).ok()?)?;
    add(vector::<runmat_runtime::context::VmFusionHostOperation>(
        operations,
    )?)?;
    add(vector::<runmat_builtins::BuiltinBindingIdentity>(
        operations,
    )?)?;
    // Managed VM output envelopes retain every reached instruction's input
    // and output shapes until their carrier set is installed, before readback.
    add(vector::<super::elementwise::host_readback::HostOutputSpec>(
        operations,
    )?)?;
    add(vector::<Vec<usize>>(operand_occurrences)?)?;
    add(vector::<usize>(
        operand_occurrences
            .checked_add(operations)?
            .checked_mul(max_rank)?,
    )?)?;
    // Every occurrence may retain a shape during resource calculation. The
    // current operation also owns its Vec-of-shapes, argument vector and a
    // temporary known-dims projection while constructing the next shape.
    add(vector::<usize>(
        operand_occurrences
            .checked_add(operations)?
            .checked_mul(max_rank)?,
    )?)?;
    add(vector::<Option<usize>>(max_rank)?)?;
    add(vector::<Vec<usize>>(2)?)?;
    add(vector::<
        runmat_runtime::builtins::common::spec::FusionHostResourceInput<'static>,
    >(2)?)?;
    // The canonical host binary calculator constructs a BroadcastPlan while
    // pricing its body. Its ten-vector geometry comes from that owner.
    add(runmat_runtime::builtins::math::host_resources::broadcast_metadata_bytes(max_rank)?)?;
    add(super::elementwise::host_liveness_preparation_bytes(
        liveness_value_ids_upper,
        operations,
        max_rank,
    )?)?;
    Some(bytes)
}

fn unavailable(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Placement:InvalidHostReadback")
        .build()
}

impl Source {
    pub(crate) fn admit(
        handle: &GpuTensorHandle,
        provider: &dyn AccelProvider,
        offsets: Vec<usize>,
        loads: Vec<super::VmHostLoadUse>,
    ) -> Option<Self> {
        let owner =
            runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(handle)?;
        if !same_provider_instance(owner, provider)
            || handle_provenance(handle) != Some(GpuHandleProvenance::Automatic)
            || handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
            || !matches!(
                (handle.view().is_logical(), handle.physical().element_type()),
                (false, runmat_accelerate_api::NumericElementType::F32)
                    | (false, runmat_accelerate_api::NumericElementType::F64)
                    | (true, runmat_accelerate_api::NumericElementType::U8)
                    | (true, runmat_accelerate_api::NumericElementType::F32)
                    | (true, runmat_accelerate_api::NumericElementType::F64)
            )
            || handle.allocation_lease().validate_live().is_err()
            || handle.view().validate(handle.physical()).is_err()
            || offsets.is_empty() && loads.is_empty()
        {
            return None;
        }
        let download =
            runmat_accelerate_api::ProviderFeasibilityQuery::exact_numeric_download(handle)?;
        if !owner
            .query_numeric_download_feasibility(handle, &download)
            .is_supported()
        {
            return None;
        }
        let ResourceByteBound::Known(host_workspace) =
            owner.download_numeric_host_workspace_bound(handle)
        else {
            return None;
        };
        let provider_scratch = owner.download_numeric_provider_scratch_bound(handle);
        if !matches!(provider_scratch, ResourceByteBound::Known(_)) {
            return None;
        }
        Some(Self {
            handle: handle.clone(),
            owner,
            offsets,
            loads,
            capabilities: owner
                .capability_snapshot_ref()
                .map(Cow::Borrowed)
                .unwrap_or_else(|| Cow::Owned(owner.capability_snapshot())),
            payload_bytes: checked_numeric_download_host_payload_bytes(handle)?,
            host_workspace,
            provider_scratch,
            queue_slots: owner.download_numeric_queue_slots(handle),
        })
    }
    pub(crate) fn matches_handle(&self, actual: &GpuTensorHandle) -> bool {
        actual.view_identity() == self.handle.view_identity()
            && actual.wire_descriptor() == self.handle.wire_descriptor()
            && actual.allocation_lease().validate_live().is_ok()
            && actual.view().validate(actual.physical()).is_ok()
            && handle_provenance(actual) == Some(GpuHandleProvenance::Automatic)
            && runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(actual)
                .is_some_and(|owner| same_provider_instance(owner, self.owner))
    }

    fn current(&self) -> bool {
        self.matches_handle(&self.handle)
            && runmat_accelerate_api::ProviderFeasibilityQuery::exact_numeric_download(&self.handle)
                .is_some_and(|query| {
                    self.owner
                        .query_numeric_download_feasibility(&self.handle, &query)
                        .is_supported()
                })
            && self.owner.capability_snapshot_ref().map_or_else(
                || {
                    self.owner
                        .capability_snapshot()
                        .eq(self.capabilities.as_ref())
                },
                |current| current == self.capabilities.as_ref(),
            )
            && checked_numeric_download_host_payload_bytes(&self.handle) == Some(self.payload_bytes)
            && self
                .owner
                .download_numeric_host_workspace_bound(&self.handle)
                == ResourceByteBound::Known(self.host_workspace)
            && self
                .owner
                .download_numeric_provider_scratch_bound(&self.handle)
                == self.provider_scratch
            && self.owner.download_numeric_queue_slots(&self.handle) == self.queue_slots
    }
}

fn source_matches(source: &Source, stack: &[Value]) -> bool {
    source.current() && source.offsets.iter().all(|&offset| {
        matches!(stack.get(offset), Some(Value::GpuTensor(actual)) if source.matches_handle(actual))
    })
}

/// Private host values live beside the selected observation, never in the
/// workspace. Every later ordinary load is revalidated after its normal lexical,
/// global, persistent and undefined-variable checks, before substitution.
pub struct PreparedVmHostInputs {
    pub entry: Vec<(usize, Value)>,
    mirrors: Vec<(Source, Value)>,
    // The selected finite readback's native payload and wrapper remain live
    // through the prepared body, including aliases in its argument vector.
    _managed_readback: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
}

impl PreparedVmHostInputs {
    pub(crate) fn cleanup(
        self,
    ) -> Result<(), runmat_accelerate_api::allocation::GpuHandleReleaseError> {
        release_sources(self.mirrors.into_iter().map(|(source, value)| {
            drop(value);
            source
        }))
    }

    pub fn substitute_load(
        &self,
        pc: usize,
        binding: &crate::graph::VarBinding,
        actual: &Value,
    ) -> Result<Option<Value>, Box<RuntimeError>> {
        let source = self.mirrors.iter().find(|(source, _)| {
            source
                .loads
                .iter()
                .any(|load| load.pc == pc && &load.binding == binding)
        });
        let Some((source, value)) = source else {
            return Ok(None);
        };
        if !source.current()
            || !matches!(actual, Value::GpuTensor(handle) if source.matches_handle(handle))
        {
            return Err(Box::new(unavailable(
                "selected host source changed at its certified ordinary load",
            )));
        }
        Ok(Some(value.clone()))
    }
}

impl CpuReadbackPlan {
    pub(crate) fn cleanup(
        self,
    ) -> Result<(), runmat_accelerate_api::allocation::GpuHandleReleaseError> {
        release_sources(self.sources)
    }
    pub(crate) fn covers_resident_slice(&self, stack: &[Value]) -> bool {
        let occurrence_count: usize = self.sources.iter().map(|source| source.offsets.len()).sum();
        self.resident_offsets.len() == occurrence_count
            && stack.iter().enumerate().all(|(offset, value)| {
                !runmat_runtime::dispatcher::value_contains_gpu(value)
                    || self.resident_offsets.binary_search(&offset).is_ok()
            })
    }

    pub(crate) async fn prepare_inputs(
        self,
        stack: &[Value],
    ) -> Result<PreparedVmHostInputs, CpuReadbackFailure> {
        let values = self.download_sources(stack).await?;
        let mut entry = Vec::new();
        let mut mirrors = Vec::new();
        for (source, value) in self.sources.into_iter().zip(values) {
            entry.extend(source.offsets.iter().map(|&offset| (offset, value.clone())));
            mirrors.push((source, value));
        }
        Ok(PreparedVmHostInputs {
            entry,
            mirrors,
            _managed_readback: None,
        })
    }

    async fn download_sources(&self, stack: &[Value]) -> Result<Vec<Value>, CpuReadbackFailure> {
        if stack.len() != self.stack_len
            || !self.covers_resident_slice(stack)
            || !self
                .sources
                .iter()
                .all(|source| source_matches(source, stack))
        {
            return Err(CpuReadbackFailure::BeforeEffect(unavailable(
                "selected host source no longer matches its compiler-certified stack",
            )));
        }
        let mut collected = Vec::with_capacity(self.sources.len());
        let mut prior_effect = false;
        for source in &self.sources {
            if !source_matches(source, stack) {
                let error = unavailable("selected host source changed during readback");
                return Err(if prior_effect {
                    CpuReadbackFailure::AfterEffect(error)
                } else {
                    CpuReadbackFailure::BeforeEffect(error)
                });
            }
            prior_effect = true;
            let value = runmat_runtime::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                source.owner,
                &source.handle,
            )
            .await
            .map_err(CpuReadbackFailure::AfterEffect)?;
            collected.push(value);
        }
        if !self.covers_resident_slice(stack)
            || !self
                .sources
                .iter()
                .all(|source| source_matches(source, stack))
        {
            return Err(CpuReadbackFailure::AfterEffect(unavailable(
                "selected host source changed before atomic operand publication",
            )));
        }
        Ok(collected)
    }
}

fn release_sources(
    sources: impl IntoIterator<Item = Source>,
) -> Result<(), runmat_accelerate_api::allocation::GpuHandleReleaseError> {
    let mut first = None;
    for source in sources {
        if let Err(error) = source.handle.release() {
            if first.is_none() {
                first = Some(error);
            }
        }
    }
    match first {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
