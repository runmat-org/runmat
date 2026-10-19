use super::*;
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostNumericAllocationError};

/// One exact class-preserving upload shape. The full logical normal result is
/// held by Runtime; a tile borrows that result and retains only one device
/// publication at a time. The last short tile has its own provider evidence.
pub(in super::super) struct StochasticNoiseUploadPhase {
    count: u64,
    pub(in super::super) query: ProviderFeasibilityQuery,
    pub(in super::super) physical: ProviderNumericUploadPhysicalEvidence,
    pub(in super::super) registration: Option<ProviderRegistrationHostQuote>,
    pub(in super::super) query_host_bytes: u64,
}

pub(in super::super) struct StochasticNoiseUploadPlan {
    pub(in super::super) full: StochasticNoiseUploadPhase,
    pub(in super::super) remainder: Option<StochasticNoiseUploadPhase>,
    // The query/representation storage remains live for every tile; this
    // receipt is admitted before constructing the first owned descriptor.
    pub(in super::super) query_receipt: Option<HostAllocationReceipt>,
}

/// A host initial state is not a fabricated resident input. Its one native
/// upload is independently selected and charged before any source effect, but
/// executes after the first t/Z stores to preserve the original prefix on a
/// transfer failure.
pub(in super::super) struct StochasticInitialUploadPlan {
    pub(in super::super) phase: StochasticNoiseUploadPhase,
    pub(in super::super) query_receipt: Option<HostAllocationReceipt>,
    // The ExactShared header retires before this payload and its claim.
    _owner_claim: Option<ExactHostAllocationClaim>,
}

/// The selected route's initial source authority. A host source cannot be
/// passed to the provider as a synthetic Device representation; its exact
/// class-preserving transfer is an independently quoted stage.
pub(in super::super) enum InitialProviderSource {
    Resident(GpuTensorHandle),
    HostUpload(ExactShared<StochasticInitialUploadPlan>),
}

impl InitialProviderSource {
    pub(in super::super) fn prepare(
        context: &RuntimeContext,
        provider: &'static dyn AccelProvider,
        prepared: &StochasticPreparedSource,
        selection: &ProviderStochasticEvolutionSelection,
    ) -> BuiltinResult<Option<Self>> {
        match prepared.state() {
            Value::GpuTensor(handle) => {
                handle.allocation_lease().validate_live().map_err(|cause| {
                    provider_failure(
                        "selected stochastic input lease retired",
                        runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                            .with_unproven_effects(),
                    )
                })?;
                let expected_element = match prepared.pattern().state_class() {
                    NumericClass::Single => runmat_accelerate_api::NumericElementType::F32,
                    NumericClass::Double => runmat_accelerate_api::NumericElementType::F64,
                    _ => return Ok(None),
                };
                handle.view().validate(handle.physical()).map_err(|cause| {
                    provider_failure(
                        "prepared stochastic resident view is invalid",
                        runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                            .with_unproven_effects(),
                    )
                })?;
                if runmat_accelerate_api::provider_for_handle(handle).is_none_or(|actual| {
                    !runmat_accelerate_api::same_provider_instance(actual, provider)
                }) || handle.shape() != prepared.pattern().shape()
                    || handle.physical().element_type() != expected_element
                    || handle.physical().storage() != GpuTensorStorage::Real
                    || handle.physical().requested_bytes()
                        < selection.plan.state.checked_byte_len().unwrap_or(u64::MAX)
                {
                    return Err(upload_plan_error(
                        "prepared stochastic resident input no longer matches its selected representation",
                    ));
                }
                Ok(Some(Self::Resident(handle.clone())))
            }
            Value::Num(_) | Value::Tensor(_) => Ok(StochasticInitialUploadPlan::prepare(
                context, provider, selection,
            )?
            .map(Self::HostUpload)),
            _ => Ok(None),
        }
    }
}

impl StochasticInitialUploadPlan {
    pub(in super::super) fn requested_query_host_bytes(
        selection: &ProviderStochasticEvolutionSelection,
    ) -> Option<u64> {
        StochasticNoiseUploadPhase::requested_query_host_bytes(selection.plan.state.shape.len())
    }

    pub(in super::super) fn prepare(
        context: &RuntimeContext,
        provider: &dyn AccelProvider,
        selection: &ProviderStochasticEvolutionSelection,
    ) -> BuiltinResult<Option<ExactShared<Self>>> {
        if !selection.proposal.validate_for(&selection.plan) {
            return Err(upload_plan_error("initial-state selection changed"));
        }
        let requested = Self::requested_query_host_bytes(selection)
            .ok_or_else(|| upload_admission_error(context, HostAllocationError::Overflow))?;
        let owner_bytes = ExactShared::<Self>::requested_bytes();
        let owner_retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(owner_bytes)
                .map_err(|_| upload_admission_error(context, HostAllocationError::Overflow))?,
            1,
        )
        .map_err(|cause| upload_admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(requested, owner_retained)
            .map_err(|cause| upload_admission_error(context, cause))?;
        let query_receipt = match HostAllocationReceipt::admit(Some(context), plan) {
            Ok(receipt) => receipt,
            Err(cause) if cause.is_capacity_denial() => return Ok(None),
            Err(cause) => return Err(upload_admission_error(context, cause)),
        };
        let Some(phase) = StochasticNoiseUploadPhase::prepare(
            provider,
            selection.plan.first_arithmetic.state_class(),
            &selection.plan.state.shape,
        )?
        else {
            return Ok(None);
        };
        if phase.query_host_bytes != requested {
            return Err(upload_plan_error("initial-state query quote changed"));
        }
        let owner_claim = match query_receipt.as_ref() {
            Some(receipt) => Some(
                receipt
                    .with_output_allocation(context, |permit| {
                        permit.reserve_exact_claim(owner_bytes)
                    })
                    .map_err(|cause| upload_admission_error(context, cause))?,
            ),
            None => None,
        };
        let owner = Self {
            phase,
            query_receipt,
            _owner_claim: owner_claim,
        };
        let owner = ExactShared::try_new(owner).map_err(|_| {
            upload_admission_error(
                context,
                HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed),
            )
        })?;
        if let Some(receipt) = owner.get().query_receipt.as_ref() {
            receipt
                .checkpoint(context)
                .map_err(|cause| upload_admission_error(context, cause))?;
        }
        Ok(Some(owner))
    }
}

impl StochasticNoiseUploadPlan {
    /// Requested simultaneous query storage before constructing either
    /// representation or query. The caller must admit this at the canonical
    /// host-allocation owner alongside its other selected-route metadata.
    pub(in super::super) fn requested_query_host_bytes(
        selection: &ProviderStochasticEvolutionSelection,
    ) -> Option<u64> {
        let total = selection.plan.validate()?;
        let full = StochasticNoiseUploadPhase::requested_query_host_bytes(2)?;
        let has_remainder = total % selection.plan.noise_tile_elements != 0;
        if has_remainder {
            full.checked_mul(2)
        } else {
            Some(full)
        }
    }

    pub(in super::super) fn prepare(
        context: &RuntimeContext,
        provider: &dyn AccelProvider,
        selection: &ProviderStochasticEvolutionSelection,
    ) -> BuiltinResult<Option<Self>> {
        let total = selection
            .plan
            .validate()
            .ok_or_else(|| upload_plan_error("selected work is invalid"))?;
        if !selection.proposal.validate_for(&selection.plan) {
            return Err(upload_plan_error("physical selection changed"));
        }
        // The transfer witness is mandatory for the serial one-tile peak:
        // merely submitting an observed upload does not retire its callback
        // owner before the next tile. Providers without this exact contract
        // decline before any random draw or source publication.
        match provider.stochastic_evolution_source_transfer_retirement_host_workspace() {
            ResourceByteBound::Known(bytes)
                if bytes != 0
                    && bytes == selection.proposal.source_transfer_retirement_host_bytes => {}
            ResourceByteBound::Unavailable(_) => return Ok(None),
            ResourceByteBound::Known(_) => {
                return Err(upload_plan_error(
                    "source-transfer retirement differs from selected physical proposal",
                ));
            }
        }
        let requested = Self::requested_query_host_bytes(selection)
            .ok_or_else(|| upload_admission_error(context, HostAllocationError::Overflow))?;
        let plan = HostAllocationPlan::new(requested, 0)
            .map_err(|cause| upload_admission_error(context, cause))?;
        let query_receipt = match HostAllocationReceipt::admit(Some(context), plan) {
            Ok(receipt) => receipt,
            Err(cause) if cause.is_capacity_denial() => return Ok(None),
            Err(cause) => return Err(upload_admission_error(context, cause)),
        };
        let class = selection.plan.first_arithmetic.noise_class();
        let full_count = selection.plan.noise_tile_elements;
        let Some(full) = StochasticNoiseUploadPhase::prepare(provider, class, &[full_count, 1])?
        else {
            return Ok(None);
        };
        let remaining = total % full_count;
        let remainder = if remaining == 0 {
            None
        } else {
            let Some(phase) =
                StochasticNoiseUploadPhase::prepare(provider, class, &[remaining, 1])?
            else {
                return Ok(None);
            };
            Some(phase)
        };
        let plan = Self {
            full,
            remainder,
            query_receipt,
        };
        let actual = plan
            .full
            .query_host_bytes
            .checked_add(
                plan.remainder
                    .as_ref()
                    .map_or(0, |phase| phase.query_host_bytes),
            )
            .ok_or_else(|| upload_plan_error("query extent overflow"))?;
        if Some(actual) != Self::requested_query_host_bytes(selection) {
            return Err(upload_plan_error("query metadata quote changed"));
        }
        Ok(Some(plan))
    }

    pub(in super::super) fn phase_for(&self, count: u64) -> Option<&StochasticNoiseUploadPhase> {
        if count == self.full.count {
            Some(&self.full)
        } else {
            self.remainder.as_ref().filter(|phase| phase.count == count)
        }
    }
}

impl StochasticNoiseUploadPhase {
    fn requested_query_host_bytes(rank: usize) -> Option<u64> {
        let ranks = [rank];
        ProviderQueryMetadataFootprint {
            operation: "transfer.upload",
            input_ranks: &ranks,
            output_ranks: &ranks,
            preparation_shape_ranks: &ranks,
            preparation_representation_slots: 1,
        }
        .requested_bytes()
    }

    fn prepare(
        provider: &dyn AccelProvider,
        class: NumericClass,
        shape: &[u64],
    ) -> BuiltinResult<Option<Self>> {
        let element_type = match class {
            NumericClass::Single => ProviderElementType::F32,
            NumericClass::Double => ProviderElementType::F64,
            _ => return Ok(None),
        };
        let count = shape
            .iter()
            .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
            .ok_or_else(|| upload_plan_error("upload shape overflows"))?;
        let output = ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape.to_vec(),
            residency: ProviderResidency::Device,
        };
        let query = class_preserving_upload_query(vec![output])
            .ok_or_else(|| upload_plan_error("no exact numeric upload query"))?;
        let ranks = [shape.len()];
        let footprint = ProviderQueryMetadataFootprint {
            operation: "transfer.upload",
            input_ranks: &ranks,
            output_ranks: &ranks,
            preparation_shape_ranks: &ranks,
            preparation_representation_slots: 1,
        };
        if !footprint.matches(&query) {
            return Err(upload_plan_error(
                "query metadata differs from the admitted plan",
            ));
        }
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
            return Ok(None);
        };
        let Some(physical) = provider.numeric_upload_physical_evidence(&query) else {
            return Ok(None);
        };
        if estimate.output_bytes != Some(physical.output_requested_bytes)
            || physical.output_requested_bytes
                < query.outputs[0].checked_byte_len().unwrap_or(u64::MAX)
        {
            return Err(upload_plan_error(
                "feasibility and physical upload disagree",
            ));
        }
        let registration = provider_registration_host_quote(provider, shape.len(), 1, 0)
            .map_err(upload_plan_error)?;
        Ok(Some(Self {
            count,
            query,
            physical,
            registration,
            query_host_bytes: footprint
                .requested_bytes()
                .ok_or_else(|| upload_plan_error("query extent overflow"))?,
        }))
    }
}

/// Execute only an already admitted phase. This helper never selects a route:
/// the caller supplies the exact physical scope limit, registered-output
/// permit, selected recorder and immutable query phase before any source draw.
pub(super) async fn upload_selected_phase(
    provider: &'static dyn AccelProvider,
    phase: &StochasticNoiseUploadPhase,
    host: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
    recorder: &ProviderOperationRecorder,
    scope: &ProviderResourceScope,
) -> BuiltinResult<GpuTensorHandle> {
    if host
        .shape
        .iter()
        .map(|&extent| u64::try_from(extent).ok())
        .ne(phase.query.outputs[0].shape.iter().copied().map(Some))
    {
        return Err(upload_plan_error(
            "live upload geometry differs from selected phase",
        ));
    }
    // The observed upload submits a separate completion fence. Its exact
    // callback must release that fence and its resource ticket before the
    // next source-transfer or compute phase may reuse this bounded scope.
    let retirement = recorder
        .arm_stochastic_evolution_source_transfer_retirement(provider.device_id())
        .map_err(upload_plan_error)?;
    let uploaded = observe_provider_resource_future(Some(scope.clone()), async {
        with_provider_observation(Some(recorder.clone()), || {
            upload_numeric_protecting_admitted(provider, host, protected, metadata)
        })
    })
    .await;
    let result = uploaded.map_err(|cause| {
        scope.fail(ProviderObservationFailure::ProviderFailed);
        provider_failure(
            "selected stochastic noise upload failed",
            cause.into_execution_error(),
        )
    })?;
    if result.physical().requested_bytes() != phase.physical.output_requested_bytes
        || runmat_accelerate_api::provider_for_handle(&result)
            .is_none_or(|actual| !runmat_accelerate_api::same_provider_instance(actual, provider))
    {
        scope.fail(ProviderObservationFailure::PublicationFailed);
        return Err(upload_plan_error(
            "uploaded tile violates selected physical owner",
        ));
    }
    retirement.await.map_err(upload_plan_error)?;
    Ok(result)
}
