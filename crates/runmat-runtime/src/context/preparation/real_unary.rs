//! Complete physical alternatives of one already-admitted unary body.
//! No cost, numerical-domain promise, selection, or completion right is minted.
use super::{AdmittedBuiltinPreparation, RuntimeContext};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    AccelProvider, ProviderElementType, ProviderFeasibilityQuery, ProviderRealUnaryCompound,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use runmat_execution::{
    CandidateOutputResidency, CandidatePeakLiveDemand, CandidateResourceDemand, Digest,
    ProviderInputStageDemand, ProviderInputStageDemands, ResourceAccount, ResourceByteBound,
};
use runmat_types::{NumericClass, NumericDomain};
use runmat_value::Value;
use std::error::Error;

pub struct AdmittedRealUnaryCompound {
    physical: ProviderRealUnaryCompound,
    resources: CandidateResourceDemand,
    residency: CandidateOutputResidency,
    identity: Digest,
    capabilities: runmat_accelerate_api::ProviderCapabilitySnapshot,
    source: runmat_accelerate_api::GpuTensorWireDescriptor,
    admission: super::BuiltinPreparationAdmission,
    unary: super::AdmittedRealUnary,
    current: Option<super::PreparedHostExecution>,
    form: &'static runmat_builtins::BuiltinPreparationForm,
    continuations: runmat_accelerate_api::ProviderRealUnaryContinuations,
}

impl AdmittedRealUnaryCompound {
    pub(crate) fn invalid_continuation() -> crate::RuntimeError {
        crate::build_runtime_error("compound unary did not consume its exact admitted continuation")
            .with_identifier("RunMat:Placement:InvalidDecision")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    }

    pub(crate) fn provider_execution_error(
        &self,
        runtime: Option<&RuntimeContext>,
        descriptor: &'static runmat_builtins::BuiltinErrorDescriptor,
        cause: runmat_accelerate_api::ProviderExecutionError,
    ) -> crate::RuntimeError {
        let cancelled = runtime.is_some_and(RuntimeContext::is_cancelled)
            || cause.source().is_some_and(|source| {
                source.is::<runmat_accelerate_api::ProviderCompoundCancelled>()
            });
        let mut error = crate::build_runtime_error(format!("{}: {cause}", descriptor.message))
            .with_builtin(self.admission.binding.builtin.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if cancelled {
            error = error.with_identifier("RunMat:Cancelled");
        } else if let Some(identifier) = descriptor.identifier {
            error = error.with_identifier(identifier);
        }
        crate::runtime_error::with_provider_execution_source(error.build(), cause)
    }

    pub(crate) fn preflight_execution(
        &self,
        runtime: Option<&RuntimeContext>,
        values: &[&Value],
        observed: bool,
    ) -> BuiltinResult<()> {
        self.validate_current(runtime, values)?;
        let owner =
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(self.physical.source())
                .ok_or_else(|| {
                    super::super::provider_operation::resource_error(
                        runmat_execution::ResourceAdmissionError::UnavailableProvider,
                    )
                })?;
        let overhead = if observed {
            match owner.real_unary_observation_capability(&self.physical, self.continuations) {
                runmat_accelerate_api::observation::ProviderObservationCapability::Supported(
                    overhead,
                ) => Some(overhead),
                runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => {
                    return Err(crate::build_runtime_error(
                        "compound unary observation owner is no longer available",
                    )
                    .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build())
                }
            }
        } else {
            None
        };
        let demand = self.resources_with_observation(overhead)?;
        super::compound_resources::preflight(
            runtime,
            owner,
            demand,
            self.residency,
            &self.capabilities,
            || self.validate_current(runtime, values),
        )
    }
    pub fn physical(&self) -> &ProviderRealUnaryCompound {
        &self.physical
    }
    pub fn resources(&self) -> CandidateResourceDemand {
        self.resources
    }
    pub fn resources_with_observation(
        &self,
        observation: Option<runmat_accelerate_api::observation::ProviderObservationOverhead>,
    ) -> BuiltinResult<CandidateResourceDemand> {
        super::compound_resources::with_observation(self.resources, observation)
    }
    pub fn output_residency(&self) -> CandidateOutputResidency {
        self.residency
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn admission(&self) -> super::BuiltinPreparationAdmission {
        self.admission
    }
    pub fn unary(&self) -> super::AdmittedRealUnary {
        self.unary
    }
    pub fn continuations(&self) -> runmat_accelerate_api::ProviderRealUnaryContinuations {
        self.continuations
    }
    pub(crate) fn validate_current(
        &self,
        runtime: Option<&RuntimeContext>,
        values: &[&Value],
    ) -> BuiltinResult<()> {
        let owner =
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(self.physical.source());
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(cancelled());
        }
        if owner.is_none_or(|owner| !self.matches(owner, values))
            || self
                .current
                .as_ref()
                .is_some_and(|proof| !runtime.is_some_and(|runtime| proof.is_current_for(runtime)))
        {
            return Err(crate::build_runtime_error(
                "compound unary binding/source/owner is no longer current",
            )
            .with_identifier("RunMat:Placement:InvalidDecision")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        Ok(())
    }
    pub(crate) fn select_host_branch(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Option<super::PreparedHostExecution>> {
        let Some(proof) = self.current.as_ref() else {
            return Ok(None);
        };
        if !runtime.is_some_and(|runtime| proof.is_current_for(runtime)) || !proof.select(self.form)
        {
            return Err(crate::build_runtime_error(
                "compound unary host continuation lost its original completion right",
            )
            .with_identifier("RunMat:Placement:InvalidDecision")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        Ok(Some(proof.clone()))
    }

    /// Revalidate the actual source, not a cached representation of another
    /// view. The original completion owner separately checks its generation.
    pub fn matches(&self, provider: &dyn AccelProvider, operands: &[&Value]) -> bool {
        matches!(operands, [Value::GpuTensor(source)]
            if source.view_identity() == self.physical.source().view_identity()
                && source.wire_descriptor() == self.source
                && source.allocation_lease().validate_live().is_ok()
                && source.view().validate(source.physical()).is_ok())
            && self.physical.is_owned_by(provider)
            && provider.capability_snapshot() == self.capabilities
    }
}

pub(super) fn plan(
    request: &AdmittedBuiltinPreparation,
    context: Option<&RuntimeContext>,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<AdmittedRealUnaryCompound>> {
    if context.is_some_and(RuntimeContext::is_cancelled) {
        return Err(cancelled());
    }
    let Some(unary) = request.real_unary() else {
        return Ok(None);
    };
    let [Value::GpuTensor(source)] = request.operands() else {
        return Ok(None);
    };
    if crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
        .is_none_or(|actual| !runmat_accelerate_api::same_provider_instance(actual, provider))
    {
        return Ok(None);
    }
    let Some(host) = super::host_resources::bound(request, context) else {
        return Ok(None);
    };
    let Some(physical) = ProviderRealUnaryCompound::plan(provider, source, unary.operation())
        .map_err(provider_error)?
    else {
        return Ok(None);
    };
    let Some(download) = ProviderFeasibilityQuery::exact_numeric_download(source) else {
        return Ok(None);
    };
    if super::super::provider_operation::provider_resource_query(context, provider, || {
        provider.query_numeric_download_feasibility(source, &download)
    })?
    .is_none()
    {
        return Ok(None);
    }
    let Some(payload) = runmat_accelerate_api::checked_numeric_download_host_payload_bytes(source)
    else {
        return Ok(None);
    };
    let metadata = host.operand_metadata;
    let mirror = payload.checked_add(metadata).ok_or_else(overflow)?;
    let mut host_peak = account(mirror)?
        .checked_add(provider.download_numeric_host_workspace_bound(source))
        .ok_or_else(overflow)?;
    let host_body = mirror
        .checked_add(host.bound.retained_bytes)
        .and_then(|n| n.checked_add(host.bound.scratch_bytes))
        .and_then(|n| n.checked_add(metadata))
        .ok_or_else(overflow)?;
    host_peak = host_peak.peak(account(host_body)?);
    let mut stages = Vec::with_capacity(4);
    stages.push(ProviderInputStageDemand {
        device_id: provider.device_id(),
        incremental_peak_bytes: provider.download_numeric_provider_scratch_bound(source),
        queue_slots: provider.download_numeric_queue_slots(source),
    });
    for probe in physical.probes() {
        let Some(readback) = provider.produced_numeric_readback_evidence(&probe.output()) else {
            return Ok(None);
        };
        if readback.source_requested_bytes != probe.physical().output_buffer_bytes {
            return Ok(None);
        }
        let probe_peak = probe
            .physical()
            .requested_incremental_peak_bytes()
            .ok_or_else(overflow)?;
        let scalar_peak = account(probe_peak)?
            .checked_add(readback.provider_scratch_bytes)
            .ok_or_else(overflow)?;
        stages.push(ProviderInputStageDemand {
            device_id: provider.device_id(),
            incremental_peak_bytes: scalar_peak.bound(),
            queue_slots: probe
                .physical()
                .queue_submissions
                .checked_add(readback.queue_slots)
                .ok_or_else(overflow)?,
        });
        host_peak = host_peak.peak(
            account(probe.output().checked_byte_len().ok_or_else(overflow)?)?
                .checked_add(readback.host_workspace_bytes)
                .ok_or_else(overflow)?,
        );
    }
    let real = physical.real().physical();
    let mut provider_peak = account(
        real.requested_incremental_peak_bytes()
            .ok_or_else(overflow)?,
    )?;
    let mut retained = real.output_buffer_bytes;
    let mut queue_slots = real.queue_submissions;
    let mut host_published = 0;
    // Both alternatives are bounded before the first probe. Unknown remains
    // an explicit real/complex envelope, never permission for a real shader.
    for domain in [NumericDomain::Real, NumericDomain::Complex] {
        match host.output_domain {
            crate::builtins::common::spec::FusionHostOutputDomain::Exact(exact)
                if exact != domain =>
            {
                continue
            }
            crate::builtins::common::spec::FusionHostOutputDomain::Exact(_)
            | crate::builtins::common::spec::FusionHostOutputDomain::RealOrComplex => {}
        }
        if unary.publication().publishes_host(source, domain) {
            host_published = host.bound.retained_bytes;
            continue;
        }
        let element_type = match (host.output_class, domain) {
            (NumericClass::Single, NumericDomain::Real) => ProviderElementType::F32,
            (NumericClass::Double, NumericDomain::Real) => ProviderElementType::F64,
            (NumericClass::Single, NumericDomain::Complex) => ProviderElementType::ComplexF32,
            (NumericClass::Double, NumericDomain::Complex) => ProviderElementType::ComplexF64,
            _ => return Ok(None),
        };
        let output = ProviderRepresentation {
            element_type,
            storage: if domain == NumericDomain::Real {
                ProviderStorage::DenseReal
            } else {
                ProviderStorage::DenseComplexInterleaved
            },
            layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
            shape: host
                .output_shape
                .iter()
                .map(|&n| u64::try_from(n))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| overflow())?,
            residency: ProviderResidency::Device,
        };
        let Some(upload_query) =
            runmat_accelerate_api::class_preserving_upload_query(vec![output.clone()])
        else {
            return Ok(None);
        };
        let Some(upload) = provider.upload_physical_evidence(&upload_query) else {
            return Ok(None);
        };
        let packing = if domain == NumericDomain::Complex {
            crate::builtins::common::gpu_helpers::native_complex_upload_workspace(&output)
                .ok_or_else(overflow)?
        } else {
            0
        };
        host_peak = host_peak.peak(
            account(host_body)?
                .checked_add(ResourceByteBound::Known(packing))
                .and_then(|n| n.checked_add(upload.host_workspace_bytes))
                .ok_or_else(overflow)?,
        );
        provider_peak = provider_peak.peak(
            account(upload.output_requested_bytes)?
                .checked_add(upload.provider_scratch_bytes)
                .ok_or_else(overflow)?,
        );
        retained = retained.max(upload.output_requested_bytes);
        queue_slots = queue_slots.max(upload.completion_queue_slots);
    }
    let peak_live = match provider_peak.bound() {
        ResourceByteBound::Known(peak_bytes) => Some(CandidatePeakLiveDemand {
            peak_bytes,
            already_live_bytes: 0,
        }),
        ResourceByteBound::Unavailable(_) => None,
    };
    let resources = CandidateResourceDemand {
        cpu_millicores: 1000,
        retained_bytes: retained.into(),
        scratch_bytes: match provider_peak.bound() {
            ResourceByteBound::Known(peak) => {
                peak.checked_sub(retained).ok_or_else(overflow)?.into()
            }
            ResourceByteBound::Unavailable(reason) => ResourceByteBound::Unavailable(reason),
        },
        provider_host_workspace: Some(host_peak.bound()),
        provider_input_stages: Some(
            ProviderInputStageDemands::sequential(stages).ok_or_else(overflow)?,
        ),
        queue_slots,
        peak_live,
    };
    if context.is_some_and(RuntimeContext::is_cancelled) {
        return Err(cancelled());
    }
    let admission = request.admission();
    let publication = |domain| {
        if unary.publication().publishes_host(source, domain) {
            runmat_accelerate_api::ProviderRealUnaryPublication::Host
        } else {
            runmat_accelerate_api::ProviderRealUnaryPublication::ProviderUpload
        }
    };
    let continuations = runmat_accelerate_api::ProviderRealUnaryContinuations {
        real: publication(NumericDomain::Real),
        complex: publication(NumericDomain::Complex),
    };
    let current = context
        .and_then(|context| {
            context
                .prepared_host_execution()
                .filter(|proof| proof.matches_admission(admission) && proof.is_current_for(context))
        })
        .cloned();
    if request.has_prepared_completion_right() && current.is_none() {
        return Err(crate::build_runtime_error(
            "compound unary original completion right expired during preparation",
        )
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    let capabilities = provider.capability_snapshot();
    let identity = Digest::sha256_serialized(&(
        "admitted-real-unary-compound-v1",
        admission.binding,
        admission.form,
        admission.signature,
        admission.input_count,
        admission.requested_outputs,
        request.numerical_domain(),
        unary.operation(),
        unary.publication(),
        physical.identity().map_err(|_| overflow())?,
        host.identity,
        resources,
        &capabilities,
    ))
    .map_err(|_| overflow())?;
    Ok(Some(AdmittedRealUnaryCompound {
        source: source.wire_descriptor(),
        physical,
        resources,
        residency: if host_published == 0 {
            CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            }
        } else {
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id: provider.device_id(),
                host_bytes: host_published.into(),
            }
        },
        identity,
        capabilities,
        admission,
        unary,
        current,
        form: request.form(),
        continuations,
    }))
}

fn account(bytes: u64) -> BuiltinResult<ResourceAccount> {
    ResourceAccount::default()
        .checked_add(bytes.into())
        .ok_or_else(overflow)
}
fn overflow() -> crate::RuntimeError {
    crate::build_runtime_error("compound unary physical resource extent overflows")
        .with_identifier("RunMat:Placement:ResourceOverflow")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn cancelled() -> crate::RuntimeError {
    crate::build_runtime_error("compound unary cancelled before admission")
        .with_identifier("RunMat:Cancelled")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn provider_error(cause: runmat_accelerate_api::ProviderExecutionError) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        crate::build_runtime_error("compound unary physical preparation failed")
            .with_identifier("RunMat:Placement:InvalidPhysicalRecipe")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        cause,
    )
}
