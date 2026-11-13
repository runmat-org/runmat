//! Complete physical paired POWER demand; original Unknown numerical evidence
//! remains unchanged until the retained actual probe selects an alternative.
use super::{AdmittedBuiltinPreparation, BuiltinPreparationAdmission, RuntimeContext};
use crate::BuiltinResult;
use runmat_accelerate_api::{power_service::PowerPairCompound, AccelProvider};
use runmat_execution::{
    CandidateOutputResidency, CandidatePeakLiveDemand, CandidateResourceDemand, Digest,
    ProviderInputStageDemands,
};
use runmat_value::Value;

pub struct AdmittedRealPowerCompound {
    physical: PowerPairCompound,
    resources: CandidateResourceDemand,
    identity: Digest,
    capabilities: runmat_accelerate_api::ProviderCapabilitySnapshot,
    sources: [runmat_accelerate_api::GpuTensorWireDescriptor; 2],
    admission: BuiltinPreparationAdmission,
    current: Option<super::PreparedHostExecution>,
    inputs: Vec<runmat_accelerate_api::ProviderRepresentation>,
}

impl AdmittedRealPowerCompound {
    pub fn physical(&self) -> &PowerPairCompound {
        &self.physical
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn admission(&self) -> BuiltinPreparationAdmission {
        self.admission
    }
    pub fn resources(&self) -> CandidateResourceDemand {
        self.resources
    }
    pub fn inputs(&self) -> &[runmat_accelerate_api::ProviderRepresentation] {
        &self.inputs
    }
    pub fn output_residency(&self) -> CandidateOutputResidency {
        CandidateOutputResidency::Provider {
            device_id: self.physical.probe().sources()[0].device_id(),
        }
    }
    pub fn resources_with_observation(
        &self,
        overhead: Option<runmat_accelerate_api::observation::ProviderObservationOverhead>,
    ) -> BuiltinResult<CandidateResourceDemand> {
        super::compound_resources::with_observation(self.resources, overhead)
    }
    pub fn matches(&self, owner: &dyn AccelProvider, values: &[&Value]) -> bool {
        let [Value::GpuTensor(base), Value::GpuTensor(exponent)] = values else {
            return false;
        };
        [base, exponent]
            .into_iter()
            .zip(self.physical.probe().sources())
            .zip(&self.sources)
            .all(|((actual, retained), wire)| {
                actual.view_identity() == retained.view_identity()
                    && actual.wire_descriptor() == *wire
                    && actual.view().validate(actual.physical()).is_ok()
                    && actual.allocation_lease().validate_live().is_ok()
            })
            && self.physical.is_owned_by(owner)
            && owner.capability_snapshot() == self.capabilities
    }
    pub(crate) fn validate_current(
        &self,
        runtime: Option<&RuntimeContext>,
        values: &[&Value],
    ) -> BuiltinResult<()> {
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(cancelled());
        }
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(
            &self.physical.probe().sources()[0],
        );
        if owner.is_none_or(|owner| !self.matches(owner, values))
            || self
                .current
                .as_ref()
                .is_some_and(|proof| !runtime.is_some_and(|runtime| proof.is_current_for(runtime)))
        {
            return Err(invalid(
                "paired POWER original binding/source/owner is no longer current",
            ));
        }
        Ok(())
    }
    pub(crate) fn preflight_execution(
        &self,
        runtime: Option<&RuntimeContext>,
        values: &[&Value],
        observed: bool,
    ) -> BuiltinResult<()> {
        self.validate_current(runtime, values)?;
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(
            &self.physical.probe().sources()[0],
        )
        .ok_or_else(|| invalid("paired POWER source owner unavailable"))?;
        let overhead = if observed {
            match owner.power_pair_observation_capability(&self.physical) {
                runmat_accelerate_api::observation::ProviderObservationCapability::Supported(
                    overhead,
                ) => Some(overhead),
                runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => {
                    return Err(invalid(
                        "paired POWER original observation owner unavailable",
                    ));
                }
            }
        } else {
            None
        };
        super::compound_resources::preflight(
            runtime,
            owner,
            self.resources_with_observation(overhead)?,
            self.output_residency(),
            &self.capabilities,
            || self.validate_current(runtime, values),
        )
    }
}

pub(super) fn plan(
    request: &AdmittedBuiltinPreparation,
    runtime: Option<&RuntimeContext>,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<Option<AdmittedRealPowerCompound>> {
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(cancelled());
    }
    if !request.is_real_power() {
        return Ok(None);
    }
    let [Value::GpuTensor(base), Value::GpuTensor(exponent)] = request.operands() else {
        return Ok(None);
    };
    if [base, exponent].into_iter().any(|source| {
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
            .is_none_or(|actual| !runmat_accelerate_api::same_provider_instance(actual, owner))
    }) {
        return Ok(None);
    }
    let Some(physical) = PowerPairCompound::plan(owner, base, exponent).map_err(|cause| {
        crate::runtime_error::with_provider_execution_source(
            invalid("paired POWER physical preparation failed"),
            cause,
        )
    })?
    else {
        return Ok(None);
    };
    let probe_peak = physical
        .probe()
        .physical()
        .phase_data_limits
        .into_iter()
        .max()
        .ok_or_else(overflow)?;
    let mut continuation_peak = 0_u64;
    let mut retained = 0_u64;
    let mut continuation_queues = 0_u32;
    for evaluation in [physical.real(), physical.complex()] {
        let work = &evaluation.physical().work;
        continuation_peak = continuation_peak.max(
            work.requested_incremental_peak_bytes()
                .ok_or_else(overflow)?,
        );
        retained = retained.max(work.output_buffer_bytes);
        continuation_queues = continuation_queues.max(work.queue_submissions);
    }
    // Keep all probe backing in this conservative complete-route peak until
    // the selected continuation ends; no allocator callback ordering is assumed.
    let peak = probe_peak
        .checked_add(continuation_peak)
        .ok_or_else(overflow)?;
    let resources = CandidateResourceDemand {
        cpu_millicores: 1000,
        retained_bytes: retained.into(),
        scratch_bytes: peak.checked_sub(retained).ok_or_else(overflow)?.into(),
        provider_host_workspace: Some(physical.probe().physical().host_workspace_bytes.into()),
        provider_input_stages: Some(ProviderInputStageDemands::default()),
        queue_slots: physical
            .probe()
            .physical()
            .queue_submissions
            .checked_add(continuation_queues)
            .ok_or_else(overflow)?,
        peak_live: Some(CandidatePeakLiveDemand {
            peak_bytes: peak,
            already_live_bytes: 0,
        }),
    };
    let admission = request.admission();
    let current = runtime
        .and_then(|runtime| {
            runtime
                .prepared_host_execution()
                .filter(|proof| proof.matches_admission(admission) && proof.is_current_for(runtime))
        })
        .cloned();
    if request.has_prepared_completion_right() && current.is_none() {
        return Err(invalid(
            "paired POWER original completion right expired during preparation",
        ));
    }
    let inputs = [base, exponent]
        .into_iter()
        .map(|source| {
            runmat_accelerate_api::ProviderFeasibilityQuery::exact_numeric_download(source)
                .and_then(|query| query.inputs.into_iter().next())
                .ok_or_else(|| invalid("paired POWER native source projection unavailable"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(cancelled());
    }
    let capabilities = owner.capability_snapshot();
    let identity = Digest::sha256_serialized(&(
        "admitted-real-power-compound-v1",
        admission.binding,
        admission.form,
        admission.signature,
        admission.input_count,
        admission.requested_outputs,
        request.numerical_domain(),
        physical.identity().map_err(|_| overflow())?,
        resources,
        &capabilities,
    ))
    .map_err(|_| overflow())?;
    Ok(Some(AdmittedRealPowerCompound {
        physical,
        resources,
        identity,
        capabilities,
        sources: [base.wire_descriptor(), exponent.wire_descriptor()],
        admission,
        current,
        inputs,
    }))
}

fn invalid(message: &'static str) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn overflow() -> crate::RuntimeError {
    crate::build_runtime_error("paired POWER physical resources overflow")
        .with_identifier("RunMat:Placement:ResourceOverflow")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn cancelled() -> crate::RuntimeError {
    crate::build_runtime_error("paired POWER cancelled before admission")
        .with_identifier("RunMat:Cancelled")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
