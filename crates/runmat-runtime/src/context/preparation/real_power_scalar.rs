//! Admitted native scalar POWER with a real resident source and exact host
//! control. No future scalar handle exists before the selected body uploads it.
use super::{AdmittedBuiltinPreparation, BuiltinPreparationAdmission, RuntimeContext};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    power_service::{
        prepare_resident_scalar_power, PowerPhysicalRoute, PowerTypedScalarProjection,
        PreparedResidentScalarPower,
    },
    AccelProvider, GpuTensorHandle, PowerResultDomain, PowerScalarControl,
    ProviderOperationIdentity, ProviderRepresentation, RealPowerOperand,
};
use runmat_execution::{
    CandidateOutputResidency, CandidatePeakLiveDemand, CandidateResourceDemand, Digest,
    ProviderInputStageDemand, ProviderInputStageDemands, ResourceAccount, ResourceByteBound,
};
use runmat_value::{NumericScalar, Value};

pub struct AdmittedRealPowerScalarCompound {
    prepared: Option<PreparedResidentScalarPower>,
    projection: PowerTypedScalarProjection,
    resources: CandidateResourceDemand,
    identity: Digest,
    capabilities: runmat_accelerate_api::ProviderCapabilitySnapshot,
    source: GpuTensorHandle,
    source_wire: runmat_accelerate_api::GpuTensorWireDescriptor,
    control: PowerScalarControl,
    shape: Vec<usize>,
    role: RealPowerOperand,
    admission: BuiltinPreparationAdmission,
    current: Option<super::PreparedHostExecution>,
    inputs: Vec<ProviderRepresentation>,
    output: ProviderRepresentation,
}

impl AdmittedRealPowerScalarCompound {
    pub fn projection(&self) -> &PowerTypedScalarProjection {
        &self.projection
    }
    pub fn source(&self) -> &GpuTensorHandle {
        &self.source
    }
    /// The admitted original host control is not a preparation upload input.
    pub fn host_control_offset(&self) -> usize {
        match self.role {
            RealPowerOperand::Base => 1,
            RealPowerOperand::Exponent => 0,
        }
    }
    pub fn resources(&self) -> CandidateResourceDemand {
        self.resources
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn admission(&self) -> BuiltinPreparationAdmission {
        self.admission
    }
    pub fn inputs(&self) -> &[ProviderRepresentation] {
        &self.inputs
    }
    pub fn output(&self) -> &ProviderRepresentation {
        &self.output
    }
    pub fn output_residency(&self) -> CandidateOutputResidency {
        CandidateOutputResidency::Provider {
            device_id: self.source.device_id(),
        }
    }
    pub(crate) fn take_prepared(&mut self) -> Option<PreparedResidentScalarPower> {
        self.prepared.take()
    }
    pub fn resources_with_observation(
        &self,
        overhead: Option<runmat_accelerate_api::observation::ProviderObservationOverhead>,
    ) -> BuiltinResult<CandidateResourceDemand> {
        super::compound_resources::with_observation(self.resources, overhead)
    }
    pub fn matches(&self, owner: &dyn AccelProvider, values: &[&Value]) -> bool {
        let [left, right] = values else {
            return false;
        };
        let (source, host) = match self.role {
            RealPowerOperand::Base => (left, right),
            RealPowerOperand::Exponent => (right, left),
        };
        let Value::GpuTensor(source) = source else {
            return false;
        };
        scalar(host).is_some_and(|(control, shape)| control == self.control && shape == self.shape)
            && source.view_identity() == self.source.view_identity()
            && source.wire_descriptor() == self.source_wire
            && self.projection.source_matches(source)
            && self.projection.is_current_for(owner)
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
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&self.source);
        if owner.is_none_or(|owner| !self.matches(owner, values))
            || self
                .current
                .as_ref()
                .is_some_and(|proof| !runtime.is_some_and(|runtime| proof.is_current_for(runtime)))
        {
            return Err(invalid(
                "scalar POWER original binding/source is no longer current",
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
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&self.source)
            .ok_or_else(|| invalid("scalar POWER source owner unavailable"))?;
        let overhead = if observed {
            match owner.power_scalar_observation_capability(&self.projection) {
                runmat_accelerate_api::observation::ProviderObservationCapability::Supported(
                    overhead,
                ) => Some(overhead),
                runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => {
                    return Err(invalid("scalar POWER observation owner unavailable"));
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
) -> BuiltinResult<Option<AdmittedRealPowerScalarCompound>> {
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(cancelled());
    }
    if !request.is_real_power() {
        return Ok(None);
    }
    let (source, host, role) = match request.operands() {
        [Value::GpuTensor(source), host] => (source, host, RealPowerOperand::Base),
        [host, Value::GpuTensor(source)] => (source, host, RealPowerOperand::Exponent),
        _ => return Ok(None),
    };
    if crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
        .is_none_or(|actual| !runmat_accelerate_api::same_provider_instance(actual, owner))
    {
        return Ok(None);
    }
    let Some((control, shape)) = scalar(host) else {
        return Ok(None);
    };
    let Some(prepared) = prepare_resident_scalar_power(
        owner,
        source,
        control,
        role,
        &shape,
        ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
    )
    .map_err(provider_error)?
    else {
        return Ok(None);
    };
    if prepared.physical_recipe().route != PowerPhysicalRoute::TypedDomain {
        return Ok(None);
    }
    let Some(projection) = PowerTypedScalarProjection::plan(owner, prepared.symbolic_plan())
        .map_err(|cause| {
            provider_error(runmat_accelerate_api::power_service::PowerServiceError(
                cause,
            ))
        })?
    else {
        return Ok(None);
    };
    let Some(bounds) = projection.resource_bounds() else {
        return Ok(None);
    };
    // Input preparation is sequential, but widening remains live while the
    // exact scalar is uploaded. These are the actual selected stage bounds,
    // not a placeholder for a future scalar allocation.
    let mut input_stages = Vec::with_capacity(2);
    let mut widened_live = ResourceAccount::default();
    if let Some(widened) = projection.widened {
        widened_live = widened_live
            .checked_add(widened.output_requested_bytes.into())
            .ok_or_else(overflow)?;
        input_stages.push(ProviderInputStageDemand {
            device_id: owner.device_id(),
            incremental_peak_bytes: widened_live
                .checked_add(widened.uniform_buffer_bytes.into())
                .ok_or_else(overflow)?
                .bound(),
            queue_slots: widened.queue_slots,
        });
    }
    input_stages.push(ProviderInputStageDemand {
        device_id: owner.device_id(),
        incremental_peak_bytes: widened_live
            .checked_add(projection.scalar_upload.output_requested_bytes.into())
            .and_then(|account| {
                account.checked_add(projection.scalar_upload.provider_scratch_bytes)
            })
            .ok_or_else(overflow)?
            .bound(),
        queue_slots: projection.scalar_upload.completion_queue_slots,
    });
    let resources = CandidateResourceDemand {
        cpu_millicores: 1000,
        retained_bytes: bounds.retained_bytes.into(),
        scratch_bytes: bounds.scratch_bytes,
        provider_host_workspace: Some(bounds.host_workspace_bytes.into()),
        provider_input_stages: Some(
            ProviderInputStageDemands::sequential(input_stages).ok_or_else(overflow)?,
        ),
        queue_slots: bounds.queue_slots,
        peak_live: match bounds.peak_bytes {
            ResourceByteBound::Known(peak_bytes) => Some(CandidatePeakLiveDemand {
                peak_bytes,
                already_live_bytes: 0,
            }),
            ResourceByteBound::Unavailable(_) => None,
        },
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
        return Err(invalid("scalar POWER original completion right expired"));
    }
    let query = prepared
        .feasibility_query(PowerResultDomain::Real)
        .ok_or_else(|| invalid("scalar POWER selected query unavailable"))?;
    let output = prepared
        .symbolic_plan()
        .physical_work(PowerResultDomain::Real)
        .ok_or_else(|| invalid("scalar POWER output projection unavailable"))?
        .output;
    let capabilities = owner.capability_snapshot();
    let source_identity = source.view_identity();
    let identity = Digest::sha256_serialized(&(
        "admitted-real-power-scalar-v1",
        admission.binding,
        admission.form,
        admission.signature,
        admission.input_count,
        admission.requested_outputs,
        request.numerical_domain(),
        (source_identity.allocation(), source_identity.view()),
        control,
        &shape,
        projection.identity().map_err(|_| overflow())?,
        resources,
        &capabilities,
    ))
    .map_err(|_| overflow())?;
    Ok(Some(AdmittedRealPowerScalarCompound {
        prepared: Some(prepared),
        projection,
        resources,
        identity,
        capabilities,
        source: source.clone(),
        source_wire: source.wire_descriptor(),
        control,
        shape,
        role,
        admission,
        current,
        inputs: query.inputs,
        output,
    }))
}

fn scalar(value: &Value) -> Option<(PowerScalarControl, Vec<usize>)> {
    match value {
        Value::Num(value) => Some((PowerScalarControl::DoubleBits(value.to_bits()), vec![1, 1])),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            let shape = tensor.shape();
            if shape
                .iter()
                .try_fold(1usize, |n, extent| n.checked_mul(*extent))
                != Some(1)
            {
                return None;
            }
            let control = match tensor.numeric_value_at(0)? {
                NumericScalar::F32(value) => PowerScalarControl::SingleBits(value.to_bits()),
                NumericScalar::F64(value) => PowerScalarControl::DoubleBits(value.to_bits()),
                _ => return None,
            };
            Some((control, shape.to_vec()))
        }
        _ => None,
    }
}

fn provider_error(
    error: runmat_accelerate_api::power_service::PowerServiceError,
) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        invalid("scalar POWER physical preparation failed"),
        error.into_execution_error(),
    )
}
fn invalid(message: &'static str) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn overflow() -> crate::RuntimeError {
    crate::build_runtime_error("scalar POWER physical resources overflow")
        .with_identifier("RunMat:Placement:ResourceOverflow")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn cancelled() -> crate::RuntimeError {
    crate::build_runtime_error("scalar POWER cancelled before admission")
        .with_identifier("RunMat:Cancelled")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
