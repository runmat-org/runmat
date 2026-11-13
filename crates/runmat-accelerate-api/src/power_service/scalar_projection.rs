//! Pure native TypedDomain scalar POWER stages. Descriptive future extents
//! become execution authority only after independently produced live leases
//! are checked by `PowerPairProjection::bind`.
use super::{PowerPairProjection, PowerPhysicalRoute, PowerSymbolicPlan};
use crate::{
    class_preserving_upload_query, AccelProvider, PowerInputDescription, PowerResultDomain,
    ProviderExecutionResult, ProviderFloatingCastPhysicalEvidence,
    ProviderNumericUploadPhysicalEvidence, ProviderPrecision, RealPowerOperand,
};
use runmat_execution::{ResourceAccount, ResourceByteBound};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowerTypedScalarResourceBounds {
    pub retained_bytes: u64,
    pub scratch_bytes: ResourceByteBound,
    pub peak_bytes: ResourceByteBound,
    pub host_workspace_bytes: u64,
    pub dispatches: u32,
    pub queue_slots: u32,
}

/// Minted only after the actual paired probe/evaluation and final scalar
/// publication finish. A projection or provider output class cannot create it.
pub struct PowerScalarExecutionEvidence {
    projection: runmat_execution::Digest,
    source: crate::allocation::TensorViewIdentity,
}

impl PowerScalarExecutionEvidence {
    pub(crate) fn projection(&self) -> runmat_execution::Digest {
        self.projection
    }
    pub(crate) fn source(&self) -> crate::allocation::TensorViewIdentity {
        self.source
    }
    pub(crate) fn from_consumed_pair(
        projection: runmat_execution::Digest,
        source: crate::allocation::TensorViewIdentity,
        bound: &super::PowerPairCompound,
        actual: &super::PowerPairExecutionEvidence,
    ) -> Option<Self> {
        let pair = bound.identity().ok()?;
        (actual.recipe() == pair
            && actual.sources()
                == bound
                    .probe()
                    .sources()
                    .each_ref()
                    .map(crate::GpuTensorHandle::view_identity))
        .then_some(Self { projection, source })
    }
}

#[derive(Clone)]
pub struct PowerTypedScalarProjection {
    owner: &'static dyn AccelProvider,
    symbolic: PowerSymbolicPlan,
    pub widened: Option<ProviderFloatingCastPhysicalEvidence>,
    pub scalar_upload: ProviderNumericUploadPhysicalEvidence,
    pub pair: PowerPairProjection,
    pub narrow_real: Option<ProviderFloatingCastPhysicalEvidence>,
    pub narrow_complex: Option<ProviderFloatingCastPhysicalEvidence>,
}

impl std::fmt::Debug for PowerTypedScalarProjection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PowerTypedScalarProjection")
            .field("symbolic", &self.symbolic)
            .field("widened", &self.widened)
            .field("scalar_upload", &self.scalar_upload)
            .field("pair", self.pair.physical())
            .field("narrow_real", &self.narrow_real)
            .field("narrow_complex", &self.narrow_complex)
            .finish_non_exhaustive()
    }
}

impl PowerTypedScalarProjection {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        symbolic: &PowerSymbolicPlan,
    ) -> ProviderExecutionResult<Option<Self>> {
        if symbolic.physical_recipe().route != PowerPhysicalRoute::TypedDomain {
            return Ok(None);
        }
        let current = symbolic
            .replan(owner)
            .map_err(super::PowerServiceError::into_execution_error)?;
        if current.as_ref() != Some(symbolic) {
            return Ok(None);
        }
        let Some(source) = symbolic.resident_input_description().cloned() else {
            return Ok(None);
        };
        let Some(real) = symbolic.physical_work(PowerResultDomain::Real) else {
            return Ok(None);
        };
        let Some(complex) = symbolic.physical_work(PowerResultDomain::Complex) else {
            return Ok(None);
        };
        if real.widened_resident != complex.widened_resident
            || real.scalar != complex.scalar
            || !real.paired_domain_probe
            || !complex.paired_domain_probe
        {
            return Ok(None);
        }
        let Some(upload_query) = class_preserving_upload_query(vec![real.scalar.clone()]) else {
            return Ok(None);
        };
        let Some(scalar_upload) = owner.numeric_upload_physical_evidence(&upload_query) else {
            return Ok(None);
        };
        let source_bytes = real.scalar.checked_byte_len();
        if source_bytes.is_none_or(|bytes| bytes > scalar_upload.output_requested_bytes)
            || scalar_upload.completion_queue_slots == 0
        {
            return Ok(None);
        }
        let widened = real.widened_resident.as_ref().and_then(|output| {
            owner
                .floating_cast_physical_evidence(
                    symbolic.resident_representation(),
                    ProviderPrecision::F64,
                )
                .filter(|stage| {
                    output
                        .checked_byte_len()
                        .is_some_and(|bytes| bytes <= stage.output_requested_bytes)
                })
        });
        if real.widened_resident.is_some() != widened.is_some() {
            return Ok(None);
        }
        let native = real
            .widened_resident
            .clone()
            .unwrap_or_else(|| symbolic.resident_representation().clone());
        let native_description = if let Some(stage) = widened {
            PowerInputDescription::PlannedDense {
                requested_bytes: stage.output_requested_bytes,
            }
        } else {
            source
        };
        let scalar_description = PowerInputDescription::PlannedDense {
            requested_bytes: scalar_upload.output_requested_bytes,
        };
        let (inputs, descriptions) = match symbolic.resident_operand() {
            RealPowerOperand::Base => (
                [native, real.scalar.clone()],
                [native_description, scalar_description],
            ),
            RealPowerOperand::Exponent => (
                [real.scalar.clone(), native],
                [scalar_description, native_description],
            ),
        };
        let Some(pair) = PowerPairProjection::plan(owner, inputs, descriptions)? else {
            return Ok(None);
        };
        if pair.physical().real.output != real.native_result
            || pair.physical().complex.output != complex.native_result
        {
            return Ok(None);
        }
        let narrow = |domain: PowerResultDomain| {
            let (input, output) = match domain {
                PowerResultDomain::Real => (&real.native_result, &real.single_publication),
                PowerResultDomain::Complex => (&complex.native_result, &complex.single_publication),
            };
            output.as_ref().and_then(|output| {
                owner
                    .floating_cast_physical_evidence(input, ProviderPrecision::F32)
                    .filter(|stage| {
                        output
                            .checked_byte_len()
                            .is_some_and(|bytes| bytes <= stage.output_requested_bytes)
                    })
            })
        };
        let narrow_real = narrow(PowerResultDomain::Real);
        let narrow_complex = narrow(PowerResultDomain::Complex);
        if real.single_publication.is_some() != narrow_real.is_some()
            || complex.single_publication.is_some() != narrow_complex.is_some()
        {
            return Ok(None);
        }
        Ok(Some(Self {
            owner,
            symbolic: symbolic.clone(),
            widened,
            scalar_upload,
            pair,
            narrow_real,
            narrow_complex,
        }))
    }

    pub fn matches(&self, owner: &dyn AccelProvider, symbolic: &PowerSymbolicPlan) -> bool {
        crate::same_provider_instance(self.owner, owner) && &self.symbolic == symbolic
    }
    /// Compare the selected, future-backed projection with a prepared live
    /// descriptor without changing the selected recipe. The private prepared
    /// POWER owner, not these descriptions, authenticates the retained lease.
    pub fn matches_bound(
        &self,
        owner: &dyn AccelProvider,
        symbolic: &PowerSymbolicPlan,
        source: &crate::GpuTensorHandle,
    ) -> bool {
        crate::same_provider_instance(self.owner, owner)
            && self.source_matches(source)
            && self.symbolic.matches_live_typed_binding(symbolic, source)
    }
    pub fn is_current_for(&self, owner: &dyn AccelProvider) -> bool {
        crate::same_provider_instance(self.owner, owner)
            && Self::plan(self.owner, &self.symbolic)
                .ok()
                .flatten()
                .and_then(|current| current.identity().ok())
                == self.identity().ok()
            && self.identity().is_ok()
    }

    pub fn source_matches(&self, source: &crate::GpuTensorHandle) -> bool {
        source.allocation_lease().validate_live().is_ok()
            && source.view().validate(source.physical()).is_ok()
            && source.device_id() == self.owner.device_id()
            && crate::provider_for_handle(source)
                .is_some_and(|owner| crate::same_provider_instance(owner, self.owner))
            && self
                .symbolic
                .resident_input
                .as_ref()
                .is_some_and(|description| description.matches(source))
    }

    pub fn real_output_requested_bytes(&self) -> u64 {
        self.narrow_real.map_or(
            self.pair.physical().real.work.output_buffer_bytes,
            |stage| stage.output_requested_bytes,
        )
    }

    /// Newly registered handles on one selected domain path. Probe-internal
    /// scratch is charged by its physical proposal, not as a Tensor handle.
    pub fn registered_output_count(&self) -> Option<usize> {
        2_usize
            .checked_add(usize::from(self.widened.is_some()))?
            .checked_add(usize::from(
                self.narrow_real.is_some() || self.narrow_complex.is_some(),
            ))
    }

    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        Ok(runmat_execution::Digest::sha256_serialized(&(
            "power-typed-scalar-projection-v1",
            &self.symbolic.resident,
            &self.symbolic.resident_input,
            self.symbolic.control,
            self.symbolic.resident_operand,
            &self.symbolic.host_shape,
            self.symbolic.mode,
            self.widened,
            self.scalar_upload,
            self.pair.identity()?,
            self.narrow_real,
            self.narrow_complex,
        ))?)
    }

    /// Conservative complete-stage overlap: prior allocations remain charged
    /// through final publication even when their actual leases retire sooner.
    /// An unavailable upload scratch bound remains unavailable, never zero.
    pub fn resource_bounds(&self) -> Option<PowerTypedScalarResourceBounds> {
        let cast_bytes = |stage: ProviderFloatingCastPhysicalEvidence| {
            stage
                .output_requested_bytes
                .checked_add(stage.uniform_buffer_bytes)
        };
        let widen = match self.widened {
            Some(stage) => Some(cast_bytes(stage)?),
            None => None,
        };
        let upload = ResourceAccount::default()
            .checked_add(self.scalar_upload.output_requested_bytes.into())?
            .checked_add(self.scalar_upload.provider_scratch_bytes)?;
        let probe = self
            .pair
            .physical()
            .probe
            .phase_data_limits
            .into_iter()
            .max()?;
        let continuation = [
            &self.pair.physical().real.work,
            &self.pair.physical().complex.work,
        ]
        .into_iter()
        .map(|work| work.requested_incremental_peak_bytes())
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
        let narrow = [self.narrow_real, self.narrow_complex]
            .into_iter()
            .flatten()
            .map(cast_bytes)
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let peak = upload
            .checked_add(widen.unwrap_or(0).into())?
            .checked_add(probe.into())?
            .checked_add(continuation.into())?
            .checked_add(narrow.into())?;
        let retained_bytes = [
            self.narrow_real.map_or(
                self.pair.physical().real.work.output_buffer_bytes,
                |stage| stage.output_requested_bytes,
            ),
            self.narrow_complex.map_or(
                self.pair.physical().complex.work.output_buffer_bytes,
                |stage| stage.output_requested_bytes,
            ),
        ]
        .into_iter()
        .max()?;
        let scratch_bytes = peak.subtract(retained_bytes.into())?.bound();
        let queue_slots = self
            .widened
            .map_or(0, |stage| stage.queue_slots)
            .checked_add(self.scalar_upload.completion_queue_slots)?
            .checked_add(self.pair.physical().probe.queue_submissions)?
            .checked_add(
                self.pair
                    .physical()
                    .real
                    .work
                    .queue_submissions
                    .max(self.pair.physical().complex.work.queue_submissions),
            )?
            .checked_add(
                self.narrow_real
                    .map_or(0, |stage| stage.queue_slots)
                    .max(self.narrow_complex.map_or(0, |stage| stage.queue_slots)),
            )?;
        let dispatches = self
            .widened
            .map_or(0, |stage| stage.dispatches)
            .checked_add(self.pair.physical().probe.dispatches)?
            .checked_add(
                self.pair
                    .physical()
                    .real
                    .work
                    .dispatches
                    .max(self.pair.physical().complex.work.dispatches),
            )?
            .checked_add(
                self.narrow_real
                    .map_or(0, |stage| stage.dispatches)
                    .max(self.narrow_complex.map_or(0, |stage| stage.dispatches)),
            )?;
        Some(PowerTypedScalarResourceBounds {
            retained_bytes,
            scratch_bytes,
            peak_bytes: peak.bound(),
            host_workspace_bytes: self.pair.physical().probe.host_workspace_bytes,
            dispatches,
            queue_slots,
        })
    }
}
