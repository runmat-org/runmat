use crate::{
    handle_integer_type, handle_is_logical, handle_is_transposed, handle_precision,
    provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    NumericalDomainEvidence, PowerResultDomain, PowerScalarControl, ProviderCapabilitySnapshot,
    ProviderExecutionError, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation, ProviderResidency,
    ProviderStorage, RealPowerArithmeticMode, RealPowerOperand,
};

use super::symbolic::{legacy_real_proof, validate_scalar_shape, PowerSymbolicPlan};
pub use super::symbolic::{PowerPhysicalRecipe, PowerPhysicalRoute, PowerScalarConstruction};
use super::validate::PowerServiceError;

/// An exact, pre-effect scalar program. The retained resident reference keeps
/// its allocation lease alive across asynchronous preparation and execution.
pub struct PreparedResidentScalarPower {
    pub(crate) provider: &'static dyn AccelProvider,
    pub(crate) resident: GpuTensorHandle,
    pub(crate) control: PowerScalarControl,
    pub(crate) resident_operand: RealPowerOperand,
    pub(crate) mode: RealPowerArithmeticMode,
    pub(crate) route: PowerPhysicalRoute,
    pub(crate) host_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    capability: ProviderCapabilitySnapshot,
    query: ProviderFeasibilityQuery,
    pub(super) symbolic: PowerSymbolicPlan,
    protected: Vec<GpuTensorHandle>,
}

impl PreparedResidentScalarPower {
    /// Retain additional borrowed allocations that a surrounding selected
    /// owner has already published or still needs. Every POWER intermediate
    /// must be distinct from these leases; this never transfers ownership.
    pub fn protect_borrowed(
        mut self,
        handles: &[&GpuTensorHandle],
    ) -> Result<Self, PowerServiceError> {
        for handle in handles {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|error| PowerServiceError(ProviderExecutionError::from_source(error)))?;
            if handle.device_id() != self.provider.device_id()
                || !provider_for_handle(handle)
                    .is_some_and(|owner| same_provider_instance(owner, self.provider))
            {
                return Err(PowerServiceError::pre(
                    "protected POWER lease has another owner",
                ));
            }
            self.protected.push((*handle).clone());
        }
        Ok(self)
    }

    pub(crate) fn protected_with<'a>(
        &'a self,
        inputs: &[&'a GpuTensorHandle],
    ) -> Vec<&'a GpuTensorHandle> {
        inputs
            .iter()
            .copied()
            .chain(self.protected.iter())
            .collect()
    }

    pub fn mode(&self) -> RealPowerArithmeticMode {
        self.mode
    }
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    pub fn control(&self) -> PowerScalarControl {
        self.control
    }
    pub fn physical_recipe(&self) -> PowerPhysicalRecipe {
        self.symbolic.physical_recipe()
    }
    pub(crate) fn exact_scalar_upload(&self) -> bool {
        self.symbolic.exact_scalar_upload()
    }
    /// Exact logical inputs and arithmetic mode for the existing placement
    /// planner. The planner, not this provider-local service, owns costing.
    pub fn feasibility_query(&self, domain: PowerResultDomain) -> Option<ProviderFeasibilityQuery> {
        self.symbolic.feasibility_query(domain)
    }

    pub fn symbolic_plan(&self) -> &PowerSymbolicPlan {
        &self.symbolic
    }

    pub(crate) fn validate_live(&self) -> Result<(), PowerServiceError> {
        for handle in &self.protected {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|error| PowerServiceError(ProviderExecutionError::from_source(error)))?;
            if handle.device_id() != self.provider.device_id()
                || !provider_for_handle(handle)
                    .is_some_and(|owner| same_provider_instance(owner, self.provider))
            {
                return Err(PowerServiceError::pre(
                    "protected POWER lease changed owner",
                ));
            }
        }
        let resident = &self.resident;
        if resident.allocation_lease().validate_live().is_err()
            || resident.device_id() != self.provider.device_id()
            || !provider_for_handle(resident)
                .is_some_and(|owner| same_provider_instance(owner, self.provider))
            || resident.physical().storage() != GpuTensorStorage::Real
            || handle_is_logical(resident)
            || handle_integer_type(resident).is_some()
        {
            return Err(PowerServiceError::pre(
                "prepared POWER resident is no longer live",
            ));
        }
        let precision = handle_precision(resident)
            .ok_or_else(|| PowerServiceError::pre("prepared POWER resident class changed"))?;
        let resident_class = match precision {
            ProviderPrecision::F32 => runmat_types::NumericClass::Single,
            ProviderPrecision::F64 => runmat_types::NumericClass::Double,
        };
        let control_class = match self.control.precision() {
            ProviderPrecision::F32 => runmat_types::NumericClass::Single,
            ProviderPrecision::F64 => runmat_types::NumericClass::Double,
        };
        let classes = match self.resident_operand {
            RealPowerOperand::Base => (resident_class, control_class),
            RealPowerOperand::Exponent => (control_class, resident_class),
        };
        let host = ProviderRepresentation {
            element_type: self.control.precision().into(),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: self
                .host_shape
                .iter()
                .map(|&n| u64::try_from(n))
                .collect::<Result<_, _>>()
                .map_err(PowerServiceError::from)?,
            residency: ProviderResidency::Host,
        };
        let resident_rep = representation(resident)?;
        let expected_inputs = match self.resident_operand {
            RealPowerOperand::Base => vec![resident_rep, host],
            RealPowerOperand::Exponent => vec![host, resident_rep],
        };
        if RealPowerArithmeticMode::for_classes(classes.0, classes.1) != Some(self.mode)
            || representation(resident)? != self.symbolic.resident
            || self.control != self.symbolic.control
            || self.resident_operand != self.symbolic.resident_operand
            || self.host_shape != self.symbolic.host_shape
            || self.output_shape != self.symbolic.output_shape
            || self.mode != self.symbolic.mode
            || self.route != self.symbolic.route
            || self.query != self.symbolic.query
            || self.query.workload.real_power_mode != Some(self.mode)
            || self.query.family != ProviderOperationFamily::Elementwise
            || self.query.inputs != expected_inputs
            || self.output_shape
                != if self.route == PowerPhysicalRoute::LegacyProvenReal {
                    resident.shape().to_vec()
                } else {
                    broadcast_shape(resident.shape(), &self.host_shape)?
                }
            || (self.route == PowerPhysicalRoute::LegacyProvenReal
                && (precision != self.control.precision()
                    || (precision == ProviderPrecision::F32
                        && self.provider.precision() == ProviderPrecision::F64)
                    || !legacy_real_proof(self.resident_operand, self.control.value())))
        {
            return Err(PowerServiceError::pre(
                "prepared POWER arithmetic facts changed",
            ));
        }
        let capability = self.provider.capability_snapshot();
        if capability != self.capability {
            return Err(PowerServiceError::pre(
                "prepared POWER provider capability changed",
            ));
        }
        if self
            .symbolic
            .resident_input
            .as_ref()
            .is_some_and(|input| !input.matches(resident))
        {
            return Err(PowerServiceError::pre(
                "planned POWER input description changed",
            ));
        }
        if self.route == PowerPhysicalRoute::ExactSingleBase {
            let current = self.symbolic.replan(self.provider)?;
            if current.as_ref() != Some(&self.symbolic) {
                return Err(PowerServiceError::pre("certified POWER capability changed"));
            }
        }
        if self.route == PowerPhysicalRoute::TypedDomain {
            let mut query = self.query.clone();
            let support = [PowerResultDomain::Real, PowerResultDomain::Complex].map(|domain| {
                query.workload.numerical_domain = NumericalDomainEvidence::RealPower(domain);
                self.provider.query_feasibility(&query).is_supported()
            });
            let supported = if matches!(
                self.mode,
                RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
            ) {
                support.into_iter().all(|yes| yes)
            } else {
                support.into_iter().any(|yes| yes)
            };
            if !supported {
                return Err(PowerServiceError::pre("prepared POWER feasibility changed"));
            }
        }
        Ok(())
    }
}

/// Return `None` only before provider effects, when this exact physical route
/// cannot be admitted. The operation identity is supplied by the caller's
/// canonical binding, never inferred from an API-side spelling table.
pub fn prepare_resident_scalar_power(
    provider: &'static dyn AccelProvider,
    resident: &GpuTensorHandle,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
) -> Result<Option<PreparedResidentScalarPower>, PowerServiceError> {
    validate_scalar_shape(host_shape)?;
    validate_owner(provider, resident)?;
    if resident.physical().storage() != GpuTensorStorage::Real
        || handle_is_logical(resident)
        || handle_integer_type(resident).is_some()
    {
        return Ok(None);
    }
    let symbolic = super::symbolic::plan_with_input(
        provider,
        representation(resident)?,
        control,
        resident_operand,
        host_shape,
        operation,
        Some(crate::PowerInputDescription::from_handle(resident)),
    )?;
    Ok(symbolic.map(|symbolic| prepared_from_plan(provider, resident, symbolic)))
}

/// Reconcile a pre-effect symbolic whole-work receipt with the actual live
/// intermediate. A mismatch rejects before this POWER service submits work;
/// it never silently replans a different route at the stage boundary.
pub fn prepare_resident_scalar_power_against_plan(
    provider: &'static dyn AccelProvider,
    resident: &GpuTensorHandle,
    planned: &PowerSymbolicPlan,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
) -> Result<PreparedResidentScalarPower, PowerServiceError> {
    if control != planned.control
        || resident_operand != planned.resident_operand
        || host_shape != planned.host_shape.as_slice()
        || operation != planned.query.operation
    {
        return Err(PowerServiceError::pre(
            "planned POWER source control changed",
        ));
    }
    validate_owner(provider, resident)?;
    if resident.physical().storage() != GpuTensorStorage::Real
        || handle_is_logical(resident)
        || handle_integer_type(resident).is_some()
    {
        return Err(PowerServiceError::pre(
            "planned POWER resident class changed",
        ));
    }
    let actual = representation(resident)?;
    if actual != planned.resident || provider.precision() != planned.provider_precision {
        return Err(PowerServiceError::pre(
            "planned POWER resident view changed",
        ));
    }
    if planned
        .resident_input
        .as_ref()
        .is_some_and(|input| !input.matches(resident))
    {
        return Err(PowerServiceError::pre(
            "planned POWER input description changed",
        ));
    }
    // Revalidate the future allocator projection before replacing descriptive
    // geometry with authentic live input facts. All other receipt fields must
    // remain unchanged across this transition.
    if planned.replan(provider)?.as_ref() != Some(planned) {
        return Err(PowerServiceError::pre(
            "planned POWER allocator route changed",
        ));
    }
    let Some(mut replanned) = super::symbolic::plan_with_input(
        provider,
        actual,
        control,
        resident_operand,
        host_shape,
        operation,
        Some(crate::PowerInputDescription::from_handle(resident)),
    )?
    else {
        return Err(PowerServiceError::pre(
            "planned POWER live route unavailable",
        ));
    };
    let live_input = replanned
        .query
        .workload
        .real_power_scalar
        .as_mut()
        .and_then(|work| {
            std::mem::replace(
                &mut work.input,
                planned
                    .query
                    .workload
                    .real_power_scalar
                    .as_ref()
                    .and_then(|work| work.input.clone()),
            )
        });
    let live_resident_input = std::mem::replace(
        &mut replanned.resident_input,
        planned.resident_input.clone(),
    );
    if &replanned != planned {
        return Err(PowerServiceError::pre(
            "planned POWER physical route changed",
        ));
    }
    if let Some(work) = &mut replanned.query.workload.real_power_scalar {
        work.input = live_input;
    }
    replanned.resident_input = live_resident_input;
    Ok(prepared_from_plan(provider, resident, replanned))
}

fn validate_owner(
    provider: &'static dyn AccelProvider,
    resident: &GpuTensorHandle,
) -> Result<(), PowerServiceError> {
    if !provider_for_handle(resident).is_some_and(|owner| same_provider_instance(owner, provider))
        || resident.device_id() != provider.device_id()
        || resident.allocation_lease().validate_live().is_err()
    {
        return Err(PowerServiceError::pre(
            "resident has no live exact provider owner",
        ));
    }
    Ok(())
}

fn prepared_from_plan(
    provider: &'static dyn AccelProvider,
    resident: &GpuTensorHandle,
    symbolic: PowerSymbolicPlan,
) -> PreparedResidentScalarPower {
    PreparedResidentScalarPower {
        provider,
        resident: resident.clone(),
        control: symbolic.control,
        resident_operand: symbolic.resident_operand,
        mode: symbolic.mode,
        route: symbolic.route,
        host_shape: symbolic.host_shape.clone(),
        output_shape: symbolic.output_shape.clone(),
        capability: symbolic.capability.clone(),
        query: symbolic.query.clone(),
        symbolic,
        protected: Vec::new(),
    }
}

fn representation(handle: &GpuTensorHandle) -> Result<ProviderRepresentation, PowerServiceError> {
    let precision = handle_precision(handle)
        .ok_or_else(|| PowerServiceError::pre("resident precision is missing"))?;
    Ok(ProviderRepresentation {
        element_type: precision.into(),
        storage: ProviderStorage::DenseReal,
        layout: if handle_is_transposed(handle) {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        shape: handle
            .shape()
            .iter()
            .map(|&n| u64::try_from(n))
            .collect::<Result<_, _>>()
            .map_err(PowerServiceError::from)?,
        residency: ProviderResidency::Device,
    })
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, PowerServiceError> {
    runmat_types::geometry::broadcast_concrete_shape(a, b)
        .map_err(|_| PowerServiceError::pre("incompatible POWER shapes"))
}
