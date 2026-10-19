//! Handle-free POWER geometry and physical recipe shared by ordinary resident
//! preparation and a future whole-work owner. This cannot authenticate an
//! allocation, reserve resources, or authorize execution by itself.
use crate::{
    AccelProvider, GpuTensorHandle, NumericalDomainEvidence, PowerResultDomain, PowerScalarControl,
    ProviderCapabilitySnapshot, ProviderElementType, ProviderFeasibilityQuery, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload, RealPowerArithmeticMode,
    RealPowerOperand,
};

use super::validate::PowerServiceError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerScalarConstruction {
    /// Control remains in the authenticated program; no scalar buffer exists.
    EmbeddedExactControl,
    ExactTypedUpload,
    ProviderFill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerPhysicalRoute {
    ExactSingleBase,
    TypedDomain,
    LegacyProvenReal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowerPhysicalRecipe {
    pub route: PowerPhysicalRoute,
    pub mode: RealPowerArithmeticMode,
    pub widen_resident: bool,
    pub scalar: PowerScalarConstruction,
    pub single_publication: bool,
}

/// Requested representations of the real provider intermediates. Embedded
/// exact controls remain Host representations and require no scalar buffer.
/// These are
/// not peak allocation bytes, dispatch counts, or performance estimates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowerSymbolicPhysicalWork {
    pub widened_resident: Option<ProviderRepresentation>,
    pub scalar: ProviderRepresentation,
    pub native_result: ProviderRepresentation,
    pub single_publication: Option<ProviderRepresentation>,
    pub output: ProviderRepresentation,
    pub paired_domain_probe: bool,
    pub output_reshape: bool,
}

/// Opaque, capability-bound symbolic plan. Original control bits, operand role,
/// logical scalar geometry, arithmetic mode, route, and query identity are
/// retained so an eventual live view must match before effects.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerSymbolicPlan {
    pub(crate) exact_proposal: Option<crate::ProviderRealElementwisePhysicalProposal>,
    pub(crate) resident_input: Option<crate::PowerInputDescription>,
    pub(crate) resident: ProviderRepresentation,
    pub(crate) control: PowerScalarControl,
    pub(crate) resident_operand: RealPowerOperand,
    pub(crate) host_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) mode: RealPowerArithmeticMode,
    pub(crate) route: PowerPhysicalRoute,
    pub(crate) provider_precision: ProviderPrecision,
    pub(crate) capability: ProviderCapabilitySnapshot,
    pub(crate) query: ProviderFeasibilityQuery,
}

impl PowerSymbolicPlan {
    pub fn control(&self) -> PowerScalarControl {
        self.control
    }
    pub fn resident_operand(&self) -> RealPowerOperand {
        self.resident_operand
    }
    pub fn mode(&self) -> RealPowerArithmeticMode {
        self.mode
    }
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    pub fn resident_representation(&self) -> &ProviderRepresentation {
        &self.resident
    }
    pub fn resident_input_description(&self) -> Option<&crate::PowerInputDescription> {
        self.resident_input.as_ref()
    }
    /// The only permitted difference between a selected typed receipt and
    /// its prepared form is replacement of the future input description by
    /// the live source's physical/view description. This only compares
    /// descriptors: the private preparation owner authenticates and retains
    /// the actual allocation lease, and separately revalidates the selected
    /// allocator route before making that replacement.
    pub(crate) fn matches_live_typed_binding(
        &self,
        bound: &Self,
        source: &GpuTensorHandle,
    ) -> bool {
        self.route == PowerPhysicalRoute::TypedDomain
            && bound.route == PowerPhysicalRoute::TypedDomain
            && self
                .resident_input
                .as_ref()
                .is_none_or(|input| input.matches(source))
            && matches!(
                bound.resident_input.as_ref(),
                Some(crate::PowerInputDescription::Live { .. })
            )
            && bound
                .resident_input
                .as_ref()
                .is_some_and(|input| input.matches(source))
            && self.exact_proposal == bound.exact_proposal
            && self.resident == bound.resident
            && self.control == bound.control
            && self.resident_operand == bound.resident_operand
            && self.host_shape == bound.host_shape
            && self.output_shape == bound.output_shape
            && self.mode == bound.mode
            && self.provider_precision == bound.provider_precision
            && self.capability == bound.capability
            && self.query == bound.query
    }
    pub fn scalar_shape(&self) -> &[usize] {
        &self.host_shape
    }
    pub fn operation(&self) -> &ProviderOperationIdentity {
        &self.query.operation
    }
    /// Descriptive allocation/dispatch proposal, never an executable recipe.
    pub fn exact_physical_proposal(
        &self,
    ) -> Option<&crate::ProviderRealElementwisePhysicalProposal> {
        self.exact_proposal.as_ref()
    }

    pub fn physical_recipe(&self) -> PowerPhysicalRecipe {
        if self.route == PowerPhysicalRoute::ExactSingleBase {
            return PowerPhysicalRecipe {
                route: self.route,
                mode: self.mode,
                widen_resident: false,
                scalar: PowerScalarConstruction::EmbeddedExactControl,
                single_publication: false,
            };
        }
        if self.route == PowerPhysicalRoute::LegacyProvenReal {
            return PowerPhysicalRecipe {
                route: self.route,
                mode: self.mode,
                widen_resident: false,
                scalar: PowerScalarConstruction::ProviderFill,
                single_publication: false,
            };
        }
        let mixed = matches!(
            self.mode,
            RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
        );
        PowerPhysicalRecipe {
            route: self.route,
            mode: self.mode,
            widen_resident: mixed && self.resident.element_type == ProviderElementType::F32,
            scalar: PowerScalarConstruction::ExactTypedUpload,
            single_publication: mixed,
        }
    }

    pub(crate) fn exact_scalar_upload(&self) -> bool {
        self.route == PowerPhysicalRoute::TypedDomain
    }

    pub fn feasibility_query(&self, domain: PowerResultDomain) -> Option<ProviderFeasibilityQuery> {
        if self.route == PowerPhysicalRoute::LegacyProvenReal
            || (self.route == PowerPhysicalRoute::ExactSingleBase
                && domain != PowerResultDomain::Real)
        {
            return None;
        }
        let mut query = self.query.clone();
        query.workload.numerical_domain = NumericalDomainEvidence::RealPower(domain);
        Some(query)
    }

    /// Describe one possible paired-domain branch. For typed routes the
    /// caller must separately use `feasibility_query` for that domain; this
    /// geometry alone never asserts support or a numerical-domain proof.
    pub fn physical_work(&self, domain: PowerResultDomain) -> Option<PowerSymbolicPhysicalWork> {
        if self.route == PowerPhysicalRoute::ExactSingleBase {
            if domain != PowerResultDomain::Real {
                return None;
            }
            let mut output = self.resident.clone();
            output.layout = ProviderLayout::ColumnMajorContiguous;
            output.shape = self
                .output_shape
                .iter()
                .map(|&n| u64::try_from(n).ok())
                .collect::<Option<_>>()?;
            let output_reshape = output.shape != self.resident.shape;
            return Some(PowerSymbolicPhysicalWork {
                widened_resident: None,
                // This route has no scalar allocation: the representation is
                // the original host control, not a fabricated F32 intermediate.
                scalar: self.query.inputs.get(1)?.clone(),
                native_result: output.clone(),
                single_publication: None,
                output,
                paired_domain_probe: false,
                output_reshape,
            });
        }
        if self.route == PowerPhysicalRoute::LegacyProvenReal && domain != PowerResultDomain::Real {
            return None;
        }
        let recipe = self.physical_recipe();
        let native_type = if recipe.single_publication {
            ProviderElementType::F64
        } else {
            self.resident.element_type
        };
        let native_shape = self.resident.shape.clone();
        let widened_resident = recipe.widen_resident.then(|| {
            device_rep(
                native_type,
                ProviderStorage::DenseReal,
                native_shape.clone(),
            )
        });
        let scalar = match recipe.scalar {
            PowerScalarConstruction::EmbeddedExactControl => {
                unreachable!("exact route returned above")
            }
            PowerScalarConstruction::ExactTypedUpload => device_rep(
                if recipe.single_publication {
                    ProviderElementType::F64
                } else {
                    self.control.precision().into()
                },
                ProviderStorage::DenseReal,
                self.host_shape
                    .iter()
                    .map(|&extent| u64::try_from(extent).ok())
                    .collect::<Option<Vec<_>>>()?,
            ),
            PowerScalarConstruction::ProviderFill => device_rep(
                native_type,
                ProviderStorage::DenseReal,
                native_shape.clone(),
            ),
        };
        let native_output_shape = if self.route == PowerPhysicalRoute::LegacyProvenReal {
            native_shape.clone()
        } else {
            broadcast_u64(&native_shape, &scalar.shape).ok()?
        };
        let storage = match domain {
            PowerResultDomain::Real => ProviderStorage::DenseReal,
            PowerResultDomain::Complex => ProviderStorage::DenseComplexInterleaved,
        };
        let native_result = device_rep(
            match (native_type, domain) {
                (ProviderElementType::F32, PowerResultDomain::Complex) => {
                    ProviderElementType::ComplexF32
                }
                (ProviderElementType::F64, PowerResultDomain::Complex) => {
                    ProviderElementType::ComplexF64
                }
                (type_, _) => type_,
            },
            storage,
            native_output_shape.clone(),
        );
        let single_publication = recipe.single_publication.then(|| {
            device_rep(
                if domain == PowerResultDomain::Complex {
                    ProviderElementType::ComplexF32
                } else {
                    ProviderElementType::F32
                },
                storage,
                native_output_shape.clone(),
            )
        });
        let mut output = single_publication
            .clone()
            .unwrap_or_else(|| native_result.clone());
        let output_reshape = output.shape
            != self
                .output_shape
                .iter()
                .map(|&n| u64::try_from(n).ok())
                .collect::<Option<Vec<_>>>()?;
        output.shape = self
            .output_shape
            .iter()
            .map(|&n| u64::try_from(n).ok())
            .collect::<Option<Vec<_>>>()?;
        Some(PowerSymbolicPhysicalWork {
            widened_resident,
            scalar,
            native_result,
            single_publication,
            output,
            paired_domain_probe: self.route == PowerPhysicalRoute::TypedDomain,
            output_reshape,
        })
    }
}

fn device_rep(
    element_type: ProviderElementType,
    storage: ProviderStorage,
    shape: Vec<u64>,
) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    }
}

fn broadcast_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>, PowerServiceError> {
    let a = a
        .iter()
        .map(|&n| usize::try_from(n))
        .collect::<Result<Vec<_>, _>>()?;
    let b = b
        .iter()
        .map(|&n| usize::try_from(n))
        .collect::<Result<Vec<_>, _>>()?;
    let result = runmat_types::geometry::broadcast_concrete_shape(&a, &b)
        .map_err(|_| PowerServiceError::pre("incompatible POWER shapes"))?;
    result
        .into_iter()
        .map(|n| u64::try_from(n).map_err(PowerServiceError::from))
        .collect()
}

struct ExactDescription {
    proposal: crate::ProviderRealElementwisePhysicalProposal,
    input: Option<crate::PowerInputDescription>,
}

fn derive(
    provider: &dyn AccelProvider,
    resident: ProviderRepresentation,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
    exact: Option<ExactDescription>,
) -> Result<Option<PowerSymbolicPlan>, PowerServiceError> {
    let provider_precision = provider.precision();
    let capability = provider.capability_snapshot();
    let (exact_proposal, source_input) =
        exact.map_or((None, None), |exact| (Some(exact.proposal), exact.input));
    validate_scalar_shape(host_shape)?;
    if resident.residency != ProviderResidency::Device
        || resident.storage != ProviderStorage::DenseReal
        || !matches!(
            resident.layout,
            ProviderLayout::ColumnMajorContiguous | ProviderLayout::Strided
        )
        || !matches!(
            resident.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        )
        || resident.checked_element_count().is_none()
    {
        return Ok(None);
    }
    let resident_precision = if resident.element_type == ProviderElementType::F32 {
        ProviderPrecision::F32
    } else {
        ProviderPrecision::F64
    };
    let resident_class = if resident_precision == ProviderPrecision::F32 {
        runmat_types::NumericClass::Single
    } else {
        runmat_types::NumericClass::Double
    };
    let control_class = if control.precision() == ProviderPrecision::F32 {
        runmat_types::NumericClass::Single
    } else {
        runmat_types::NumericClass::Double
    };
    let (base_class, exponent_class) = match resident_operand {
        RealPowerOperand::Base => (resident_class, control_class),
        RealPowerOperand::Exponent => (control_class, resident_class),
    };
    let mode = RealPowerArithmeticMode::for_classes(base_class, exponent_class)
        .ok_or_else(|| PowerServiceError::pre("unsupported POWER operand classes"))?;
    if matches!(mode, RealPowerArithmeticMode::MixedDoubleThenSingle { .. })
        && provider_precision != ProviderPrecision::F64
        && exact_proposal.is_none()
    {
        return Ok(None);
    }
    if provider_precision == ProviderPrecision::F32
        && control.precision() == ProviderPrecision::F64
        && !control.value().is_nan()
        && f64::from(control.value() as f32) != control.value()
    {
        return Ok(None);
    }
    let output_shape = broadcast_u64(
        &resident.shape,
        &host_shape
            .iter()
            .map(|&n| u64::try_from(n))
            .collect::<Result<Vec<_>, _>>()?,
    )?
    .into_iter()
    .map(usize::try_from)
    .collect::<Result<Vec<_>, _>>()?;
    let host = ProviderRepresentation {
        element_type: control.precision().into(),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: host_shape
            .iter()
            .map(|&n| u64::try_from(n))
            .collect::<Result<_, _>>()?,
        residency: ProviderResidency::Host,
    };
    let inputs = match resident_operand {
        RealPowerOperand::Base => vec![resident.clone(), host],
        RealPowerOperand::Exponent => vec![host, resident.clone()],
    };
    let query = ProviderFeasibilityQuery {
        operation,
        family: ProviderOperationFamily::Elementwise,
        inputs,
        outputs: Vec::new(),
        workload: ProviderWorkload {
            real_power_mode: Some(mode),
            real_power_scalar: exact_proposal.as_ref().map(|_| crate::PowerScalarWorkload {
                control,
                input: source_input,
            }),
            ..ProviderWorkload::default()
        },
    };
    let advertised = capability
        .operations
        .iter()
        .any(|candidate| candidate.identity == query.operation && candidate.family == query.family);
    let route = if exact_proposal.is_some() {
        PowerPhysicalRoute::ExactSingleBase
    } else if advertised {
        PowerPhysicalRoute::TypedDomain
    } else if resident_precision == control.precision()
        && !(resident_precision == ProviderPrecision::F32
            && provider_precision == ProviderPrecision::F64)
        && legacy_real_proof(resident_operand, control.value())
    {
        PowerPhysicalRoute::LegacyProvenReal
    } else {
        return Ok(None);
    };
    let output_shape = if route == PowerPhysicalRoute::LegacyProvenReal {
        resident
            .shape
            .iter()
            .map(|&n| usize::try_from(n))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        output_shape
    };
    Ok(Some(PowerSymbolicPlan {
        exact_proposal,
        resident_input: None,
        resident,
        control,
        resident_operand,
        host_shape: host_shape.to_vec(),
        output_shape,
        mode,
        route,
        provider_precision,
        capability,
        query,
    }))
}

pub(super) fn validate_scalar_shape(host_shape: &[usize]) -> Result<(), PowerServiceError> {
    if host_shape
        .iter()
        .try_fold(1usize, |n, &extent| n.checked_mul(extent))
        != Some(1)
    {
        return Err(PowerServiceError::pre(
            "POWER scalar control has a non-scalar shape",
        ));
    }
    Ok(())
}

/// Effect-free geometry projection for a hypothetical resident representation.
/// `Some` is not full feasibility, source binding, owner proof, or placement:
/// an exact route without backing description has unknown retained peak.
/// Future dense allocator outputs should use `plan_scalar_power_for_dense_output`;
/// live preparation independently binds actual extent and checks feasibility.
pub fn plan_resident_scalar_power(
    provider: &dyn AccelProvider,
    resident: ProviderRepresentation,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
) -> Result<Option<PowerSymbolicPlan>, PowerServiceError> {
    plan_with_input(
        provider,
        resident,
        control,
        resident_operand,
        host_shape,
        operation,
        None,
    )
}

pub(super) fn plan_with_input(
    provider: &dyn AccelProvider,
    resident: ProviderRepresentation,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
    input: Option<crate::PowerInputDescription>,
) -> Result<Option<PowerSymbolicPlan>, PowerServiceError> {
    validate_scalar_shape(host_shape)?;
    // Unavailable byte geometry makes only this specialization unavailable.
    // Do not fabricate a maximum extent, or suppress an independent existing
    // typed/legacy route whose provider owns its own physical admission.
    if let Some(crate::PowerInputDescription::PlannedDense { requested_bytes }) = &input {
        if resident.layout != ProviderLayout::ColumnMajorContiguous
            || resident
                .checked_byte_len()
                .is_none_or(|logical| logical > *requested_bytes)
            || provider
                .plan_dense_real_output_buffer_bytes(&resident)
                .map_err(PowerServiceError)?
                != Some(*requested_bytes)
        {
            return Ok(None);
        }
    }
    let source_bytes = input
        .as_ref()
        .map(crate::PowerInputDescription::requested_bytes)
        .or_else(|| resident.checked_byte_len());
    let exact_proposal = source_bytes
        .and_then(|bytes| exact_single_base_request(&resident, control, resident_operand, bytes))
        .map(|request| provider.propose_real_elementwise(&request))
        .transpose()
        .map_err(PowerServiceError)?
        .flatten();
    let Some(mut plan) = derive(
        provider,
        resident,
        control,
        resident_operand,
        host_shape,
        operation,
        exact_proposal.map(|proposal| ExactDescription {
            proposal,
            input: input.clone(),
        }),
    )?
    else {
        return Ok(None);
    };
    plan.resident_input = input;
    if plan.route == PowerPhysicalRoute::ExactSingleBase {
        if plan
            .query
            .workload
            .real_power_scalar
            .as_ref()
            .is_some_and(|work| work.input.is_none())
        {
            // Preserve shape-only projection without inventing retained source
            // bytes or confusing its query with admitted finite-budget work.
            return Ok(Some(plan));
        }
        if !plan
            .feasibility_query(PowerResultDomain::Real)
            .is_some_and(|query| provider.query_feasibility(&query).is_supported())
        {
            return Ok(None);
        }
    } else if plan.route == PowerPhysicalRoute::TypedDomain {
        let supported = [PowerResultDomain::Real, PowerResultDomain::Complex].map(|domain| {
            plan.feasibility_query(domain)
                .is_some_and(|query| provider.query_feasibility(&query).is_supported())
        });
        let admitted = if matches!(
            plan.mode,
            RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
        ) {
            supported.into_iter().all(|yes| yes)
        } else {
            supported.into_iter().any(|yes| yes)
        };
        if !admitted {
            return Ok(None);
        }
    }
    Ok(Some(plan))
}

/// Plan a POWER input that will be produced by the existing dense real output
/// allocator. The extent is reprojected on every replan; no live handle or
/// allocation authority is manufactured from this future description.
pub fn plan_scalar_power_for_dense_output(
    provider: &dyn AccelProvider,
    resident: ProviderRepresentation,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
) -> Result<Option<PowerSymbolicPlan>, PowerServiceError> {
    let Some(requested_bytes) = provider
        .plan_dense_real_output_buffer_bytes(&resident)
        .map_err(PowerServiceError)?
    else {
        return Ok(None);
    };
    plan_with_input(
        provider,
        resident,
        control,
        resident_operand,
        host_shape,
        operation,
        Some(crate::PowerInputDescription::PlannedDense { requested_bytes }),
    )
}

impl PowerSymbolicPlan {
    pub fn replan(&self, provider: &dyn AccelProvider) -> Result<Option<Self>, PowerServiceError> {
        plan_with_input(
            provider,
            self.resident.clone(),
            self.control,
            self.resident_operand,
            &self.host_shape,
            self.query.operation.clone(),
            self.resident_input.clone(),
        )
    }
}

/// Description only; ProviderRealElementwiseRecipe owns live execution rights.
pub fn exact_single_base_request(
    resident: &ProviderRepresentation,
    control: PowerScalarControl,
    role: RealPowerOperand,
    source_buffer_bytes: u64,
) -> Option<crate::ProviderRealElementwiseRequest> {
    if role != RealPowerOperand::Base
        || resident.element_type != ProviderElementType::F32
        || resident.storage != ProviderStorage::DenseReal
        || resident.residency != ProviderResidency::Device
        || !(resident.layout == ProviderLayout::ColumnMajorContiguous
            || (resident.layout == ProviderLayout::Strided && resident.shape.len() == 2))
        || resident.checked_element_count().is_none()
        || crate::ExactSingleBasePower::for_control(control).is_none()
    {
        return None;
    }
    Some(crate::ProviderRealElementwiseRequest {
        operation: crate::ProviderRealElementwiseOperation::ExactSingleBasePower { control },
        source: resident.clone(),
        source_buffer_bytes,
    })
}

pub(super) fn legacy_real_proof(resident_operand: RealPowerOperand, control: f64) -> bool {
    match resident_operand {
        RealPowerOperand::Base => {
            PowerResultDomain::real_for_any_real_base(control) && control != 0.0
        }
        RealPowerOperand::Exponent => control.is_finite() && control > 0.0 && control != 1.0,
    }
}
