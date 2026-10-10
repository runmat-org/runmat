//! Handle-free POWER geometry and physical recipe shared by ordinary resident
//! preparation and a future whole-work owner. This cannot authenticate an
//! allocation, reserve resources, or authorize execution by itself.
use crate::{
    AccelProvider, NumericalDomainEvidence, PowerResultDomain, PowerScalarControl,
    ProviderCapabilitySnapshot, ProviderElementType, ProviderFeasibilityQuery, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload, RealPowerArithmeticMode,
    RealPowerOperand,
};

use super::validate::PowerServiceError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerScalarConstruction {
    ExactTypedUpload,
    ProviderFill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerPhysicalRoute {
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

/// Requested representations of the real provider intermediates. These are
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
    pub fn scalar_shape(&self) -> &[usize] {
        &self.host_shape
    }
    pub fn operation(&self) -> &ProviderOperationIdentity {
        &self.query.operation
    }

    pub fn physical_recipe(&self) -> PowerPhysicalRecipe {
        let mixed = matches!(
            self.mode,
            RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
        );
        PowerPhysicalRecipe {
            route: self.route,
            mode: self.mode,
            widen_resident: mixed && self.resident.element_type == ProviderElementType::F32,
            scalar: if self.exact_scalar_upload() {
                PowerScalarConstruction::ExactTypedUpload
            } else {
                PowerScalarConstruction::ProviderFill
            },
            single_publication: mixed,
        }
    }

    pub(crate) fn exact_scalar_upload(&self) -> bool {
        let mixed = matches!(
            self.mode,
            RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
        );
        (!mixed
            && self.control.precision() == ProviderPrecision::F32
            && self.provider_precision == ProviderPrecision::F64)
            || (self.control.value() == 0.0 && self.control.value().is_sign_negative())
            || self.control.value().is_nan()
    }

    pub fn feasibility_query(&self, domain: PowerResultDomain) -> Option<ProviderFeasibilityQuery> {
        if self.route == PowerPhysicalRoute::LegacyProvenReal {
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
            PowerScalarConstruction::ExactTypedUpload => device_rep(
                if recipe.single_publication {
                    ProviderElementType::F64
                } else {
                    self.control.precision().into()
                },
                ProviderStorage::DenseReal,
                vec![1, 1],
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

fn derive(
    provider_precision: ProviderPrecision,
    capability: ProviderCapabilitySnapshot,
    resident: ProviderRepresentation,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
) -> Result<Option<PowerSymbolicPlan>, PowerServiceError> {
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
            ..ProviderWorkload::default()
        },
    };
    let advertised = capability
        .operations
        .iter()
        .any(|candidate| candidate.identity == query.operation && candidate.family == query.family);
    let route = if advertised {
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

/// Effect-free admission for a hypothetical future resident representation.
/// `Some` is a physical query/recipe, not source binding, owner proof, or a
/// placement decision. The live-handle preparation must compare it later.
pub fn plan_resident_scalar_power(
    provider: &dyn AccelProvider,
    resident: ProviderRepresentation,
    control: PowerScalarControl,
    resident_operand: RealPowerOperand,
    host_shape: &[usize],
    operation: ProviderOperationIdentity,
) -> Result<Option<PowerSymbolicPlan>, PowerServiceError> {
    let Some(plan) = derive(
        provider.precision(),
        provider.capability_snapshot(),
        resident,
        control,
        resident_operand,
        host_shape,
        operation,
    )?
    else {
        return Ok(None);
    };
    if plan.route == PowerPhysicalRoute::TypedDomain {
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

pub(super) fn legacy_real_proof(resident_operand: RealPowerOperand, control: f64) -> bool {
    match resident_operand {
        RealPowerOperand::Base => {
            PowerResultDomain::real_for_any_real_base(control) && control != 0.0
        }
        RealPowerOperand::Exponent => control.is_finite() && control > 0.0 && control != 1.0,
    }
}
