//! Physical feasibility of the already-admitted ordinary sum route.
//! Incomplete kernel inventories remain unknown for automatic profitability;
//! direct resident hooks retain their own independently admitted behavior.
use super::*;
use runmat_accelerate_api::ProviderSumWorkload;
use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumInvocation, SumOutputClass};

mod floating;
mod global;
mod integer;
mod omit;
#[cfg(test)]
mod tests;

pub(crate) struct SumWork {
    pub estimate: ProviderResourceEstimate,
    #[cfg(feature = "wgpu")]
    pub kernel: SumKernelRoute,
    #[cfg(feature = "wgpu")]
    pub input_cast: bool,
    #[cfg(feature = "wgpu")]
    pub integer_input_cast: bool,
    #[cfg(feature = "wgpu")]
    pub output_cast: bool,
    #[cfg(feature = "wgpu")]
    pub nan_map: bool,
    #[cfg(feature = "wgpu")]
    pub scalar_visits: u64,
    #[cfg(feature = "wgpu")]
    pub queue_submissions: u32,
    #[cfg(feature = "wgpu")]
    pub allocations: u32,
    #[cfg(feature = "wgpu")]
    pub uniform_bytes: u64,
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum SumKernelRoute {
    Host,
    Global,
    Dimension2,
    DimensionN,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    for invocation in [SumInvocation::Sum, SumInvocation::NanSum] {
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(
                runmat_builtins::sum_invocation_entry(invocation)
                    .identity
                    .name,
            ),
            family: ProviderOperationFamily::Reduction,
        });
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    analyze(provider, wgpu, query).map(|result| match result {
        Ok(work) => ProviderFeasibility::supported(work.estimate),
        Err(rejection) => rejection,
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<SumWork> {
    analyze(provider, true, query)?.ok()
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    let Some(work) = physical_work(provider, query) else {
        return ProviderObservationCapability::Unsupported;
    };
    if work.queue_submissions == 0 {
        return ProviderObservationCapability::Unsupported;
    }
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions: work.queue_submissions,
        requested_peak_buffer_bytes: u64::from(work.queue_submissions)
            * crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
        clear_commands: work.queue_submissions,
        map_callbacks: work.queue_submissions,
        wgpu_compute_query: (work.kernel != SumKernelRoute::Host)
            .then(|| {
                crate::backend::wgpu::dispatch::observation::prospective_encoder_query(
                    // Omit-NaN adds a real no-op compute warmup encoder;
                    // unlike a transfer/flush submission it admits a query.
                    work.queue_submissions,
                    work.queue_submissions,
                )
            })
            .flatten(),
    })
}

fn analyze(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<Result<SumWork, ProviderFeasibility>> {
    if ![SumInvocation::Sum, SumInvocation::NanSum]
        .into_iter()
        .any(|invocation| {
            query.operation.0
                == runmat_builtins::sum_invocation_entry(invocation)
                    .identity
                    .name
        })
    {
        return None;
    }
    if let Some(route) = query.workload.sum.as_ref() {
        if query.operation.0
            != runmat_builtins::sum_invocation_entry(route.invocation)
                .identity
                .name
        {
            return Some(Err(ProviderFeasibility::rejected(
                ProviderRejectionCode::InvalidShape,
                "sum.invocation_identity_mismatch",
            )));
        }
    }
    Some(work(provider, wgpu, query))
}

fn work(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Result<SumWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Reduction
        || query.inputs.len() != 1
        || !query.outputs.is_empty()
    {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "sum.operand_or_family",
        ));
    }
    let Some(route) = query.workload.sum.as_ref() else {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.missing_admitted_route",
        ));
    };
    let input = &query.inputs[0];
    let Some(elements) = input.checked_element_count() else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "sum.shape_overflow",
        ));
    };
    if query
        .workload
        .elements
        .is_some_and(|declared| declared != elements)
        || !valid_controls(input, route)
    {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "sum.admitted_geometry_or_class",
        ));
    }
    let mut output_shape = input.shape.clone();
    for &axis in &route.axes {
        let Some(extent) = output_shape.get_mut(
            usize::try_from(axis)
                .map_err(|_| reject(ProviderRejectionCode::InvalidShape, "sum.axis_overflow"))?,
        ) else {
            return Err(reject(
                ProviderRejectionCode::InvalidShape,
                "sum.axis_out_of_bounds",
            ));
        };
        *extent = 1;
    }
    let Some(output_elements) = output_shape
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "sum.output_overflow",
        ));
    };
    let lanes = match route.input_domain {
        NumericDomain::Real => 1_u64,
        NumericDomain::Complex => 2_u64,
    };
    let Some(output_bytes) = output_elements
        .checked_mul(route.publication_class.byte_width() as u64)
        .and_then(|bytes| bytes.checked_mul(lanes))
    else {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.output_bytes_overflow",
        ));
    };
    if !wgpu {
        // InProcess executes the typed hook synchronously. Shape proves the
        // retained output size, not its temporary host allocations.
        return Ok(SumWork {
            estimate: ProviderResourceEstimate {
                output_bytes: Some(output_bytes),
                transient_bytes: None,
                dispatches: Some(0),
            },
            #[cfg(feature = "wgpu")]
            kernel: SumKernelRoute::Host,
            #[cfg(feature = "wgpu")]
            input_cast: false,
            #[cfg(feature = "wgpu")]
            integer_input_cast: false,
            #[cfg(feature = "wgpu")]
            output_cast: false,
            #[cfg(feature = "wgpu")]
            nan_map: false,
            #[cfg(feature = "wgpu")]
            scalar_visits: elements,
            #[cfg(feature = "wgpu")]
            queue_submissions: 0,
            #[cfg(feature = "wgpu")]
            allocations: 0,
            #[cfg(feature = "wgpu")]
            uniform_bytes: 0,
        });
    }
    if provider.precision() != ProviderPrecision::F64 {
        return Err(reject(
            ProviderRejectionCode::UnsupportedElementType,
            "sum.double_accumulator_requires_f64",
        ));
    }
    if !provider.supports_shader_requirements(
        runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords,
    ) {
        return Err(reject(
            ProviderRejectionCode::UnsupportedElementType,
            "sum.selected_f64_reduction_requires_ieee_words",
        ));
    }
    // Host-origin allocation geometry is known. Resident views need the
    // provider's authenticated backing shape before this physical recipe can
    // represent them; an apparent contiguous public layout is insufficient.
    if route.accumulation_class != NumericClass::Double
        || route.input_domain != NumericDomain::Real
        || input.storage != ProviderStorage::DenseReal
        // A logical public element does not specify this adapter's physical
        // word width. Do not price its host upload as one byte per element.
        || input.element_type == ProviderElementType::Logical
        || input.layout != ProviderLayout::ColumnMajorContiguous
        || input.residency != ProviderResidency::Host
        || output_elements == 0
    {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.physical_recipe_unresolved",
        ));
    }
    if route.input_class.integer_class().is_some() {
        return integer::work(
            provider,
            input,
            route,
            elements,
            output_elements,
            output_bytes,
        );
    }
    match route.missing {
        ReductionNaN::Include => floating::work(
            provider,
            input,
            route,
            elements,
            output_elements,
            output_bytes,
        ),
        ReductionNaN::Omit => omit::work(
            provider,
            input,
            route,
            elements,
            output_elements,
            output_bytes,
        ),
    }
}

fn valid_controls(input: &ProviderRepresentation, route: &ProviderSumWorkload) -> bool {
    if route.axes.windows(2).any(|pair| pair[0] >= pair[1])
        || route
            .axes
            .iter()
            .any(|&axis| axis >= input.shape.len() as u64)
        || route.publication_class != route.output.resolve_numeric_class(route.input_class)
    {
        return false;
    }
    let expected_accumulation =
        if route.output == SumOutputClass::Native && route.input_class.integer_class().is_some() {
            route.input_class
        } else {
            NumericClass::Double
        };
    if route.accumulation_class != expected_accumulation {
        return false;
    }
    match (route.input_domain, input.storage, input.element_type) {
        (NumericDomain::Real, ProviderStorage::DenseReal, ProviderElementType::Logical) => {
            route.input_class == NumericClass::Double
        }
        (NumericDomain::Real, ProviderStorage::DenseReal, element) => {
            element == element_type(route.input_class)
        }
        (NumericDomain::Complex, ProviderStorage::DenseComplexInterleaved, element) => {
            matches!(
                (route.input_class, element),
                (NumericClass::Single, ProviderElementType::ComplexF32)
                    | (NumericClass::Double, ProviderElementType::ComplexF64)
            )
        }
        _ => false,
    }
}

fn element_type(class: NumericClass) -> ProviderElementType {
    match class {
        NumericClass::Double => ProviderElementType::F64,
        NumericClass::Single => ProviderElementType::F32,
        NumericClass::Int8 => ProviderElementType::I8,
        NumericClass::UInt8 => ProviderElementType::U8,
        NumericClass::Int16 => ProviderElementType::I16,
        NumericClass::UInt16 => ProviderElementType::U16,
        NumericClass::Int32 => ProviderElementType::I32,
        NumericClass::UInt32 => ProviderElementType::U32,
        NumericClass::Int64 => ProviderElementType::I64,
        NumericClass::UInt64 => ProviderElementType::U64,
    }
}
