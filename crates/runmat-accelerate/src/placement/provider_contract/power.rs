//! Power capability is conditional on request-owned numerical-domain evidence.
use super::*;
use runmat_accelerate_api::{
    NumericalDomainEvidence, PowerResultDomain, RealPowerArithmeticMode, RealPowerOperand,
};

#[path = "power/mixed.rs"]
mod mixed;

#[derive(Clone, Copy)]
pub(crate) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

pub(crate) struct PowerWork {
    physical_output_bytes: Option<u64>,
    #[cfg(feature = "wgpu")]
    pub(crate) elements: u64,
    pub(crate) temporary_bytes: u64,
    pub(crate) unknown_peak: bool,
    #[cfg(feature = "wgpu")]
    pub(crate) readback_bytes: u64,
    pub(crate) dispatches: u32,
    output: ProviderRepresentation,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.push(ProviderCapabilityOperation {
        identity: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
    });
}

pub(super) fn feasibility(
    profile: HookProfile,
    precision: ProviderPrecision,
    provider: &(impl AccelProvider + ?Sized),
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 != runmat_builtins::POWER_CATALOG_ENTRY.identity.name {
        return None;
    }
    let work = match work(profile, precision, provider, query) {
        Ok(work) => work,
        Err((code, detail)) => return Some(ProviderFeasibility::rejected(code, detail)),
    };
    let mut accounted = query.clone();
    if query.workload.real_power_scalar.is_some() {
        // The certified route embeds its exact host control; no F64 buffer or
        // upload is part of the physical provider request.
        accounted.inputs.truncate(1);
    }
    // Strided matrix views are read through validated shader strides. Their
    // underlying allocations require no materialization or extra dispatch.
    for input in &mut accounted.inputs {
        if input.layout == ProviderLayout::Strided {
            input.layout = ProviderLayout::ColumnMajorContiguous;
        }
    }
    if accounted.outputs.is_empty() {
        accounted.outputs.push(work.output.clone());
    }
    // F32 on a Double-preferred WGPU owner is admitted only by an explicit
    // arithmetic mode and the profile-specific POWER recipe below. The mixed
    // mode retains its distinct Double-compute/Single-publication contract.
    let additional_elements: &[ProviderElementType] =
        if query.workload.real_power_mode.is_some_and(|mode| {
            matches!(
                mode,
                RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
                    | RealPowerArithmeticMode::NativeSingle
            )
        }) {
            &[ProviderElementType::F32, ProviderElementType::ComplexF32]
        } else {
            &[]
        };
    let result = match dense_feasibility_with_extra_elements(
        snapshot,
        &accounted,
        ProviderRepresentation::checked_byte_len,
        additional_elements,
    ) {
        ProviderFeasibility::Supported { mut estimate } => {
            if let Some(bytes) = work.physical_output_bytes {
                estimate.output_bytes = Some(bytes);
            }
            if work.unknown_peak && snapshot.max_allocation_bytes.is_some() {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnknownResourceEstimate,
                    "power.mixed.resident_peak.unknown",
                ));
            }
            let Some(total) = estimate
                .transient_bytes
                .and_then(|bytes| bytes.checked_add(work.temporary_bytes))
            else {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "power.bytes.overflow",
                ));
            };
            if !work.unknown_peak
                && snapshot
                    .max_allocation_bytes
                    .is_some_and(|maximum| total > maximum)
            {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "power.probe.memory_limit",
                ));
            }
            estimate.transient_bytes = if work.unknown_peak { None } else { Some(total) };
            estimate.dispatches = Some(work.dispatches);
            ProviderFeasibility::Supported { estimate }
        }
        rejected => rejected,
    };
    Some(result)
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: ProviderPrecision,
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<PowerWork> {
    if query.operation.0 != runmat_builtins::POWER_CATALOG_ENTRY.identity.name {
        return None;
    }
    work(HookProfile::Wgpu, precision, provider, query).ok()
}

fn work(
    profile: HookProfile,
    precision: ProviderPrecision,
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Result<PowerWork, (ProviderRejectionCode, &'static str)> {
    use ProviderRejectionCode as Rejection;
    let NumericalDomainEvidence::RealPower(domain) = query.workload.numerical_domain else {
        return Err((
            Rejection::UnsupportedOperation,
            "power.resolved_domain.required",
        ));
    };
    if query.family != ProviderOperationFamily::Elementwise
        || query.inputs.len() != 2
        || query.outputs.len() > 1
    {
        return Err((Rejection::InvalidShape, "power.operand_roles.invalid"));
    }
    if query.workload.real_power_scalar.is_some() {
        return exact_scalar_work(provider, query, domain);
    }
    let mode = query.workload.real_power_mode;
    let typed_single_on_double = {
        #[cfg(feature = "wgpu")]
        {
            matches!(profile, HookProfile::Wgpu) && precision == ProviderPrecision::F64
        }
        #[cfg(not(feature = "wgpu"))]
        {
            false
        }
    };
    let mixed_single_operand = match mode {
        Some(RealPowerArithmeticMode::MixedDoubleThenSingle { single_operand }) => {
            if precision != ProviderPrecision::F64 {
                return Err((
                    Rejection::UnsupportedElementType,
                    "power.mixed.native_double.required",
                ));
            }
            Some(single_operand)
        }
        Some(RealPowerArithmeticMode::NativeSingle)
            if precision != ProviderPrecision::F32 && !typed_single_on_double =>
        {
            return Err((
                Rejection::UnsupportedElementType,
                "power.native_single.required",
            ));
        }
        Some(RealPowerArithmeticMode::NativeDouble) if precision != ProviderPrecision::F64 => {
            return Err((
                Rejection::UnsupportedElementType,
                "power.native_double.required",
            ));
        }
        _ => None,
    };
    let (native_element, native_complex, bytes) = match precision {
        ProviderPrecision::F32 => (
            ProviderElementType::F32,
            ProviderElementType::ComplexF32,
            4u64,
        ),
        ProviderPrecision::F64 => (
            ProviderElementType::F64,
            ProviderElementType::ComplexF64,
            8u64,
        ),
    };
    let (element, complex) = if mode == Some(RealPowerArithmeticMode::NativeSingle) {
        (ProviderElementType::F32, ProviderElementType::ComplexF32)
    } else {
        (native_element, native_complex)
    };
    if matches!(profile, HookProfile::InProcess) && precision != ProviderPrecision::F64 {
        return Err((
            Rejection::UnsupportedElementType,
            "power.native_double.required",
        ));
    }
    for (ordinal, input) in query.inputs.iter().enumerate() {
        let strided_allowed = {
            #[cfg(feature = "wgpu")]
            {
                matches!(profile, HookProfile::Wgpu)
                    && input.layout == ProviderLayout::Strided
                    && input.shape.len() == 2
            }
            #[cfg(not(feature = "wgpu"))]
            {
                false
            }
        };
        if input.layout != ProviderLayout::ColumnMajorContiguous && !strided_allowed {
            return Err((Rejection::UnsupportedLayout, "power.layout.unsupported"));
        }
        let expected_type = if matches!(
            (mixed_single_operand, ordinal),
            (Some(RealPowerOperand::Base), 0) | (Some(RealPowerOperand::Exponent), 1)
        ) {
            ProviderElementType::F32
        } else {
            element
        };
        if input.element_type != expected_type {
            return Err((
                Rejection::UnsupportedElementType,
                "power.precision.mismatch",
            ));
        }
        if input.storage != ProviderStorage::DenseReal {
            return Err((Rejection::UnsupportedStorage, "power.real_storage.required"));
        }
    }
    if mixed_single_operand.is_some() {
        let host_inputs: Vec<_> = query
            .inputs
            .iter()
            .filter(|input| input.residency == ProviderResidency::Host)
            .collect();
        if host_inputs.len() != 1
            || host_inputs[0].checked_element_count() != Some(1)
            || query
                .inputs
                .iter()
                .filter(|input| input.residency == ProviderResidency::Device)
                .count()
                != 1
        {
            return Err((Rejection::InvalidShape, "power.mixed.host_scalar.required"));
        }
    }
    if typed_single_on_double
        && mode == Some(RealPowerArithmeticMode::NativeSingle)
        && !query
            .inputs
            .iter()
            .any(|input| input.residency == ProviderResidency::Device)
    {
        return Err((
            Rejection::InvalidShape,
            "power.native_single.resident.required",
        ));
    }
    let rank = query.inputs[0].shape.len().max(query.inputs[1].shape.len());
    let align = |shape: &[u64]| {
        let mut shape = if rank >= 2 && shape.len() == 1 {
            vec![1, shape[0]]
        } else {
            shape.to_vec()
        };
        shape.resize(rank, 1);
        shape
    };
    let left = align(&query.inputs[0].shape);
    let right = align(&query.inputs[1].shape);
    let mut shape = Vec::with_capacity(rank);
    for (&a, &b) in left.iter().zip(&right) {
        shape.push(if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err((Rejection::InvalidShape, "power.broadcast.invalid"));
        });
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or((Rejection::InvalidShape, "power.shape.overflow"))?;
    let width = if domain == PowerResultDomain::Complex {
        2u64
    } else {
        1
    };
    let words = elements
        .checked_mul(width)
        .ok_or((Rejection::ResourceLimit, "power.output.overflow"))?;
    let public_element = if mixed_single_operand.is_some() {
        ProviderElementType::F32
    } else {
        element
    };
    let public_complex = if mixed_single_operand.is_some() {
        ProviderElementType::ComplexF32
    } else {
        complex
    };
    let output = ProviderRepresentation {
        element_type: if width == 2 {
            public_complex
        } else {
            public_element
        },
        storage: if width == 2 {
            ProviderStorage::DenseComplexInterleaved
        } else {
            ProviderStorage::DenseReal
        },
        shape,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: if mixed_single_operand.is_some()
            || (typed_single_on_double && mode == Some(RealPowerArithmeticMode::NativeSingle))
        {
            ProviderResidency::Device
        } else {
            ProviderResidency::Host
        },
    };
    if let Some(declared) = query.outputs.first() {
        if declared.shape != output.shape
            || declared.element_type != output.element_type
            || declared.storage != output.storage
            || declared.layout != output.layout
            || declared.residency != output.residency
        {
            return Err((Rejection::InvalidShape, "power.output_contract.mismatch"));
        }
    }
    let result = PowerWork {
        physical_output_bytes: None,
        #[cfg(feature = "wgpu")]
        elements,
        temporary_bytes: 0,
        unknown_peak: false,
        #[cfg(feature = "wgpu")]
        readback_bytes: 0,
        dispatches: 0,
        output,
    };
    let _ = (profile, words, bytes);
    if matches!(profile, HookProfile::InProcess) {
        let mut result = result;
        if mixed_single_operand.is_some() {
            let extra = mixed::in_process_physical_work(provider, query, &result.output, words)?;
            result.temporary_bytes = extra.temporary_bytes;
            result.dispatches = extra.dispatches;
            result.unknown_peak = extra.unknown_peak;
        }
        return Ok(result);
    }
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) {
        let mut result = result;
        if rank > crate::backend::wgpu::params::BCAST_MAX_RANK
            || words > u64::from(u32::MAX)
            || result.output.shape.iter().any(|d| *d > u64::from(u32::MAX))
        {
            return Err((Rejection::ResourceLimit, "power.gpu_index_limit"));
        }
        if elements != 0 {
            // Paired flags, two scalar full-reduction stages, then power.
            result.temporary_bytes = elements
                .checked_add(2)
                .and_then(|n| n.checked_mul(bytes))
                .ok_or((Rejection::ResourceLimit, "power.probe.overflow"))?;
            result.readback_bytes = bytes;
            result.dispatches = 4;
        }
        if mixed_single_operand.is_some() {
            let extra = mixed::physical_work(provider, query, &result.output, words)?;
            result.temporary_bytes = result
                .temporary_bytes
                .checked_add(extra.temporary_bytes)
                .ok_or((Rejection::ResourceLimit, "power.mixed.bytes.overflow"))?;
            result.dispatches = result
                .dispatches
                .checked_add(extra.dispatches)
                .ok_or((Rejection::ResourceLimit, "power.mixed.dispatch.overflow"))?;
            result.unknown_peak = extra.unknown_peak;
        }
        return Ok(result);
    }
    Ok(result)
}

fn exact_scalar_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
    domain: PowerResultDomain,
) -> Result<PowerWork, (ProviderRejectionCode, &'static str)> {
    use ProviderRejectionCode as Reject;
    let reject = (Reject::UnsupportedOperation, "power.exact_scalar.contract");
    if domain != PowerResultDomain::Real
        || query.workload.real_power_mode
            != Some(RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base,
            })
        || query.inputs[1].element_type != ProviderElementType::F64
        || query.inputs[1].storage != ProviderStorage::DenseReal
        || query.inputs[1].residency != ProviderResidency::Host
        || query.inputs[1].layout != ProviderLayout::ColumnMajorContiguous
        || query.inputs[1].checked_element_count() != Some(1)
    {
        return Err(reject);
    }
    let source = &query.inputs[0];
    let scalar = query.workload.real_power_scalar.as_ref().ok_or(reject)?;
    let logical_source_bytes = source.checked_byte_len().ok_or(reject)?;
    let backing_bytes = if let Some(input) = &scalar.input {
        match input {
            runmat_accelerate_api::PowerInputDescription::Live { physical, view } => {
                if view.validate(physical).is_err()
                    || physical.element_type() != runmat_accelerate_api::NumericElementType::F32
                    || physical.storage() != runmat_accelerate_api::GpuTensorStorage::Real
                    || view.is_logical()
                    || view.shape().iter().map(|&n| n as u64).collect::<Vec<_>>() != source.shape
                    || !matches!(
                        (source.layout, view.layout()),
                        (
                            ProviderLayout::ColumnMajorContiguous,
                            runmat_accelerate_api::GpuTensorLayout::Contiguous
                        ) | (
                            ProviderLayout::Strided,
                            runmat_accelerate_api::GpuTensorLayout::Transpose(_)
                        )
                    )
                {
                    return Err(reject);
                }
                Some(physical.requested_bytes())
            }
            runmat_accelerate_api::PowerInputDescription::PlannedDense { requested_bytes } => {
                if source.layout != ProviderLayout::ColumnMajorContiguous
                    || *requested_bytes < logical_source_bytes
                    || provider
                        .plan_dense_real_output_buffer_bytes(source)
                        .map_err(|_| {
                            (
                                Reject::ProviderUnavailable,
                                "power.exact_scalar.input_projection_failed",
                            )
                        })?
                        != Some(*requested_bytes)
                {
                    return Err(reject);
                }
                Some(*requested_bytes)
            }
        }
    } else {
        None
    };
    let request = runmat_accelerate_api::power_service::exact_single_base_request(
        source,
        scalar.control,
        RealPowerOperand::Base,
        backing_bytes.unwrap_or(logical_source_bytes),
    )
    .ok_or(reject)?;
    let physical = match provider.propose_real_elementwise(&request) {
        Ok(Some(physical)) => physical,
        Ok(None) => return Err(reject),
        Err(error) if error.is_unsupported() => return Err(reject),
        Err(_) => {
            return Err((
                Reject::ProviderUnavailable,
                "power.exact_scalar.proposal_failed",
            ))
        }
    };
    let source_shape = source
        .shape
        .iter()
        .map(|&n| usize::try_from(n).ok())
        .collect::<Option<Vec<_>>>()
        .ok_or(reject)?;
    let scalar_shape = query.inputs[1]
        .shape
        .iter()
        .map(|&n| usize::try_from(n).ok())
        .collect::<Option<Vec<_>>>()
        .ok_or(reject)?;
    let shape = runmat_types::geometry::broadcast_concrete_shape(&source_shape, &scalar_shape)
        .map_err(|_| reject)?;
    let mut output = source.clone();
    output.layout = ProviderLayout::ColumnMajorContiguous;
    output.shape = shape.into_iter().map(|n| n as u64).collect();
    if query
        .outputs
        .first()
        .is_some_and(|declared| declared != &output)
    {
        return Err(reject);
    }
    let logical = output.checked_byte_len().ok_or(reject)?;
    let temporary_bytes = physical
        .output_buffer_bytes
        .checked_sub(logical)
        .and_then(|pad| pad.checked_add(physical.uniform_buffer_bytes))
        .and_then(|bytes| {
            bytes.checked_add(
                backing_bytes
                    .unwrap_or(logical_source_bytes)
                    .checked_sub(logical_source_bytes)?,
            )
        })
        .ok_or(reject)?;
    Ok(PowerWork {
        physical_output_bytes: Some(physical.output_buffer_bytes),
        #[cfg(feature = "wgpu")]
        elements: source.checked_element_count().ok_or(reject)?,
        temporary_bytes,
        // Shape geometry does not prove the size of a retained allocation,
        // including a nonempty subview of a larger backing buffer.
        unknown_peak: backing_bytes.is_none(),
        #[cfg(feature = "wgpu")]
        readback_bytes: 0,
        dispatches: physical.queue_submissions,
        output,
    })
}

#[cfg(all(test, feature = "wgpu"))]
#[path = "power/mixed_tests.rs"]
mod mixed_tests;

#[cfg(all(test, feature = "wgpu"))]
#[path = "power/native_single_tests.rs"]
mod native_single_tests;

#[cfg(test)]
#[path = "power/inprocess_tests.rs"]
mod inprocess_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_domain_is_not_real_and_complex_output_bytes_are_explicit() {
        let provider = crate::simple_provider::InProcessProvider::new();
        let mut snapshot = provider.capability_snapshot();
        append(&mut snapshot);
        let input = ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 3],
            residency: ProviderResidency::Host,
        };
        let mut query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(
                runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
            ),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![input.clone(), input],
            outputs: vec![],
            workload: Default::default(),
        };
        assert!(!feasibility(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &provider,
            &snapshot,
            &query
        )
        .unwrap()
        .is_supported());
        query.workload.numerical_domain =
            NumericalDomainEvidence::RealPower(PowerResultDomain::Complex);
        let ProviderFeasibility::Supported { estimate } = feasibility(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &provider,
            &snapshot,
            &query,
        )
        .unwrap() else {
            panic!("resolved support")
        };
        assert_eq!(estimate.output_bytes, Some(48));
        assert_eq!(estimate.transient_bytes, Some(96));
        #[cfg(feature = "wgpu")]
        {
            let work = wgpu_work(ProviderPrecision::F64, &provider, &query).unwrap();
            assert_eq!(work.temporary_bytes, 40);
            assert_eq!(work.readback_bytes, 8);
            assert_eq!(work.dispatches, 4);
            query.inputs[0].layout = ProviderLayout::Strided;
            let strided = wgpu_work(ProviderPrecision::F64, &provider, &query).unwrap();
            assert_eq!(strided.temporary_bytes, work.temporary_bytes);
            assert_eq!(strided.readback_bytes, work.readback_bytes);
            assert_eq!(strided.dispatches, work.dispatches);
            assert!(super::work(
                HookProfile::InProcess,
                ProviderPrecision::F64,
                &provider,
                &query
            )
            .is_err());
            query.inputs[0].shape = vec![1, 3, 1];
            assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
            query.inputs[0].shape = vec![1, 3];
            query.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
            query.inputs[0].element_type = ProviderElementType::F32;
            query.inputs[1].element_type = ProviderElementType::F32;
            // A typed classification probe is not full standalone execution.
            assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
            query.inputs[0].element_type = ProviderElementType::F64;
            query.inputs[1].element_type = ProviderElementType::F64;
            for input in &mut query.inputs {
                input.shape = vec![0, 3];
            }
            let empty = wgpu_work(ProviderPrecision::F64, &provider, &query).unwrap();
            assert_eq!(empty.temporary_bytes, 0);
            assert_eq!(empty.readback_bytes, 0);
            assert_eq!(empty.dispatches, 0);
            query.inputs[0].shape = vec![u64::MAX, 2];
            query.inputs[1].shape = vec![1, 1];
            assert!(wgpu_work(ProviderPrecision::F64, &provider, &query).is_none());
        }
    }
}
