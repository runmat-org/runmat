use super::fusion::fusion_operation_token;
use crate::fusion::FusionKind;
use runmat_accelerate_api::{
    AccelProvider, ProviderCapabilityOperation, ProviderCapabilitySnapshot,
    ProviderConcurrencyCapabilities, ProviderElementType, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderPrecision, ProviderRejectionCode, ProviderRepresentation, ProviderResidency,
    ProviderResourceEstimate, ProviderStorage, PROVIDER_CAPABILITY_SCHEMA_VERSION,
};
use runmat_value::{NumericDType, Value};

mod arithmetic;
mod bounded_transfer;
mod complex_components;
pub(crate) mod convolution;
mod exponentials;
pub(crate) mod extrema;
pub(crate) mod floating_conversion;
#[cfg(feature = "wgpu")]
pub(crate) mod fspecial;
pub(crate) mod fusion_elementwise;
pub(crate) mod fusion_reduction;
mod iir;
mod image_stages;
pub(crate) mod imfilter;
mod integer_conversion;
pub(crate) mod logarithm_roots;
mod logical;
pub(crate) mod matmul;
pub(crate) mod power;
mod random_child;
pub(crate) mod real_unary;
pub(crate) mod remainder;
pub(crate) mod resident_select;
pub(crate) mod rounding;
pub(crate) mod sum;
pub(crate) mod trigonometry;

const IN_PROCESS_OPERATIONS: &[(&str, ProviderOperationFamily)] = &[
    (
        runmat_accelerate_api::random::PROVIDER_GPU_RANDOM_OPERATION,
        ProviderOperationFamily::Library,
    ),
    (
        runmat_accelerate_api::PROVIDER_IIR_CAUSAL_PASS_OPERATION,
        ProviderOperationFamily::Library,
    ),
    (
        runmat_accelerate_api::PROVIDER_IIR_ZERO_PHASE_OPERATION,
        ProviderOperationFamily::Library,
    ),
    (
        runmat_accelerate_api::PROVIDER_IIR_PREPARED_PASS_OPERATION,
        ProviderOperationFamily::Library,
    ),
    ("transfer.upload", ProviderOperationFamily::Upload),
    ("transfer.download", ProviderOperationFamily::Download),
    ("legacy.elementwise", ProviderOperationFamily::Elementwise),
    ("legacy.reduction", ProviderOperationFamily::Reduction),
    ("legacy.matmul", ProviderOperationFamily::MatrixMultiply),
    (
        fusion_operation_token(&FusionKind::ElementwiseChain),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::MatmulEpilogue),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::PowerStepNormalize),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::ExplainedVariance),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::ImageNormalize),
        ProviderOperationFamily::Fusion,
    ),
];

#[cfg(feature = "wgpu")]
const WGPU_OPERATIONS: &[(&str, ProviderOperationFamily)] = &[
    (
        runmat_accelerate_api::random::PROVIDER_GPU_RANDOM_OPERATION,
        ProviderOperationFamily::Library,
    ),
    (
        runmat_accelerate_api::PROVIDER_IIR_CAUSAL_PASS_OPERATION,
        ProviderOperationFamily::Library,
    ),
    (
        runmat_accelerate_api::PROVIDER_IIR_ZERO_PHASE_OPERATION,
        ProviderOperationFamily::Library,
    ),
    (
        runmat_accelerate_api::PROVIDER_IIR_PREPARED_PASS_OPERATION,
        ProviderOperationFamily::Library,
    ),
    ("transfer.upload", ProviderOperationFamily::Upload),
    ("transfer.download", ProviderOperationFamily::Download),
    ("legacy.elementwise", ProviderOperationFamily::Elementwise),
    ("legacy.reduction", ProviderOperationFamily::Reduction),
    ("legacy.matmul", ProviderOperationFamily::MatrixMultiply),
    (
        fusion_operation_token(&FusionKind::ElementwiseChain),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::Reduction),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::MatmulEpilogue),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::PowerStepNormalize),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::ExplainedVariance),
        ProviderOperationFamily::Fusion,
    ),
    (
        fusion_operation_token(&FusionKind::ImageNormalize),
        ProviderOperationFamily::Fusion,
    ),
];

pub(crate) fn tensor_representation(
    value: &Value,
    provider_precision: ProviderPrecision,
) -> Option<ProviderRepresentation> {
    let (element_type, storage, shape, residency) = match value {
        Value::Tensor(tensor) => (
            provider_element_type(tensor.numeric_dtype()),
            ProviderStorage::DenseReal,
            tensor.shape().to_vec(),
            ProviderResidency::Host,
        ),
        Value::GpuTensor(handle) => {
            let element_type = if runmat_accelerate_api::handle_is_logical(handle) {
                ProviderElementType::Logical
            } else if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
                ProviderElementType::from(integer_type)
            } else {
                let precision =
                    runmat_accelerate_api::handle_precision(handle).unwrap_or(provider_precision);
                match runmat_accelerate_api::handle_storage(handle) {
                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => {
                        match precision {
                            ProviderPrecision::F32 => ProviderElementType::ComplexF32,
                            ProviderPrecision::F64 => ProviderElementType::ComplexF64,
                        }
                    }
                    _ => ProviderElementType::from(precision),
                }
            };
            let storage = match runmat_accelerate_api::handle_storage(handle) {
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => {
                    ProviderStorage::DenseComplexInterleaved
                }
                _ => ProviderStorage::DenseReal,
            };
            (
                element_type,
                storage,
                handle.shape().to_vec(),
                ProviderResidency::Device,
            )
        }
        Value::Num(_) => (
            ProviderElementType::F64,
            ProviderStorage::DenseReal,
            vec![1, 1],
            ProviderResidency::Host,
        ),
        Value::Int(integer) => (
            provider_element_type(integer.numeric_dtype()),
            ProviderStorage::DenseReal,
            vec![1, 1],
            ProviderResidency::Host,
        ),
        _ => return None,
    };
    Some(ProviderRepresentation {
        element_type,
        storage,
        layout: if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_transposed(handle))
        {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        shape: shape
            .into_iter()
            .map(|extent| u64::try_from(extent).unwrap_or(u64::MAX))
            .collect(),
        residency,
    })
}

pub(crate) fn dense_output_representation(
    provider_precision: ProviderPrecision,
    shape: &[usize],
) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: ProviderElementType::from(provider_precision),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape
            .iter()
            .map(|&extent| u64::try_from(extent).unwrap_or(u64::MAX))
            .collect(),
        residency: ProviderResidency::Device,
    }
}

pub(crate) fn provider_rejection_token(code: ProviderRejectionCode) -> &'static str {
    match code {
        ProviderRejectionCode::UnsupportedOperation => "provider_operation_unsupported",
        ProviderRejectionCode::UnsupportedElementType => "provider_element_type_unsupported",
        ProviderRejectionCode::UnsupportedStorage => "provider_storage_unsupported",
        ProviderRejectionCode::UnsupportedLayout => "provider_layout_unsupported",
        ProviderRejectionCode::UnsupportedRank => "provider_rank_unsupported",
        ProviderRejectionCode::InvalidShape => "provider_shape_invalid",
        ProviderRejectionCode::ResourceLimit => "provider_resource_limit",
        ProviderRejectionCode::ProviderUnavailable => "provider_unavailable",
        ProviderRejectionCode::UnknownResourceEstimate => "provider_resource_estimate_unknown",
    }
}

fn provider_element_type(dtype: NumericDType) -> ProviderElementType {
    match dtype {
        NumericDType::F64 => ProviderElementType::F64,
        NumericDType::F32 => ProviderElementType::F32,
        NumericDType::I8 => ProviderElementType::I8,
        NumericDType::I16 => ProviderElementType::I16,
        NumericDType::I32 => ProviderElementType::I32,
        NumericDType::I64 => ProviderElementType::I64,
        NumericDType::U8 => ProviderElementType::U8,
        NumericDType::U16 => ProviderElementType::U16,
        NumericDType::U32 => ProviderElementType::U32,
        NumericDType::U64 => ProviderElementType::U64,
    }
}

pub(crate) fn in_process_capability_snapshot(
    provider: &(impl AccelProvider + ?Sized),
) -> ProviderCapabilitySnapshot {
    // Native mixed-precision arithmetic and synchronous hook observations.
    let mut snapshot = dense_capability_snapshot(provider, 17, IN_PROCESS_OPERATIONS);
    logical::append(&mut snapshot);
    resident_select::append(&mut snapshot);
    sum::append(&mut snapshot);
    extrema::append(&mut snapshot, false);
    convolution::append(&mut snapshot, false);
    imfilter::append(&mut snapshot, false);
    integer_conversion::append(&mut snapshot);
    arithmetic::append(&mut snapshot);
    complex_components::append(&mut snapshot);
    exponentials::append(&mut snapshot);
    floating_conversion::append(&mut snapshot);
    trigonometry::append(&mut snapshot);
    power::append(&mut snapshot);
    logarithm_roots::append(&mut snapshot);
    rounding::append(&mut snapshot);
    remainder::append(&mut snapshot);
    snapshot
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_capability_snapshot(
    provider: &(impl AccelProvider + ?Sized),
    max_buffer_bytes: u64,
) -> ProviderCapabilitySnapshot {
    // Revision 24 includes the exact native real exponential hook inventory.
    let mut snapshot = dense_capability_snapshot(provider, 24, WGPU_OPERATIONS);
    // This operation has its own typed F32/F64 feasibility. Do not widen the
    // general element-type inventory: unrelated hooks retain their preferred
    // precision contracts.
    // A selected F64 reduction emits IEEE word helpers. Keep the provider's
    // advertised operation and revision aligned with its enabled device
    // features; ordinary F64 arithmetic and integer reductions stay present.
    if provider.precision() == ProviderPrecision::F64
        && !provider.supports_shader_requirements(
            runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords,
        )
    {
        snapshot.operations.retain(|operation| {
            [
                FusionKind::Reduction,
                FusionKind::MatmulEpilogue,
                FusionKind::PowerStepNormalize,
            ]
            .iter()
            .all(|kind| {
                operation.identity != ProviderOperationIdentity::new(fusion_operation_token(kind))
            })
        });
        snapshot.revision = 25;
    }
    // WebGPU's adapter limit is a maximum for one buffer. It does not
    // establish a total device-memory capacity.
    snapshot.max_allocation_bytes = Some(max_buffer_bytes);
    logical::append(&mut snapshot);
    resident_select::append(&mut snapshot);
    sum::append(&mut snapshot);
    if provider.precision() == ProviderPrecision::F64
        && !provider.supports_shader_requirements(
            runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords,
        )
    {
        snapshot.operations.retain(|operation| {
            [
                runmat_types::SumInvocation::Sum,
                runmat_types::SumInvocation::NanSum,
            ]
            .into_iter()
            .all(|invocation| {
                operation.identity
                    != ProviderOperationIdentity::new(
                        runmat_builtins::sum_invocation_entry(invocation)
                            .identity
                            .name,
                    )
            })
        });
    }
    extrema::append(&mut snapshot, true);
    convolution::append(&mut snapshot, true);
    fspecial::append(&mut snapshot, true);
    imfilter::append(&mut snapshot, true);
    integer_conversion::append(&mut snapshot);
    arithmetic::append(&mut snapshot);
    complex_components::append(&mut snapshot);
    exponentials::append(&mut snapshot);
    floating_conversion::append(&mut snapshot);
    trigonometry::append(&mut snapshot);
    power::append(&mut snapshot);
    logarithm_roots::append(&mut snapshot);
    rounding::append(&mut snapshot);
    remainder::append(&mut snapshot);
    snapshot
}

pub(crate) fn in_process_feasibility(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    if let Some(result) = random_child::validate(query) {
        return result;
    }
    if let Some(result) = image_stages::validate(query) {
        return result;
    }
    if let Some(result) = iir::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = sum::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = extrema::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = convolution::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = imfilter::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = resident_select::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = matmul::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = fusion_elementwise::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = fusion_reduction::validate(false, query) {
        return result;
    }
    if let Some(result) = rounding::validate(provider, rounding::HookProfile::InProcess, query) {
        return result;
    }
    if let Some(result) = remainder::validate(provider, remainder::HookProfile::InProcess, query) {
        return result;
    }
    if let Some(result) = integer_conversion::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = trigonometry::validate(provider, false, query) {
        return result;
    }
    if let Some(result) = floating_conversion::validate(provider, false, query) {
        return result;
    }
    if let Some(result) =
        logical::validate(logical::HookProfile::InProcess, provider.precision(), query)
    {
        return result;
    }
    let snapshot = exact_float_upload_snapshot(in_process_capability_snapshot(provider), query);
    if let Some(result) = exponentials::validate(provider, false, &snapshot, query) {
        return result;
    }
    if let Some(result) = logarithm_roots::feasibility(
        real_unary::HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        query,
    ) {
        return result;
    }
    if let Some(result) = power::feasibility(
        power::HookProfile::InProcess,
        provider.precision(),
        provider,
        &snapshot,
        query,
    ) {
        return result;
    }
    arithmetic::feasibility(
        arithmetic::HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        query,
    )
    .or_else(|| {
        complex_components::feasibility(
            complex_components::HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            query,
        )
    })
    .unwrap_or_else(|| dense_feasibility(&snapshot, query))
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_feasibility(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
    max_buffer_bytes: u64,
) -> ProviderFeasibility {
    if let Some(result) = image_stages::validate(query) {
        return result;
    }
    if let Some(result) = iir::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = sum::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = extrema::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = convolution::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = fspecial::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = imfilter::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = resident_select::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = matmul::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = fusion_elementwise::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = fusion_reduction::validate(true, query) {
        return result;
    }
    if let Some(result) = rounding::validate(provider, rounding::HookProfile::Wgpu, query) {
        return result;
    }
    if let Some(result) = remainder::validate(provider, remainder::HookProfile::Wgpu, query) {
        return result;
    }
    if let Some(result) = integer_conversion::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = trigonometry::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = floating_conversion::validate(provider, true, query) {
        return result;
    }
    if let Some(result) = logical::validate(logical::HookProfile::Wgpu, provider.precision(), query)
    {
        return result;
    }
    // The provider owns one immutable inventory. Ordinary query paths borrow
    // it; only the exact upload exception needs a detached modified value.
    let fallback_inventory = provider
        .capability_snapshot_ref()
        .is_none()
        .then(|| wgpu_capability_snapshot(provider, max_buffer_bytes));
    let inventory = provider
        .capability_snapshot_ref()
        .or(fallback_inventory.as_ref())
        .expect("one capability inventory route is selected");
    let upload = (query.operation.0 == "transfer.upload"
        && query.family == ProviderOperationFamily::Upload)
        .then(|| exact_float_upload_snapshot(inventory.clone(), query));
    let snapshot = upload.as_ref().unwrap_or(inventory);
    if let Some(result) = exponentials::validate(provider, true, snapshot, query) {
        return result;
    }
    if let Some(result) = logarithm_roots::feasibility(
        real_unary::HookProfile::Wgpu,
        provider.precision(),
        snapshot,
        query,
    ) {
        return result;
    }
    if let Some(result) = power::feasibility(
        power::HookProfile::Wgpu,
        provider.precision(),
        provider,
        snapshot,
        query,
    ) {
        return result;
    }
    arithmetic::feasibility(
        arithmetic::HookProfile::Wgpu,
        provider.precision(),
        snapshot,
        query,
    )
    .or_else(|| {
        complex_components::feasibility(
            complex_components::HookProfile::Wgpu,
            provider.precision(),
            snapshot,
            query,
        )
    })
    .unwrap_or_else(|| {
        bounded_transfer::wgpu_dense_feasibility(snapshot, query, provider.precision())
    })
}

pub(crate) fn dense_capability_snapshot(
    provider: &(impl AccelProvider + ?Sized),
    revision: u64,
    operations: &[(&'static str, ProviderOperationFamily)],
) -> ProviderCapabilitySnapshot {
    let device = provider.device_info_struct();
    let float_type = ProviderElementType::from(provider.precision());
    ProviderCapabilitySnapshot {
        schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
        revision,
        device: device.clone(),
        operations: operations
            .iter()
            .map(|(identity, family)| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(*identity),
                family: *family,
            })
            .collect(),
        element_types: vec![
            ProviderElementType::Logical,
            ProviderElementType::I8,
            ProviderElementType::I16,
            ProviderElementType::I32,
            ProviderElementType::I64,
            ProviderElementType::U8,
            ProviderElementType::U16,
            ProviderElementType::U32,
            ProviderElementType::U64,
            float_type,
            match float_type {
                ProviderElementType::F32 => ProviderElementType::ComplexF32,
                _ => ProviderElementType::ComplexF64,
            },
        ],
        max_rank: None,
        max_allocation_bytes: device.memory_bytes,
        sparse: None,
        concurrency: ProviderConcurrencyCapabilities {
            spawn_handles: provider.spawn_handle_concurrency(),
            concurrent_dispatch: true,
            cancellation: false,
            transactional_results: true,
        },
    }
}

/// Numeric upload stores the host payload's float class verbatim, even when
/// the provider's compute precision differs. Advertise that extra class only
/// for an exact or bounded floating upload query: widening the general snapshot would
/// incorrectly admit F32 operands to F64-only compute hooks (or vice versa).
fn exact_float_upload_snapshot(
    mut snapshot: ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> ProviderCapabilitySnapshot {
    let exact = !query.inputs.is_empty()
        && query.inputs.len() == query.outputs.len()
        && query
            .inputs
            .iter()
            .zip(&query.outputs)
            .all(|(input, output)| {
                input.element_type == output.element_type
                    && floating_upload_storage(input.element_type) == Some(input.storage)
                    && input.storage == output.storage
                    && input.layout == ProviderLayout::ColumnMajorContiguous
                    && output.layout == ProviderLayout::ColumnMajorContiguous
                    && input.shape == output.shape
                    && input.residency == ProviderResidency::Host
                    && output.residency == ProviderResidency::Device
            });
    let bounded = query.inputs.is_empty()
        && query.outputs.is_empty()
        && query
            .workload
            .transfer_output_bounds
            .as_ref()
            .is_some_and(|bounds| {
                !bounds.is_empty()
                    && bounds.iter().all(|bound| {
                        floating_upload_storage(bound.element_type) == Some(bound.storage)
                            && bound.layout == ProviderLayout::ColumnMajorContiguous
                            && bound.residency == ProviderResidency::Device
                    })
            });
    if query.operation.0 == "transfer.upload"
        && query.family == ProviderOperationFamily::Upload
        && (exact || bounded)
    {
        for class in [
            ProviderElementType::F32,
            ProviderElementType::F64,
            ProviderElementType::ComplexF32,
            ProviderElementType::ComplexF64,
        ] {
            if !snapshot.element_types.contains(&class) {
                snapshot.element_types.push(class);
            }
        }
    }
    snapshot
}

fn floating_upload_storage(element: ProviderElementType) -> Option<ProviderStorage> {
    match element {
        ProviderElementType::F32 | ProviderElementType::F64 => Some(ProviderStorage::DenseReal),
        ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64 => {
            Some(ProviderStorage::DenseComplexInterleaved)
        }
        _ => None,
    }
}

pub(crate) fn dense_feasibility(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    dense_feasibility_with_bytes(snapshot, query, ProviderRepresentation::checked_byte_len)
}

fn dense_feasibility_with_bytes(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
    allocation_bytes: impl Fn(&ProviderRepresentation) -> Option<u64>,
) -> ProviderFeasibility {
    dense_feasibility_with_extra_elements(snapshot, query, allocation_bytes, &[])
}

/// A query-local arithmetic mode may admit exact additional floating classes
/// without cloning or widening the provider's immutable capability inventory.
fn dense_feasibility_with_extra_elements(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
    allocation_bytes: impl Fn(&ProviderRepresentation) -> Option<u64>,
    additional_elements: &[ProviderElementType],
) -> ProviderFeasibility {
    if !snapshot.supports_operation(&query.operation, query.family) {
        return ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "provider.operation.unsupported",
        );
    }

    if let Some(feasibility) = bounded_transfer::feasibility(snapshot, query) {
        return feasibility;
    }
    if let Some(feasibility) = bounded_transfer::exact_contract(query) {
        return feasibility;
    }

    let mut total_bytes = 0_u64;
    let mut largest_allocation = 0_u64;
    for representation in query.inputs.iter().chain(&query.outputs) {
        if !snapshot
            .element_types
            .contains(&representation.element_type)
            && !additional_elements.contains(&representation.element_type)
        {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "provider.element_type.unsupported",
            );
        }
        if matches!(representation.storage, ProviderStorage::Sparse) {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedStorage,
                "provider.storage.sparse_unsupported",
            );
        }
        if representation.layout != ProviderLayout::ColumnMajorContiguous {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedLayout,
                "provider.layout.unsupported",
            );
        }
        if snapshot
            .max_rank
            .is_some_and(|max_rank| representation.shape.len() > max_rank as usize)
        {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedRank,
                "provider.rank.unsupported",
            );
        }
        let Some(bytes) = allocation_bytes(representation) else {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::InvalidShape,
                "provider.shape.overflow",
            );
        };
        let Some(next_total) = total_bytes.checked_add(bytes) else {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.bytes.overflow",
            );
        };
        total_bytes = next_total;
        largest_allocation = largest_allocation.max(bytes);
    }
    if snapshot
        .max_allocation_bytes
        .is_some_and(|maximum| largest_allocation > maximum)
    {
        return ProviderFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "provider.memory.limit",
        );
    }
    let output_bytes = query.outputs.iter().try_fold(0_u64, |total, output| {
        total.checked_add(allocation_bytes(output)?)
    });
    ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: Some(total_bytes),
        output_bytes,
        dispatches: Some(
            if matches!(
                query.family,
                ProviderOperationFamily::Upload | ProviderOperationFamily::Download
            ) {
                0
            } else {
                1
            },
        ),
    })
}

#[cfg(all(test, feature = "wgpu"))]
#[path = "provider_contract/ieee_capability_tests.rs"]
mod ieee_capability_tests;

#[cfg(test)]
mod tests {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn native_single_representation_keeps_full_low_rank_shape() {
        use super::*;
        for (shape, values) in [(Vec::new(), vec![1.0_f32]), (vec![2], vec![1.0_f32, 2.0])] {
            let expected_shape: Vec<u64> = shape
                .iter()
                .map(|&dimension| u64::try_from(dimension).expect("fixture dimension"))
                .collect();
            let value =
                Value::Tensor(runmat_value::Tensor::from_f32(values, shape.clone()).unwrap());
            for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
                let representation = tensor_representation(&value, precision).unwrap();
                assert_eq!(representation.shape, expected_shape);
                assert_eq!(representation.element_type, ProviderElementType::F32);
                assert_eq!(representation.storage, ProviderStorage::DenseReal);
                assert_eq!(representation.residency, ProviderResidency::Host);
            }
        }
    }

    #[test]
    fn authored_double_scalar_representation_does_not_follow_provider_precision() {
        use super::*;
        for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
            let scalar = tensor_representation(&Value::Num(1.0 + f64::EPSILON), precision).unwrap();
            assert_eq!(scalar.element_type, ProviderElementType::F64);
            assert_eq!(scalar.shape, vec![1, 1]);
            let tensor = tensor_representation(
                &Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                precision,
            )
            .unwrap();
            assert_eq!(tensor.element_type, ProviderElementType::F32);
            assert_ne!(scalar.element_type, tensor.element_type);
        }
    }

    #[test]
    fn native_integer_scalar_representation_keeps_its_physical_class() {
        use super::*;
        use runmat_value::IntValue;
        for (integer, element) in [
            (IntValue::I8(-1), ProviderElementType::I8),
            (IntValue::I16(-1), ProviderElementType::I16),
            (IntValue::I32(-1), ProviderElementType::I32),
            (IntValue::I64(i64::MIN), ProviderElementType::I64),
            (IntValue::U8(1), ProviderElementType::U8),
            (IntValue::U16(1), ProviderElementType::U16),
            (IntValue::U32(1), ProviderElementType::U32),
            (IntValue::U64(u64::MAX), ProviderElementType::U64),
        ] {
            for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
                let representation =
                    tensor_representation(&Value::Int(integer.clone()), precision).unwrap();
                assert_eq!(representation.element_type, element);
                assert_eq!(representation.storage, ProviderStorage::DenseReal);
                assert_eq!(representation.shape, vec![1, 1]);
                assert_eq!(representation.residency, ProviderResidency::Host);
            }
        }
    }
    use super::*;

    #[test]
    fn advertised_fusion_operations_use_canonical_kind_identity() {
        let known = [
            FusionKind::ElementwiseChain,
            FusionKind::Reduction,
            FusionKind::MatmulEpilogue,
            FusionKind::ImageNormalize,
            FusionKind::PowerStepNormalize,
            FusionKind::ExplainedVariance,
        ];
        let check = |operations: &[(&str, ProviderOperationFamily)]| {
            assert!(
                !operations.iter().any(|(operation, _)| *operation
                    == fusion_operation_token(&FusionKind::CenteredGram)),
                "structural recognition must not advertise a covariance substitution"
            );
            for (operation, family) in operations {
                if matches!(*family, ProviderOperationFamily::Fusion) {
                    assert!(known
                        .iter()
                        .any(|kind| *operation == fusion_operation_token(kind)));
                }
            }
        };
        check(IN_PROCESS_OPERATIONS);
        #[cfg(feature = "wgpu")]
        check(WGPU_OPERATIONS);
    }
    use runmat_accelerate_api::{
        ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
        ProviderWorkload,
    };

    fn representation(element_type: ProviderElementType) -> ProviderRepresentation {
        ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![4, 4],
            residency: ProviderResidency::Host,
        }
    }

    #[test]
    fn unsupported_operation_is_rejected_before_resources_are_estimated() {
        let snapshot = ProviderCapabilitySnapshot {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 1,
            device: runmat_accelerate_api::ApiDeviceInfo {
                device_id: 1,
                name: "test".into(),
                vendor: "test".to_string(),
                memory_bytes: Some(1024),
                backend: None,
            },
            operations: Vec::new(),
            element_types: vec![ProviderElementType::F64],
            max_rank: None,
            max_allocation_bytes: Some(1024),
            sparse: None,
            concurrency: ProviderConcurrencyCapabilities {
                spawn_handles: runmat_accelerate_api::SpawnHandleConcurrency::Reject,
                concurrent_dispatch: false,
                cancellation: false,
                transactional_results: false,
            },
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("legacy.elementwise"),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![representation(ProviderElementType::F64)],
            outputs: vec![representation(ProviderElementType::F64)],
            workload: ProviderWorkload::default(),
        };

        assert!(matches!(
            dense_feasibility(&snapshot, &query),
            ProviderFeasibility::Rejected { .. }
        ));
    }

    #[test]
    fn exact_single_upload_preserves_float_class_without_widening_compute() {
        use crate::simple_provider::InProcessProvider;
        use runmat_accelerate_api::observation::ProviderObservationCapability;
        use runmat_accelerate_api::ProviderTransferOutputBound;

        let provider = InProcessProvider::new();
        let host = representation(ProviderElementType::F32);
        let mut device = host.clone();
        device.residency = ProviderResidency::Device;
        let upload = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("transfer.upload"),
            family: ProviderOperationFamily::Upload,
            inputs: vec![host.clone()],
            outputs: vec![device.clone()],
            workload: ProviderWorkload {
                requested_outputs: Some(1),
                ..Default::default()
            },
        };
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&upload)
        else {
            panic!("actual InProcess F32 upload must be feasible");
        };
        assert_eq!(estimate.output_bytes, Some(4 * 4 * 4));
        assert!(matches!(
            provider.operation_observation_capability(&upload),
            ProviderObservationCapability::Supported(_)
        ));

        let mut invalid = upload.clone();
        invalid.outputs[0].shape = vec![2, 8];
        assert!(!provider.query_feasibility(&invalid).is_supported());

        let compute = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("legacy.elementwise"),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![device.clone()],
            outputs: vec![device],
            workload: ProviderWorkload::default(),
        };
        assert!(!provider.query_feasibility(&compute).is_supported());

        let bounded = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("transfer.upload"),
            family: ProviderOperationFamily::Upload,
            inputs: Vec::new(),
            outputs: Vec::new(),
            workload: ProviderWorkload {
                requested_outputs: Some(1),
                transfer_output_bounds: Some(vec![ProviderTransferOutputBound {
                    element_type: ProviderElementType::F32,
                    storage: ProviderStorage::DenseReal,
                    layout: ProviderLayout::ColumnMajorContiguous,
                    residency: ProviderResidency::Device,
                    max_elements: 16,
                }]),
                ..Default::default()
            },
        };
        assert!(provider.query_feasibility(&bounded).is_supported());

        let mut native = upload.clone();
        native.inputs[0].element_type = ProviderElementType::F64;
        native.outputs[0].element_type = ProviderElementType::F64;
        assert!(provider.query_feasibility(&native).is_supported());
        for element_type in [
            ProviderElementType::ComplexF32,
            ProviderElementType::ComplexF64,
        ] {
            let mut complex = native.clone();
            for representation in complex.inputs.iter_mut().chain(&mut complex.outputs) {
                representation.element_type = element_type;
                representation.storage = ProviderStorage::DenseComplexInterleaved;
            }
            let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&complex)
            else {
                panic!("native complex upload must preserve its class: {element_type:?}");
            };
            assert_eq!(estimate.output_bytes, complex.outputs[0].checked_byte_len());
            assert!(provider.upload_physical_evidence(&complex).is_some());
            complex.outputs[0].storage = ProviderStorage::DenseReal;
            assert!(!provider.query_feasibility(&complex).is_supported());
            assert!(provider.upload_physical_evidence(&complex).is_none());
        }
        let mut malformed = Vec::new();
        let mut shape = native.clone();
        shape.outputs[0].shape = vec![2, 8];
        malformed.push(shape);
        let mut count = native.clone();
        count.outputs.push(count.outputs[0].clone());
        malformed.push(count);
        let mut requested = native.clone();
        requested.workload.requested_outputs = Some(2);
        malformed.push(requested);
        let mut source_residency = native.clone();
        source_residency.inputs[0].residency = ProviderResidency::Device;
        malformed.push(source_residency);
        let mut result_residency = native.clone();
        result_residency.outputs[0].residency = ProviderResidency::Host;
        malformed.push(result_residency);
        let mut class = native.clone();
        class.outputs[0].element_type = ProviderElementType::F32;
        malformed.push(class);
        for query in malformed {
            assert!(
                !provider.query_feasibility(&query).is_supported(),
                "malformed native-F64 upload was admitted: {query:?}"
            );
            assert!(matches!(
                provider.operation_observation_capability(&query),
                ProviderObservationCapability::Unsupported
            ));
        }
    }
}
