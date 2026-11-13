//! Typed in-process feasibility for the request-bound GPU child stream.

use super::*;
use runmat_accelerate_api::random::{
    GpuIntegerRequest, GpuNormalTransform, GpuRandomAlgorithm,
    PROVIDER_GPU_INTEGER_BATCH_OPERATION, PROVIDER_GPU_INTEGER_OPERATION,
    PROVIDER_GPU_RANDOM_OPERATION,
};

pub(super) fn validate(query: &ProviderFeasibilityQuery) -> Option<ProviderFeasibility> {
    if query.operation.0 == PROVIDER_GPU_INTEGER_BATCH_OPERATION {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "random.gpu_child.integer.batch.no_in_process_state_owner",
        ));
    }
    if query.operation.0 == PROVIDER_GPU_INTEGER_OPERATION {
        return validate_integer(query);
    }
    if query.operation.0 != PROVIDER_GPU_RANDOM_OPERATION {
        return None;
    }
    let reject = ProviderFeasibility::rejected;
    let Some(workload) = query.workload.gpu_random else {
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "random.gpu_child.missing_workload",
        ));
    };
    if query.family != ProviderOperationFamily::Library
        || !query.inputs.is_empty()
        || query.outputs.len() != 1
    {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "random.gpu_child.invalid_query",
        ));
    }
    let output = &query.outputs[0];
    let components_per_element = match output.storage {
        ProviderStorage::DenseReal => 1_u64,
        ProviderStorage::DenseComplexInterleaved => 2_u64,
        ProviderStorage::Sparse => {
            return Some(reject(
                ProviderRejectionCode::UnsupportedStorage,
                "random.gpu_child.storage",
            ));
        }
    };
    let bytes_per_component = match output.element_type {
        ProviderElementType::F32 => 4_u64,
        ProviderElementType::F64 => 8_u64,
        _ => {
            return Some(reject(
                ProviderRejectionCode::UnsupportedElementType,
                "random.gpu_child.class",
            ));
        }
    };
    if output.layout != ProviderLayout::ColumnMajorContiguous
        || output.residency != ProviderResidency::Device
        || workload.reservation.algorithm != GpuRandomAlgorithm::Threefry4x64_20
        || workload.reservation.normal_transform != GpuNormalTransform::BoxMuller
    {
        return Some(reject(
            ProviderRejectionCode::UnsupportedOperation,
            "random.gpu_child.contract",
        ));
    }
    let count = if output.shape.contains(&0) {
        Some(0_u64)
    } else {
        output
            .shape
            .iter()
            .try_fold(1_u64, |n, dim| n.checked_mul(*dim))
    }
    .and_then(|n| n.checked_mul(components_per_element));
    let Some(count) = count else {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "random.gpu_child.geometry",
        ));
    };
    if count != workload.reservation.count
        || workload
            .reservation
            .first
            .checked_add(u128::from(count))
            .is_none()
    {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "random.gpu_child.range",
        ));
    }
    let Some(output_bytes) = count.checked_mul(bytes_per_component) else {
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "random.gpu_child.bytes",
        ));
    };
    Some(ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: Some(output_bytes),
        output_bytes: Some(output_bytes),
        dispatches: Some(u32::from(count != 0)),
    }))
}

fn validate_integer(query: &ProviderFeasibilityQuery) -> Option<ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    let Some(workload) = query.workload.gpu_integer else {
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "random.gpu_child.integer.missing_workload",
        ));
    };
    if query.family != ProviderOperationFamily::Library
        || !query.inputs.is_empty()
        || query.outputs.len() != 1
    {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "random.gpu_child.integer.query",
        ));
    }
    let output = &query.outputs[0];
    let class = match output.element_type {
        ProviderElementType::F32 => runmat_accelerate_api::NumericElementType::F32,
        ProviderElementType::F64 => runmat_accelerate_api::NumericElementType::F64,
        ProviderElementType::I32 => runmat_accelerate_api::NumericElementType::I32,
        ProviderElementType::U32 => runmat_accelerate_api::NumericElementType::U32,
        _ => {
            return Some(reject(
                ProviderRejectionCode::UnsupportedElementType,
                "random.gpu_child.integer.class",
            ))
        }
    };
    if output.storage != ProviderStorage::DenseReal
        || output.layout != ProviderLayout::ColumnMajorContiguous
        || output.residency != ProviderResidency::Device
    {
        return Some(reject(
            ProviderRejectionCode::UnsupportedStorage,
            "random.gpu_child.integer.output",
        ));
    }
    let Some(count) = output
        .checked_element_count()
        .and_then(|count| usize::try_from(count).ok())
    else {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "random.gpu_child.integer.shape",
        ));
    };
    let shape = [count];
    let request = GpuIntegerRequest {
        shape: &shape,
        class,
        domain: workload.domain,
        reservation: workload.reservation,
    };
    if request.fixed_work_32().is_none() {
        return Some(reject(
            ProviderRejectionCode::UnsupportedOperation,
            "random.gpu_child.integer.fixed_work",
        ));
    }
    let Some(output_bytes) = output.checked_byte_len() else {
        return Some(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "random.gpu_child.integer.bytes",
        ));
    };
    Some(ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: Some(output_bytes),
        output_bytes: Some(output_bytes),
        dispatches: Some(u32::from(count != 0)),
    }))
}
