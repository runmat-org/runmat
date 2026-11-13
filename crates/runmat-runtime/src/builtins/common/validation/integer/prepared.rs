//! Prepared, invocation-cooperative binary64 integer boundary for direct values.
//!
//! Compatibility admission belongs to the caller and must precede this API.
//! Aggregate/GC traversal is intentionally not inferred from a direct buffer:
//! it needs a separate lifetime-safe resumable traversal contract.

use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, GpuTensorStorage,
    HostIntegerDataOwned, HostIntegerTensorOwned, HostNumericDataOwned, HostNumericTensorOwned,
    NumericElementType,
};
use runmat_value::{IntegerStorage, Value};

use crate::builtins::common::gpu_helpers::{gpu_class_metadata_matches, ResidencyRestoreGuard};
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

use super::integer_is_exact_f64;

const SCAN_CHUNK: usize = 4096;

/// The execution failure is retained as a typed cause until the identity maps
/// it into its own namespace. Provider failures retain their original source.
#[derive(Debug)]
pub enum IntegerBoundaryFailure {
    Execution(ExecutionServiceError),
    Operational(Box<RuntimeError>),
    AggregateNeedsTraversal,
}

impl IntegerBoundaryFailure {
    pub fn into_runtime_error(self) -> RuntimeError {
        match self {
            Self::Execution(source) => {
                let mut error = source.clone().into_runtime_error();
                error.source = Some(Box::new(source));
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            }
            Self::Operational(error) => *error,
            Self::AggregateNeedsTraversal => build_runtime_error(
                "integer exactness check: aggregate payload requires a lifetime-safe traversal",
            )
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build(),
        }
    }
}

/// Retains a borrowed source and, for a resident wide integer, the single
/// validated native download. A caller may consume the payload instead of
/// gathering the same handle a second time. The source handle remains live
/// while this receipt is live; output-restoration protection remains the
/// caller's separate responsibility.
pub struct PreparedIntegerBoundary<'a> {
    source: &'a Value,
    exact: bool,
    resident_payload: Option<PreparedIntegerPayload>,
}

/// One authenticated readback, in its native physical representation.
pub enum PreparedIntegerPayload {
    Real(HostIntegerTensorOwned),
    Complex(HostNumericTensorOwned),
}

impl<'a> PreparedIntegerBoundary<'a> {
    pub fn source(&self) -> &'a Value {
        self.source
    }

    pub fn is_exact(&self) -> bool {
        self.exact
    }

    pub fn resident_payload(&self) -> Option<&PreparedIntegerPayload> {
        self.resident_payload.as_ref()
    }

    pub fn into_resident_payload(self) -> Option<PreparedIntegerPayload> {
        self.resident_payload
    }
}

impl PreparedIntegerPayload {
    /// Materialize the already-authenticated resident readback without a
    /// second provider selection or download.
    pub fn into_value(self) -> Result<Value, String> {
        match self {
            Self::Real(gathered) => runmat_value::Tensor::new_integer(
                crate::builtins::common::gpu_helpers::integer_storage_from_owned(gathered.data),
                gathered.shape,
            )
            .map(Value::Tensor),
            Self::Complex(gathered) => {
                crate::builtins::common::gpu_helpers::value_from_numeric_download(gathered, false)
            }
        }
    }

    pub fn real(&self) -> Option<&HostIntegerTensorOwned> {
        match self {
            Self::Real(value) => Some(value),
            Self::Complex(_) => None,
        }
    }

    pub fn complex(&self) -> Option<&HostNumericTensorOwned> {
        match self {
            Self::Complex(value) => Some(value),
            Self::Real(_) => None,
        }
    }
}

async fn checkpoint(
    cooperation: &mut Option<&mut CpuCooperationBudget>,
    completed: usize,
) -> Result<(), IntegerBoundaryFailure> {
    if let Some(budget) = cooperation.as_deref_mut() {
        budget
            .checkpoint(completed as u64)
            .await
            .map_err(IntegerBoundaryFailure::Execution)?;
    }
    Ok(())
}

async fn scan_storage(
    storage: &IntegerStorage,
    cooperation: &mut Option<&mut CpuCooperationBudget>,
) -> Result<bool, IntegerBoundaryFailure> {
    if !matches!(storage, IntegerStorage::I64(_) | IntegerStorage::U64(_)) {
        return Ok(true);
    }
    for start in (0..storage.len()).step_by(SCAN_CHUNK) {
        let end = start.saturating_add(SCAN_CHUNK).min(storage.len());
        let exact = (start..end).all(|index| {
            storage
                .value_at(index)
                .is_some_and(|value| integer_is_exact_f64(&value))
        });
        checkpoint(cooperation, end - start).await?;
        if !exact {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn scan_download(
    data: &HostIntegerDataOwned,
    cooperation: &mut Option<&mut CpuCooperationBudget>,
) -> Result<bool, IntegerBoundaryFailure> {
    let len = match data {
        HostIntegerDataOwned::I8(values) => values.len(),
        HostIntegerDataOwned::I16(values) => values.len(),
        HostIntegerDataOwned::I32(values) => values.len(),
        HostIntegerDataOwned::I64(values) => values.len(),
        HostIntegerDataOwned::U8(values) => values.len(),
        HostIntegerDataOwned::U16(values) => values.len(),
        HostIntegerDataOwned::U32(values) => values.len(),
        HostIntegerDataOwned::U64(values) => values.len(),
    };
    if !matches!(
        data,
        HostIntegerDataOwned::I64(_) | HostIntegerDataOwned::U64(_)
    ) {
        return Ok(true);
    }
    for start in (0..len).step_by(SCAN_CHUNK) {
        let end = start.saturating_add(SCAN_CHUNK).min(len);
        let exact = match data {
            HostIntegerDataOwned::I64(values) => values[start..end]
                .iter()
                .all(|&value| integer_is_exact_f64(&runmat_value::IntValue::I64(value))),
            HostIntegerDataOwned::U64(values) => values[start..end]
                .iter()
                .all(|&value| integer_is_exact_f64(&runmat_value::IntValue::U64(value))),
            _ => true,
        };
        checkpoint(cooperation, end - start).await?;
        if !exact {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn scan_complex_download(
    data: &HostNumericDataOwned,
    cooperation: &mut Option<&mut CpuCooperationBudget>,
) -> Result<bool, IntegerBoundaryFailure> {
    let len = data.len();
    for start in (0..len).step_by(SCAN_CHUNK) {
        let end = start.saturating_add(SCAN_CHUNK).min(len);
        let exact = match data {
            HostNumericDataOwned::I64(values) => values[start..end]
                .iter()
                .all(|&value| integer_is_exact_f64(&runmat_value::IntValue::I64(value))),
            HostNumericDataOwned::U64(values) => values[start..end]
                .iter()
                .all(|&value| integer_is_exact_f64(&runmat_value::IntValue::U64(value))),
            _ => true,
        };
        checkpoint(cooperation, end - start).await?;
        if !exact {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Prepare a direct scalar, dense, sparse, complex, or resident integer value.
/// The same invocation-owned CPU budget is shared with subsequent numerical
/// work; no budget, scheduler, or timer is created here. An absent budget is
/// the legacy synchronous caller path, not a browser cooperation claim.
pub async fn prepare_direct_integer_f64_boundary<'a>(
    source: &'a Value,
    mut cooperation: Option<&mut CpuCooperationBudget>,
) -> Result<PreparedIntegerBoundary<'a>, IntegerBoundaryFailure> {
    checkpoint(&mut cooperation, 0).await?;
    let mut resident_payload = None;
    let exact = match source {
        Value::Int(value) => integer_is_exact_f64(value),
        Value::Tensor(tensor) => match tensor.integer_storage() {
            Some(storage) => scan_storage(storage, &mut cooperation).await?,
            None => true,
        },
        Value::SparseTensor(tensor) => match tensor.integer_storage() {
            Some(storage) => scan_storage(storage, &mut cooperation).await?,
            None => true,
        },
        Value::ComplexTensor(tensor) => match tensor.integer_storage() {
            Some(storage) => {
                scan_storage(storage.real(), &mut cooperation).await?
                    && scan_storage(storage.imag(), &mut cooperation).await?
            }
            None => true,
        },
        Value::GpuTensor(handle) => {
            let Some(expected_type) = handle_integer_type(handle) else {
                return Ok(PreparedIntegerBoundary {
                    source,
                    exact: true,
                    resident_payload,
                });
            };
            let storage = handle_storage(handle);
            if handle_is_logical(handle)
                || handle_precision(handle).is_some()
                || !gpu_class_metadata_matches(handle, None, Some(expected_type), false)
            {
                return Err(IntegerBoundaryFailure::Operational(Box::new(
                    build_runtime_error(
                        "integer exactness check: input handle has contradictory integer metadata",
                    )
                    .with_gpu_gather_retry(GpuGatherRetry::Never)
                    .build(),
                )));
            }
            // A device namespace alone is not an allocation authority. Check
            // the existing lease and physical view before even a metadata-only
            // exactness receipt is returned for a narrow integer class.
            handle.allocation_lease().validate_live().map_err(|cause| {
                IntegerBoundaryFailure::Operational(Box::new(
                    build_runtime_error(format!(
                        "integer exactness check: input allocation is unavailable: {cause}"
                    ))
                    .with_source(cause)
                    .with_gpu_gather_retry(GpuGatherRetry::Never)
                    .build(),
                ))
            })?;
            handle.view().validate(handle.physical()).map_err(|cause| {
                IntegerBoundaryFailure::Operational(Box::new(
                    build_runtime_error(format!(
                        "integer exactness check: input view is invalid: {cause}"
                    ))
                    .with_source(cause)
                    .with_gpu_gather_retry(GpuGatherRetry::Never)
                    .build(),
                ))
            })?;
            let expected_len =
                crate::builtins::common::spec::checked_reduction_elements(handle.shape())
                    .ok_or_else(|| {
                        IntegerBoundaryFailure::Operational(Box::new(
                            build_runtime_error(
                                "integer exactness check: input shape exceeds platform range",
                            )
                            .with_gpu_gather_retry(GpuGatherRetry::Never)
                            .build(),
                        ))
                    })?;
            let readback = runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                    IntegerBoundaryFailure::Operational(Box::new(
                        build_runtime_error(
                            "integer exactness check: no acceleration provider owns the input handle",
                        )
                        .with_source(runmat_accelerate_api::ProviderExecutionError::from(source))
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build(),
                    ))
                })?;
            if expected_type.element_size() <= 4 {
                true
            } else {
                let _residency = ResidencyRestoreGuard::new(handle);
                if storage == GpuTensorStorage::ComplexInterleaved {
                    let gathered = readback.download_numeric().await.map_err(|source_error| {
                        let mut error = build_runtime_error(format!(
                            "integer exactness check: owner-preserving native complex download failed: {source_error}"
                        ))
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build();
                        error.source = Some(source_error.into_boxed_dyn_error());
                        IntegerBoundaryFailure::Operational(Box::new(error))
                    })?;
                    let expected_elements = expected_len.checked_mul(2);
                    if gathered.shape != handle.shape()
                        || gathered.storage != GpuTensorStorage::ComplexInterleaved
                        || gathered.data.element_type() != NumericElementType::from(expected_type)
                        || expected_elements != Some(gathered.data.len())
                    {
                        return Err(IntegerBoundaryFailure::Operational(Box::new(
                            build_runtime_error(
                                "integer exactness check: provider returned contradictory native complex payload metadata",
                            )
                            .with_gpu_gather_retry(GpuGatherRetry::Never)
                            .build(),
                        )));
                    }
                    let exact = scan_complex_download(&gathered.data, &mut cooperation).await?;
                    resident_payload = Some(PreparedIntegerPayload::Complex(gathered));
                    exact
                } else {
                    let gathered = readback.download_numeric().await.map_err(|source_error| {
                        let mut error = build_runtime_error(format!(
                        "integer exactness check: owner-preserving download failed: {source_error}"
                    ))
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build();
                        error.source = Some(source_error.into_boxed_dyn_error());
                        IntegerBoundaryFailure::Operational(Box::new(error))
                    })?;
                    if gathered.storage != GpuTensorStorage::Real {
                        return Err(IntegerBoundaryFailure::Operational(Box::new(
                            build_runtime_error("integer exactness check: provider returned nonreal integer payload")
                                .with_gpu_gather_retry(GpuGatherRetry::Never).build())));
                    }
                    // Move the existing native integer backing into the same
                    // prepared carrier; no widened numeric copy or mirror.
                    let data = match gathered.data {
                        HostNumericDataOwned::I8(data) => HostIntegerDataOwned::I8(data),
                        HostNumericDataOwned::I16(data) => HostIntegerDataOwned::I16(data),
                        HostNumericDataOwned::I32(data) => HostIntegerDataOwned::I32(data),
                        HostNumericDataOwned::I64(data) => HostIntegerDataOwned::I64(data),
                        HostNumericDataOwned::U8(data) => HostIntegerDataOwned::U8(data),
                        HostNumericDataOwned::U16(data) => HostIntegerDataOwned::U16(data),
                        HostNumericDataOwned::U32(data) => HostIntegerDataOwned::U32(data),
                        HostNumericDataOwned::U64(data) => HostIntegerDataOwned::U64(data),
                        _ => return Err(IntegerBoundaryFailure::Operational(Box::new(
                            build_runtime_error(
                                "integer exactness check: provider changed native integer class",
                            )
                            .with_gpu_gather_retry(GpuGatherRetry::Never)
                            .build(),
                        ))),
                    };
                    let gathered = HostIntegerTensorOwned {
                        data,
                        shape: gathered.shape,
                    };
                    let actual_len = match &gathered.data {
                        HostIntegerDataOwned::I8(values) => values.len(),
                        HostIntegerDataOwned::I16(values) => values.len(),
                        HostIntegerDataOwned::I32(values) => values.len(),
                        HostIntegerDataOwned::I64(values) => values.len(),
                        HostIntegerDataOwned::U8(values) => values.len(),
                        HostIntegerDataOwned::U16(values) => values.len(),
                        HostIntegerDataOwned::U32(values) => values.len(),
                        HostIntegerDataOwned::U64(values) => values.len(),
                    };
                    if gathered.shape != handle.shape()
                        || gathered.data.element_type() != expected_type
                        || expected_len != actual_len
                    {
                        return Err(IntegerBoundaryFailure::Operational(Box::new(
                        build_runtime_error(
                            "integer exactness check: provider returned contradictory integer payload metadata",
                        )
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build(),
                    )));
                    }
                    let exact = scan_download(&gathered.data, &mut cooperation).await?;
                    resident_payload = Some(PreparedIntegerPayload::Real(gathered));
                    exact
                }
            }
        }
        Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Object(_)
        | Value::Closure(_)
        | Value::OutputList(_) => {
            return Err(IntegerBoundaryFailure::AggregateNeedsTraversal);
        }
        _ => true,
    };
    Ok(PreparedIntegerBoundary {
        source,
        exact,
        resident_payload,
    })
}

#[cfg(test)]
#[path = "prepared/tests.rs"]
mod tests;
