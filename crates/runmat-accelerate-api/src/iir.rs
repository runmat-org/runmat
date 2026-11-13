//! One numerical coefficient source for each provider IIR invocation.
mod index_plan;
pub use index_plan::{
    ProviderIirIndexCancelled, ProviderIirZeroPhaseIndexBuilder, ProviderIirZeroPhaseIndexPlan,
};

use runmat_execution::cancellation::CancellationToken;
use serde::Serialize;

use crate::{
    handle_is_logical, handle_storage, provider_for_handle, same_provider_instance,
    upload_numeric_protecting, AccelProvider, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage,
    HostNumericDataView, HostNumericTensorView, InvalidNumericUploadResult, NumericElementType,
    NumericUploadCompletionError, NumericUploadError, NumericUploadResult, ProviderElementType,
    ProviderFeasibilityQuery, ProviderIirRoute, ProviderIirWorkload, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload, PROVIDER_IIR_CAUSAL_PASS_OPERATION,
    PROVIDER_IIR_ZERO_PHASE_OPERATION,
};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ProviderIirCoefficientSource<'a> {
    /// The provider reads and normalizes the resident coefficient buffers.
    Resident {
        numerator: &'a GpuTensorHandle,
        denominator: &'a GpuTensorHandle,
    },
    /// The denominator is the scalar one; the numerator remains resident.
    UnitDenominatorFir { numerator: &'a GpuTensorHandle },
    /// One checked, uploaded normalization reused by subsequent passes.
    Prepared(&'a ProviderPreparedIirCoefficients),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderIirFilterOptions {
    /// Zero-based dimension along which filtering should be applied.
    pub dim: usize,
    /// Optional initial conditions (state vector) residing on the device.
    pub zi: Option<GpuTensorHandle>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderIirFilterResult {
    /// Filtered output tensor, matching the input signal shape.
    pub output: GpuTensorHandle,
    /// Final conditions in order-first state geometry: filter order followed by
    /// the signal axes other than the filtered dimension.
    pub final_state: Option<GpuTensorHandle>,
}

/// Canonical public/provider state geometry. Channel-major state buffers have
/// physical offset `channel * state_len + state_index`; the dimensions after
/// the first follow the signal's remaining column-major axis order.
pub fn provider_iir_state_shape(
    signal_shape: &[usize],
    axis: usize,
    state_len: usize,
) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(axis < signal_shape.len(), "iir_filter: invalid signal axis");
    let mut shape = Vec::new();
    shape.try_reserve_exact(signal_shape.len().max(2))?;
    shape.push(state_len);
    shape.extend(
        signal_shape
            .iter()
            .enumerate()
            .filter_map(|(index, &extent)| (index != axis).then_some(extent)),
    );
    if shape.len() == 1 {
        shape.push(1);
    }
    Ok(shape)
}

/// Describe a causal pass before coefficient/state uploads. The projected
/// device representations use the signal's actual physical class and view;
/// the provider must reject unsupported layout or precision before effects.
pub fn provider_iir_causal_query(
    input: &GpuTensorHandle,
    coefficient_len: usize,
    axis: usize,
) -> anyhow::Result<ProviderFeasibilityQuery> {
    let physical = input.physical().element_type();
    let element_type = if handle_is_logical(input) {
        ProviderElementType::Logical
    } else {
        match physical {
            NumericElementType::F32 => ProviderElementType::F32,
            NumericElementType::F64 => ProviderElementType::F64,
            NumericElementType::I8 => ProviderElementType::I8,
            NumericElementType::I16 => ProviderElementType::I16,
            NumericElementType::I32 => ProviderElementType::I32,
            NumericElementType::I64 => ProviderElementType::I64,
            NumericElementType::U8 => ProviderElementType::U8,
            NumericElementType::U16 => ProviderElementType::U16,
            NumericElementType::U32 => ProviderElementType::U32,
            NumericElementType::U64 => ProviderElementType::U64,
        }
    };
    let layout = match input.view().layout() {
        GpuTensorLayout::Contiguous => ProviderLayout::ColumnMajorContiguous,
        GpuTensorLayout::Transpose(_) => ProviderLayout::Strided,
    };
    let storage = match handle_storage(input) {
        GpuTensorStorage::Real => ProviderStorage::DenseReal,
        GpuTensorStorage::ComplexInterleaved => ProviderStorage::DenseComplexInterleaved,
    };
    let signal_shape: Vec<u64> = input
        .shape()
        .iter()
        .map(|&extent| u64::try_from(extent))
        .collect::<Result<_, _>>()?;
    let state_len = coefficient_len.saturating_sub(1);
    let state_shape: Vec<u64> = provider_iir_state_shape(input.shape(), axis, state_len)?
        .into_iter()
        .map(u64::try_from)
        .collect::<Result<_, _>>()?;
    let coefficient_len = u64::try_from(coefficient_len)?;
    let axis = u32::try_from(axis)?;
    let representation = |shape, residency, layout| ProviderRepresentation {
        element_type,
        storage,
        layout,
        shape,
        residency,
    };
    let resident = |shape| {
        representation(
            shape,
            ProviderResidency::Device,
            ProviderLayout::ColumnMajorContiguous,
        )
    };
    Ok(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(PROVIDER_IIR_CAUSAL_PASS_OPERATION),
        family: ProviderOperationFamily::Library,
        inputs: vec![
            resident(vec![coefficient_len, 1]),
            resident(vec![coefficient_len, 1]),
            representation(signal_shape.clone(), ProviderResidency::Device, layout),
            resident(state_shape.clone()),
        ],
        outputs: vec![resident(signal_shape), resident(state_shape)],
        workload: ProviderWorkload {
            iir: Some(ProviderIirWorkload {
                route: ProviderIirRoute::CausalPass,
                dim: axis,
                state_len: coefficient_len.saturating_sub(1),
                reflection_len: 0,
            }),
            ..ProviderWorkload::default()
        },
    })
}

/// One admitted zero-phase operation. The provider owns the reflection,
/// forward/reverse passes, trim, and all temporary allocations. `initial_state`
/// is the normalized steady-state vector before endpoint scaling.
#[derive(Debug, Clone)]
pub struct ProviderIirZeroPhaseRequest<'a> {
    pub input: &'a GpuTensorHandle,
    pub coefficients: &'a ProviderPreparedIirCoefficients,
    pub indices: &'a ProviderIirZeroPhaseIndexPlan,
    /// Every caller-owned resident operand retained across provider stages.
    /// These are alias exclusions, not a source of numerical coefficients.
    pub borrowed: &'a [&'a GpuTensorHandle],
    pub initial_state: &'a [f64],
    pub axis: usize,
    pub reflection_len: usize,
    /// Revision observed by side-effect-free whole-route feasibility.
    pub capability_revision: u64,
    pub cancellation: CancellationToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderIirZeroPhaseFault {
    NonFiniteInput,
    UnrepresentableReflection,
}

impl std::fmt::Display for ProviderIirZeroPhaseFault {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteInput => "iir_zero_phase: input must be finite",
            Self::UnrepresentableReflection => {
                "iir_zero_phase: endpoint reflection exceeds floating-point range"
            }
        })
    }
}

impl std::error::Error for ProviderIirZeroPhaseFault {}

impl ProviderIirZeroPhaseRequest<'_> {
    pub fn feasibility_query(&self) -> anyhow::Result<ProviderFeasibilityQuery> {
        let numerator = self.coefficients.numerator();
        let denominator = self.coefficients.denominator();
        for handle in [self.input, numerator, denominator]
            .into_iter()
            .chain(self.borrowed.iter().copied())
        {
            handle.allocation_lease().validate_live()?;
            handle.view().validate(handle.physical())?;
        }
        let input_owner = provider_for_handle(self.input)
            .ok_or_else(|| anyhow::anyhow!("iir_zero_phase: input allocation has no owner"))?;
        anyhow::ensure!(
            [numerator, denominator].into_iter().all(|handle| {
                handle.device_id() == self.input.device_id()
                    && provider_for_handle(handle)
                        .is_some_and(|owner| same_provider_instance(input_owner, owner))
            }),
            "iir_zero_phase: prepared coefficients have a different device owner"
        );
        anyhow::ensure!(
            numerator.shape().len() == 2
                && numerator.shape()[1] == 1
                && numerator.shape()[0] != 0
                && denominator.shape() == numerator.shape(),
            "iir_zero_phase: prepared coefficient geometry mismatch"
        );
        anyhow::ensure!(
            numerator.view().layout() == GpuTensorLayout::Contiguous
                && denominator.view().layout() == GpuTensorLayout::Contiguous
                && handle_storage(numerator) == GpuTensorStorage::Real
                && handle_storage(denominator) == GpuTensorStorage::Real
                && numerator.physical().element_type() == self.input.physical().element_type()
                && denominator.physical().element_type() == self.input.physical().element_type(),
            "iir_zero_phase: prepared coefficient physical class mismatch"
        );
        anyhow::ensure!(
            self.indices.matches(
                self.input.shape(),
                numerator.shape()[0],
                self.axis,
                self.reflection_len,
            ),
            "iir_zero_phase: reflected index plan does not match submitted operands"
        );
        let coefficient_len = self
            .coefficients
            .numerator()
            .shape()
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| anyhow::anyhow!("iir_zero_phase: coefficient geometry overflow"))?;
        provider_iir_zero_phase_query(self.input, coefficient_len, self.axis, self.reflection_len)
    }
}

/// Derive the complete route's immutable feasibility geometry from the actual
/// input view. Runtime admission calls this before coefficient upload; provider
/// submission rederives the same query from the received request.
pub fn provider_iir_zero_phase_query(
    input: &GpuTensorHandle,
    coefficient_len: usize,
    axis: usize,
    reflection_len: usize,
) -> anyhow::Result<ProviderFeasibilityQuery> {
    let element_type = match input.physical().element_type() {
        NumericElementType::F32 => ProviderElementType::F32,
        NumericElementType::F64 => ProviderElementType::F64,
        NumericElementType::I8 => ProviderElementType::I8,
        NumericElementType::I16 => ProviderElementType::I16,
        NumericElementType::I32 => ProviderElementType::I32,
        NumericElementType::I64 => ProviderElementType::I64,
        NumericElementType::U8 => ProviderElementType::U8,
        NumericElementType::U16 => ProviderElementType::U16,
        NumericElementType::U32 => ProviderElementType::U32,
        NumericElementType::U64 => ProviderElementType::U64,
    };
    let element_type = if handle_is_logical(input) {
        ProviderElementType::Logical
    } else {
        element_type
    };
    let layout = match input.view().layout() {
        GpuTensorLayout::Contiguous => ProviderLayout::ColumnMajorContiguous,
        GpuTensorLayout::Transpose(_) => ProviderLayout::Strided,
    };
    let signal_shape: Vec<u64> = input
        .shape()
        .iter()
        .map(|&extent| u64::try_from(extent))
        .collect::<Result<_, _>>()?;
    let coefficient_len = u64::try_from(coefficient_len)?;
    let state_len = coefficient_len.saturating_sub(1);
    let reflection_len = u64::try_from(reflection_len)?;
    let axis = u32::try_from(axis)?;
    let storage = match handle_storage(input) {
        GpuTensorStorage::Real => ProviderStorage::DenseReal,
        GpuTensorStorage::ComplexInterleaved => ProviderStorage::DenseComplexInterleaved,
    };
    let representation = |shape: Vec<u64>, residency, layout, storage| ProviderRepresentation {
        element_type,
        storage,
        layout,
        shape,
        residency,
    };
    let host = |shape| {
        representation(
            shape,
            ProviderResidency::Host,
            ProviderLayout::ColumnMajorContiguous,
            ProviderStorage::DenseReal,
        )
    };
    // Match provider_iir_state_shape: the leading dimension is filter order,
    // irrespective of the signal's filtering axis.
    let state_shape = vec![state_len, 1];
    let signal = representation(
        signal_shape.clone(),
        ProviderResidency::Device,
        layout,
        storage,
    );
    let output = representation(
        signal_shape,
        ProviderResidency::Device,
        ProviderLayout::ColumnMajorContiguous,
        storage,
    );
    Ok(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(PROVIDER_IIR_ZERO_PHASE_OPERATION),
        family: ProviderOperationFamily::Library,
        inputs: vec![
            host(vec![coefficient_len, 1]),
            host(vec![coefficient_len, 1]),
            signal,
            host(state_shape),
        ],
        outputs: vec![output],
        workload: ProviderWorkload {
            iir: Some(ProviderIirWorkload {
                route: ProviderIirRoute::ZeroPhase,
                dim: axis,
                state_len,
                reflection_len,
            }),
            ..ProviderWorkload::default()
        },
    })
}

/// Provider-bound normalized coefficients. The constructor checks the structural
/// normalization contract and owns both uploads. A failed second upload retires
/// the first rather than publishing a partial preparation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderPreparedIirCoefficients {
    numerator: GpuTensorHandle,
    denominator: GpuTensorHandle,
}

impl ProviderPreparedIirCoefficients {
    pub fn upload(
        provider: &'static dyn AccelProvider,
        numerator: &[f64],
        denominator: &[f64],
        precision: ProviderPrecision,
        protected: &[&GpuTensorHandle],
    ) -> NumericUploadResult<Self> {
        if numerator.is_empty() || numerator.len() != denominator.len() {
            return Err(NumericUploadError::from(anyhow::anyhow!(
                "iir_filter: prepared coefficient lengths are incompatible"
            )));
        }
        if denominator.first() != Some(&1.0) {
            return Err(NumericUploadError::from(anyhow::anyhow!(
                "iir_filter: prepared leading denominator must equal one"
            )));
        }
        if numerator
            .iter()
            .chain(denominator)
            .any(|value| !value.is_finite())
        {
            return Err(NumericUploadError::from(anyhow::anyhow!(
                "iir_filter: prepared coefficients must be finite"
            )));
        }
        if provider.precision() != precision {
            return Err(NumericUploadError::from(anyhow::anyhow!(
                "iir_filter: admitted precision is unavailable on provider"
            )));
        }
        for handle in protected {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|source| NumericUploadError::from(anyhow::Error::new(source)))?;
        }
        let shape = [numerator.len(), 1];
        let (numerator_single, denominator_single) = match precision {
            ProviderPrecision::F32 => {
                let narrow = |values: &[f64]| -> NumericUploadResult<Vec<f32>> {
                    let mut result = Vec::new();
                    result
                        .try_reserve_exact(values.len())
                        .map_err(|source| NumericUploadError::from(anyhow::Error::new(source)))?;
                    for &value in values {
                        let converted = value as f32;
                        if !converted.is_finite() {
                            return Err(NumericUploadError::from(anyhow::anyhow!(
                                "iir_filter: prepared coefficient exceeds single precision range"
                            )));
                        }
                        result.push(converted);
                    }
                    Ok(result)
                };
                (Some(narrow(numerator)?), Some(narrow(denominator)?))
            }
            ProviderPrecision::F64 => (None, None),
        };
        let numerator_data = match &numerator_single {
            Some(values) => HostNumericDataView::F32(values),
            None => HostNumericDataView::F64(numerator),
        };
        let denominator_data = match &denominator_single {
            Some(values) => HostNumericDataView::F32(values),
            None => HostNumericDataView::F64(denominator),
        };
        let mut denominator_protected = Vec::new();
        let protected_len = protected.len().checked_add(1).ok_or_else(|| {
            NumericUploadError::from(anyhow::anyhow!(
                "iir_filter: borrowed coefficient count overflow"
            ))
        })?;
        denominator_protected
            .try_reserve_exact(protected_len)
            .map_err(|source| NumericUploadError::from(anyhow::Error::new(source)))?;
        denominator_protected.extend_from_slice(protected);
        let numerator_handle = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: numerator_data,
                shape: &shape,
                storage: GpuTensorStorage::Real,
            },
            protected,
        )?;
        let numerator_handle = require_exact_owner(provider, numerator_handle)?;
        denominator_protected.push(&numerator_handle);
        let denominator_result = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: denominator_data,
                shape: &shape,
                storage: GpuTensorStorage::Real,
            },
            &denominator_protected,
        )
        .and_then(|handle| require_exact_owner(provider, handle));
        let denominator_handle = match denominator_result {
            Ok(handle) => handle,
            Err(primary) => {
                return Err(NumericUploadError::Completion(
                    NumericUploadCompletionError::new(primary, numerator_handle.release().err()),
                ));
            }
        };
        Ok(Self {
            numerator: numerator_handle,
            denominator: denominator_handle,
        })
    }

    pub fn numerator(&self) -> &GpuTensorHandle {
        &self.numerator
    }

    pub fn denominator(&self) -> &GpuTensorHandle {
        &self.denominator
    }
}

fn require_exact_owner(
    provider: &'static dyn AccelProvider,
    handle: GpuTensorHandle,
) -> NumericUploadResult<GpuTensorHandle> {
    if provider_for_handle(&handle).is_some_and(|owner| same_provider_instance(owner, provider)) {
        Ok(handle)
    } else {
        Err(InvalidNumericUploadResult::reject(
            handle,
            "iir_filter: prepared coefficient allocation owner mismatch".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccelDownloadFuture, HostTensorView, NumericElementType};
    use std::sync::{atomic::Ordering, Mutex};

    #[test]
    fn state_geometry_is_order_first_for_every_filtered_axis() {
        assert_eq!(provider_iir_state_shape(&[3, 4], 1, 1).unwrap(), vec![1, 3]);
        assert_eq!(
            provider_iir_state_shape(&[2, 3, 4], 1, 2).unwrap(),
            vec![2, 2, 4]
        );
        assert_eq!(provider_iir_state_shape(&[3, 4], 0, 2).unwrap(), vec![2, 4]);
        assert!(provider_iir_state_shape(&[3, 4], 2, 1).is_err());
    }

    struct UploadProvider {
        queued: Mutex<Vec<GpuTensorHandle>>,
        observed_single: Mutex<Vec<Vec<f32>>>,
        precision: ProviderPrecision,
    }

    impl AccelProvider for UploadProvider {
        fn device_id(&self) -> u32 {
            17
        }
        fn device_info(&self) -> String {
            "prepared-iir-upload-test".into()
        }
        fn precision(&self) -> ProviderPrecision {
            self.precision
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("legacy upload must not run")
        }
        fn upload_numeric_raw(
            &self,
            host: &HostNumericTensorView,
        ) -> crate::ProviderExecutionResult<GpuTensorHandle> {
            if let HostNumericDataView::F32(data) = host.data {
                self.observed_single.lock().unwrap().push(data.to_vec());
            }
            Ok(self
                .queued
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing queued handle"))?)
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("unused download") })
        }
    }

    #[test]
    fn prepared_upload_rejects_borrowed_coefficient_alias_without_retiring_caller() {
        let (borrowed, owner) = crate::test_support::owned(
            17,
            vec![2, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
            false,
        );
        let provider: &'static UploadProvider = Box::leak(Box::new(UploadProvider {
            queued: Mutex::new(vec![borrowed.clone()]),
            observed_single: Mutex::new(Vec::new()),
            precision: ProviderPrecision::F64,
        }));
        let error = ProviderPreparedIirCoefficients::upload(
            provider,
            &[0.5, 0.5],
            &[1.0, 0.0],
            ProviderPrecision::F64,
            &[&borrowed],
        )
        .expect_err("borrowed coefficient may not be reused for prepared upload");
        assert!(crate::is_invalid_numeric_upload_result(&error));
        assert_eq!(owner.retired.load(Ordering::SeqCst), 0);
        drop(borrowed);
        assert_eq!(owner.retired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn prepared_single_upload_requests_native_single_payload() {
        let (numerator, _) = crate::test_support::owned(
            17,
            vec![2, 1],
            NumericElementType::F32,
            GpuTensorStorage::Real,
            false,
        );
        let provider: &'static UploadProvider = Box::leak(Box::new(UploadProvider {
            queued: Mutex::new(vec![numerator]),
            observed_single: Mutex::new(Vec::new()),
            precision: ProviderPrecision::F32,
        }));
        let error = ProviderPreparedIirCoefficients::upload(
            provider,
            &[0.5, 0.5],
            &[1.0, 0.0],
            ProviderPrecision::F32,
            &[],
        )
        .expect_err("the synthetic owner is not the invoking provider");
        assert!(crate::is_invalid_numeric_upload_result(&error));
        assert_eq!(
            *provider.observed_single.lock().unwrap(),
            vec![vec![0.5, 0.5]]
        );
    }
}
