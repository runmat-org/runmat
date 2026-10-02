//! One numerical coefficient source for each provider IIR invocation.
use serde::Serialize;

use crate::{
    provider_for_handle, same_provider_instance, upload_numeric_protecting, AccelProvider,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    InvalidNumericUploadResult, NumericUploadError, NumericUploadResult, ProviderPrecision,
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
    /// Final conditions for the filter state (same shape as the requested `zi` layout).
    pub final_state: Option<GpuTensorHandle>,
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
            return Err(NumericUploadError::Provider(anyhow::anyhow!(
                "iir_filter: prepared coefficient lengths are incompatible"
            )));
        }
        if denominator.first() != Some(&1.0) {
            return Err(NumericUploadError::Provider(anyhow::anyhow!(
                "iir_filter: prepared leading denominator must equal one"
            )));
        }
        if numerator
            .iter()
            .chain(denominator)
            .any(|value| !value.is_finite())
        {
            return Err(NumericUploadError::Provider(anyhow::anyhow!(
                "iir_filter: prepared coefficients must be finite"
            )));
        }
        if provider.precision() != precision {
            return Err(NumericUploadError::Provider(anyhow::anyhow!(
                "iir_filter: admitted precision is unavailable on provider"
            )));
        }
        for handle in protected {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|source| NumericUploadError::Provider(anyhow::Error::new(source)))?;
        }
        let shape = [numerator.len(), 1];
        let (numerator_single, denominator_single) = match precision {
            ProviderPrecision::F32 => {
                let narrow = |values: &[f64]| -> NumericUploadResult<Vec<f32>> {
                    let mut result = Vec::new();
                    result.try_reserve_exact(values.len()).map_err(|source| {
                        NumericUploadError::Provider(anyhow::Error::new(source))
                    })?;
                    for &value in values {
                        let converted = value as f32;
                        if !converted.is_finite() {
                            return Err(NumericUploadError::Provider(anyhow::anyhow!(
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
        denominator_protected
            .try_reserve_exact(protected.len().saturating_add(1))
            .map_err(|source| NumericUploadError::Provider(anyhow::Error::new(source)))?;
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
        let denominator_handle = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: denominator_data,
                shape: &shape,
                storage: GpuTensorStorage::Real,
            },
            &denominator_protected,
        )?;
        let denominator_handle = require_exact_owner(provider, denominator_handle)?;
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
        ) -> anyhow::Result<GpuTensorHandle> {
            if let HostNumericDataView::F32(data) = host.data {
                self.observed_single.lock().unwrap().push(data.to_vec());
            }
            self.queued
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing queued handle"))
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
