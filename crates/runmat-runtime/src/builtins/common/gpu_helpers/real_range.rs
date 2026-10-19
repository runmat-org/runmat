//! Exact-owner scalar probes for represented real GPU inputs.
//! Extrema alone do not establish NaN absence: global provider extrema may omit NaNs.

use super::{
    binary_gpu_output_matches, download_native_values_async, exact_provider_for_handle,
    expected_handle_numeric_element_type, numeric_descriptor_matches, provider_hook_is_unsupported,
    reject_gpu_output, unary_gpu_output_matches, BinaryGpuOutputContract, GpuOutputAliasPolicy,
    UnaryGpuOutputContract,
};
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderElementType, ProviderLayout,
    ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::{ExactHostTensor2dAllocation, NumericDType, NumericScalar};

/// Upper bound for request-local host payloads of one scalar real-range probe.
/// The caller separately charges the rank-sized `ProviderRepresentation.shape`
/// payloads emitted by `hook`; provider-owned device/readback allocations are
/// governed by the selected provider's physical resource admission.
pub fn scalar_probe_host_upper_bytes(max_rank: usize) -> Option<u64> {
    fn slots<T>(count: usize) -> Option<usize> {
        count.checked_mul(std::mem::size_of::<T>())
    }

    // The extrema path reserves both output slots before its first hook.
    // Adoption protects the input and both outputs simultaneously.
    let mut bytes = slots::<GpuTensorHandle>(2)?;
    bytes = bytes.checked_add(slots::<&GpuTensorHandle>(3)?)?;
    bytes = bytes.checked_add(slots::<usize>(2)?)?; // reduction contract shape
    bytes = bytes.checked_add(slots::<f64>(2)?)?; // represented extrema
    bytes = bytes.checked_add(slots::<NumericScalar>(1)?)?; // decoded scalar
                                                            // An accepted one-element provider download is moved into one host tensor;
                                                            // the exact numeric owner's quote covers its payload, initial shape, and
                                                            // shared storage metadata. Native-value decoding clones the two-axis
                                                            // shape while that tensor remains live. The logical route may also own a
                                                            // one-byte conversion vector before the native-value vector is produced.
    bytes = bytes
        .checked_add(ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, 1, 1).ok()?)?;
    bytes = bytes.checked_add(slots::<usize>(2)?)?;
    bytes = bytes.checked_add(slots::<u8>(1)?)?;
    // The observer retains all four provider stages in its proof record, so
    // all operation identities coexist rather than only the longest one.
    for operation in ["logical_isnan", "reduce_any", "reduce_min", "reduce_max"] {
        bytes = bytes.checked_add(operation.len())?;
    }
    // The NaN proof retains one mask view at input rank and one scalar view;
    // the subsequent extrema proof instead retains two scalar views. Only
    // one proof runs at a time, but their operation identities remain in the
    // caller's observed stage vector. These are view-owner host records, not
    // device allocation payloads and not the separate observation shapes.
    let scalar_view = GpuTensorHandle::view_host_upper_bytes(2)?;
    let mask_view = GpuTensorHandle::view_host_upper_bytes(max_rank)?;
    let views = mask_view
        .checked_add(scalar_view)?
        .max(scalar_view.checked_mul(2)?);
    u64::try_from(bytes).ok()?.checked_add(views)
}

/// Selected catalog owner of probe diagnostics. The descriptor, not the
/// presentation name, supplies the externally visible error identifier.
#[derive(Clone, Copy)]
pub struct RealProbeDiagnostic {
    pub builtin: BuiltinCatalogIdentity,
    pub internal_error: &'static BuiltinErrorDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealExtrema {
    pub minimum: f64,
    pub maximum: Option<f64>,
    pub precision: ProviderPrecision,
}

#[derive(Clone, Debug)]
pub enum RealProbeStage {
    Provider {
        operation: ProviderOperationIdentity,
        input: ProviderRepresentation,
    },
    ScalarReadback,
}

fn representation(
    handle: &GpuTensorHandle,
    builtin: RealProbeDiagnostic,
) -> BuiltinResult<ProviderRepresentation> {
    let element_type = if runmat_accelerate_api::handle_is_logical(handle) {
        ProviderElementType::Logical
    } else {
        runmat_accelerate_api::handle_precision(handle)
            .ok_or_else(|| error(builtin, "probe input lacks numeric precision"))?
            .into()
    };
    Ok(ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: if runmat_accelerate_api::handle_is_transposed(handle) {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        shape: handle
            .shape()
            .iter()
            .map(|extent| u64::try_from(*extent))
            .collect::<Result<_, _>>()
            .map_err(|_| error(builtin, "probe shape exceeds representation bounds"))?,
        residency: ProviderResidency::Device,
    })
}

fn hook(
    observe: &mut dyn FnMut(RealProbeStage),
    operation: &'static str,
    input: &GpuTensorHandle,
    builtin: RealProbeDiagnostic,
) -> BuiltinResult<()> {
    observe(RealProbeStage::Provider {
        operation: ProviderOperationIdentity::new(operation),
        input: representation(input, builtin)?,
    });
    Ok(())
}

fn error(builtin: RealProbeDiagnostic, detail: impl std::fmt::Display) -> RuntimeError {
    crate::runtime_descriptor_error_with_detail(
        builtin.builtin.name,
        builtin.internal_error,
        detail.to_string(),
    )
}

fn provider_error(
    builtin: RealProbeDiagnostic,
    operation: &str,
    source: impl std::fmt::Display + Into<runmat_accelerate_api::ProviderExecutionError>,
) -> RuntimeError {
    let mut failure = error(builtin, format!("{operation} failed: {source}"));
    failure = crate::runtime_error::with_provider_execution_source(failure, source.into());
    failure
}

fn input_precision(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    builtin: RealProbeDiagnostic,
) -> BuiltinResult<ProviderPrecision> {
    let owner = exact_provider_for_handle(input)
        .ok_or_else(|| error(builtin, "GPU input has no owning provider"))?;
    input.allocation_lease().validate_live().map_err(|source| {
        error(
            builtin,
            format!("GPU input allocation is unavailable: {source}"),
        )
    })?;
    let precision = runmat_accelerate_api::handle_precision(input)
        .ok_or_else(|| error(builtin, "range reduction requires a real floating input"))?;
    if !runmat_accelerate_api::same_provider_instance(owner, provider)
        || !expected_handle_numeric_element_type(input)
            .is_ok_and(|element| numeric_descriptor_matches(input, element, GpuTensorStorage::Real))
    {
        return Err(error(
            builtin,
            "range reduction requires an owned real numeric input",
        ));
    }
    Ok(precision)
}

struct Reductions<'a> {
    input: &'a GpuTensorHandle,
    owned: Vec<GpuTensorHandle>,
}

impl Reductions<'_> {
    fn adopt(
        &mut self,
        provider: &'static dyn AccelProvider,
        output: GpuTensorHandle,
        builtin: RealProbeDiagnostic,
    ) -> BuiltinResult<()> {
        let mut protected = Vec::with_capacity(self.owned.len() + 1);
        protected.push(self.input);
        protected.extend(&self.owned);
        let contract = BinaryGpuOutputContract {
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(self.input),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        };
        let valid = !protected
            .iter()
            .any(|handle| handle.shares_allocation_with(&output))
            && binary_gpu_output_matches(&output, self.input, self.input, provider, &contract)
            && expected_handle_numeric_element_type(&output).is_ok_and(|element| {
                numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            });
        if !valid {
            return Err(reject_gpu_output(
                output,
                error(
                    builtin,
                    "provider returned malformed or aliased range reduction",
                ),
            ));
        }
        self.owned.push(output);
        Ok(())
    }
}

/// Returns represented extrema or `None` for empty, invalid reduction values,
/// or an explicitly unsupported hook. Extrema are not proof of NaN absence;
/// providers may omit mixed NaNs. Accepted provider temporaries remain owned
/// across awaits and are reclaimed if the probe is abandoned.
pub async fn probe_real_extrema(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    need_maximum: bool,
    builtin: RealProbeDiagnostic,
) -> BuiltinResult<Option<RealExtrema>> {
    probe_real_extrema_observed(provider, input, need_maximum, builtin, &mut |_| {}).await
}

pub async fn probe_real_extrema_observed(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    need_maximum: bool,
    builtin: RealProbeDiagnostic,
    observe: &mut dyn FnMut(RealProbeStage),
) -> BuiltinResult<Option<RealExtrema>> {
    if input.shape().contains(&0) {
        return Ok(None);
    }
    let precision = input_precision(provider, input, builtin)?;
    let mut reductions = Reductions {
        input,
        owned: Vec::with_capacity(if need_maximum { 2 } else { 1 }),
    };
    hook(observe, "reduce_min", input, builtin)?;
    match provider.reduce_min(input).await {
        Ok(output) => reductions.adopt(provider, output, builtin)?,
        Err(failure) if failure.is_unsupported() => return Ok(None),
        Err(failure) => return Err(provider_error(builtin, "reduce_min", failure)),
    }
    if need_maximum {
        hook(observe, "reduce_max", input, builtin)?;
        match provider.reduce_max(input).await {
            Ok(output) => reductions.adopt(provider, output, builtin)?,
            Err(failure) => {
                return Err(provider_error(
                    builtin,
                    "reduce_max",
                    failure.after_effect(),
                ));
            }
        }
    }
    let mut extrema = Vec::with_capacity(reductions.owned.len());
    for output in &reductions.owned {
        observe(RealProbeStage::ScalarReadback);
        let host = download_native_values_async(provider, output)
            .await
            .map_err(|failure| {
                crate::builtins::common::map_control_flow_with_builtin(
                    failure,
                    builtin.builtin.name,
                )
            })?;
        let [value] = host.data.as_slice() else {
            return Err(error(
                builtin,
                "range reduction did not download one scalar",
            ));
        };
        let represented = match (precision, value) {
            (ProviderPrecision::F32, NumericScalar::F32(value)) => f64::from(*value),
            (ProviderPrecision::F64, NumericScalar::F64(value)) => *value,
            _ => return Err(error(builtin, "range reduction changed numeric precision")),
        };
        if represented.is_nan() {
            return Ok(None);
        }
        extrema.push(represented);
    }
    let minimum = extrema[0];
    let maximum = extrema.get(1).copied();
    if maximum.is_some_and(|maximum| minimum > maximum) {
        return Ok(None);
    }
    Ok(Some(RealExtrema {
        minimum,
        maximum,
        precision,
    }))
}

/// `Some(true)` proves no NaN is present without gathering the input tensor.
/// Unsupported mask/reduction hooks return `None`; malformed outputs and real
/// provider failures propagate. This proof is separate from extrema because
/// the latter may legally omit mixed NaNs.
pub async fn probe_real_nan_absence(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    builtin: RealProbeDiagnostic,
) -> BuiltinResult<Option<bool>> {
    probe_real_nan_absence_observed(provider, input, builtin, &mut |_| {}).await
}

fn logical_probe_encoding_matches(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_is_logical(handle)
        && expected_handle_numeric_element_type(handle).is_ok_and(|element| {
            matches!(
                element,
                runmat_accelerate_api::NumericElementType::F32
                    | runmat_accelerate_api::NumericElementType::F64
                    | runmat_accelerate_api::NumericElementType::U8
            ) && numeric_descriptor_matches(handle, element, GpuTensorStorage::Real)
        })
}

pub async fn probe_real_nan_absence_observed(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    builtin: RealProbeDiagnostic,
    observe: &mut dyn FnMut(RealProbeStage),
) -> BuiltinResult<Option<bool>> {
    if input.shape().contains(&0) {
        return Ok(None);
    }
    input_precision(provider, input, builtin)?;
    hook(observe, "logical_isnan", input, builtin)?;
    let mask = match provider.logical_isnan(input) {
        Ok(mask) => mask,
        Err(failure) if provider_hook_is_unsupported(&failure) => return Ok(None),
        Err(failure) => return Err(provider_error(builtin, "logical_isnan", failure)),
    };
    let logical_contract = UnaryGpuOutputContract {
        storage: GpuTensorStorage::Real,
        precision: runmat_accelerate_api::handle_precision(&mask),
        integer: runmat_accelerate_api::handle_integer_type(&mask),
        logical: true,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !logical_probe_encoding_matches(&mask)
        || !unary_gpu_output_matches(&mask, input, provider, logical_contract)
    {
        return Err(reject_gpu_output(
            mask,
            error(builtin, "provider returned malformed or aliased NaN mask"),
        ));
    }
    hook(observe, "reduce_any", &mask, builtin)?;
    let any = match provider.reduce_any(&mask, false).await {
        Ok(any) => any,
        Err(failure) if provider_hook_is_unsupported(&failure) => return Ok(None),
        Err(failure) => return Err(provider_error(builtin, "reduce_any", failure)),
    };
    let aggregate_contract = BinaryGpuOutputContract {
        shape: vec![1, 1],
        storage: GpuTensorStorage::Real,
        precision: runmat_accelerate_api::handle_precision(&any),
        integer: runmat_accelerate_api::handle_integer_type(&any),
        logical: true,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !logical_probe_encoding_matches(&any)
        || any.shares_allocation_with(input)
        || !binary_gpu_output_matches(&any, &mask, input, provider, &aggregate_contract)
    {
        return Err(reject_gpu_output(
            any,
            error(
                builtin,
                "provider returned malformed or aliased NaN reduction",
            ),
        ));
    }
    observe(RealProbeStage::ScalarReadback);
    let host = download_native_values_async(provider, &any)
        .await
        .map_err(|failure| {
            crate::builtins::common::map_control_flow_with_builtin(failure, builtin.builtin.name)
        })?;
    let [NumericScalar::U8(value)] = host.data.as_slice() else {
        return Err(error(
            builtin,
            "NaN reduction did not download one logical scalar",
        ));
    };
    match *value {
        0 => Ok(Some(true)),
        1 => Ok(Some(false)),
        _ => Err(error(builtin, "NaN reduction returned a nonlogical scalar")),
    }
}

#[cfg(test)]
mod lease_tests {
    use super::*;
    use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

    #[test]
    fn reduction_container_owns_temporary_but_not_input_or_exported_alias() {
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let temporary = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let alias = temporary.handle.clone();
        let reductions = Reductions {
            input: &input.handle,
            owned: vec![temporary.handle],
        };
        drop(reductions);
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(temporary.probe.retirements(), 0);
        alias.allocation_lease().validate_live().unwrap();
        drop(alias);
        assert_eq!(temporary.probe.retirements(), 1);
        input.handle.allocation_lease().validate_live().unwrap();
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[cfg(test)]
mod tests;
