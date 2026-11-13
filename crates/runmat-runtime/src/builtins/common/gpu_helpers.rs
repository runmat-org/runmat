use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostIntegerDataOwned, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorOwned,
    IntegerElementType, NumericElementType, PreparedNumericDownloadStage, ProviderPrecision,
};
use runmat_types::{standard, ClassIdentity};
use runmat_value::{
    ComplexStorage, ComplexTensor, IntegerStorage, LogicalArray, NumericDType, Tensor, Value,
};
use runmat_value::{IntegerComplexStorage, NumericScalar, NumericStorage};

use crate::build_runtime_error;

pub(crate) mod host_upload;
mod managed_download;
pub(crate) mod query_metadata;
mod residency_cleanup;
pub use managed_download::{
    adopt_settled_native_download, download_exact_native, download_exact_native_readback,
    quote_exact_native_download, quote_exact_native_readback, ExactDownloadFailure,
    ExactDownloadPhase, ExactNativeDownloadQuote,
};
pub(crate) use managed_download::{
    download_exact_native_readback_with_diagnostics, download_readback_value_in_context,
    download_readback_value_with_admission_in_context,
    download_value_preserving_residency_in_context,
};
pub(crate) mod complex_transfer;
pub use complex_transfer::native_complex_upload_workspace;
pub(crate) use residency_cleanup::clear_input_residency;
mod real_range;
pub use real_range::{
    probe_real_extrema, probe_real_extrema_observed, probe_real_nan_absence,
    probe_real_nan_absence_observed, scalar_probe_host_upper_bytes, RealExtrema,
    RealProbeDiagnostic, RealProbeStage,
};

pub fn integer_storage_from_owned(data: HostIntegerDataOwned) -> IntegerStorage {
    match data {
        HostIntegerDataOwned::I8(values) => IntegerStorage::I8(values),
        HostIntegerDataOwned::I16(values) => IntegerStorage::I16(values),
        HostIntegerDataOwned::I32(values) => IntegerStorage::I32(values),
        HostIntegerDataOwned::I64(values) => IntegerStorage::I64(values),
        HostIntegerDataOwned::U8(values) => IntegerStorage::U8(values),
        HostIntegerDataOwned::U16(values) => IntegerStorage::U16(values),
        HostIntegerDataOwned::U32(values) => IntegerStorage::U32(values),
        HostIntegerDataOwned::U64(values) => IntegerStorage::U64(values),
    }
}

/// Resolve the provider that actually owns `handle`.
///
/// `provider_for_handle` retains a legacy active-provider fallback for older
/// callers. Handle operations must additionally prove that the provider's
/// device namespace matches the durable handle identity before touching it.
pub fn exact_provider_for_handle(handle: &GpuTensorHandle) -> Option<&'static dyn AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|provider| provider.device_id() == handle.device_id())
}

/// A pre-effect resident compute lookup must distinguish execution policy
/// from a missing, stale or differently assigned physical owner.
#[derive(Debug)]
pub enum ResidentComputeDecline {
    PolicyDenied,
    InvalidOwner,
    InvalidSource(runmat_accelerate_api::ProviderExecutionError),
}

#[derive(Debug)]
pub struct ResidentComputePolicyDenied {
    original_source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}
impl std::fmt::Display for ResidentComputePolicyDenied {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("resident computation denied by invocation acceleration policy")
    }
}
impl std::error::Error for ResidentComputePolicyDenied {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.original_source.as_deref().map(|source| source as _)
    }
}
impl ResidentComputeDecline {
    /// Retain the domain diagnostic and the original ownership failure.
    /// Only a proven, pre-effect policy decline requests dispatcher migration.
    pub fn with_domain_error(self, mut error: crate::RuntimeError) -> crate::RuntimeError {
        match self {
            Self::PolicyDenied => {
                error.gpu_gather_retry = crate::GpuGatherRetry::Requested;
                error.source = Some(Box::new(ResidentComputePolicyDenied {
                    original_source: error.source.take(),
                }));
                error
            }
            Self::InvalidOwner => {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            }
            Self::InvalidSource(source) => {
                crate::runtime_error::with_provider_execution_source(error, source)
            }
        }
    }
}

/// Used only at canonical resident compute ingress, before provider effects.
/// A readback capability proves that data egress is legal without granting
/// compute. Capability metadata proves the exact registered or scoped-thread
/// owner; a failed compute lookup alone is never evidence of policy denial.
pub fn resolve_resident_compute_owner(
    handle: &GpuTensorHandle,
) -> Result<&'static dyn AccelProvider, ResidentComputeDecline> {
    let readback = runmat_accelerate_api::numeric_readback_for_handle(handle)
        .map_err(|source| ResidentComputeDecline::InvalidSource(source.into()))?;
    if let Some(provider) = exact_provider_for_handle(handle) {
        // First registration can supersede a scoped owner between lookups.
        // Computation must use the same facade whose source was validated.
        readback
            .authorize_provider(provider)
            .map_err(|source| ResidentComputeDecline::InvalidSource(source.into()))?;
        return Ok(provider);
    }
    let has_wgpu_access = readback
        .owner_has_wgpu_access()
        .map_err(|source| ResidentComputeDecline::InvalidSource(source.into()))?;
    Err(
        if crate::context::current_acceleration_policy()
            .is_some_and(|policy| !policy.permits_provider_metadata(has_wgpu_access))
        {
            ResidentComputeDecline::PolicyDenied
        } else {
            ResidentComputeDecline::InvalidOwner
        },
    )
}

pub fn require_resident_compute_owner(
    handle: &GpuTensorHandle,
    error: impl FnOnce() -> crate::RuntimeError,
) -> crate::BuiltinResult<&'static dyn AccelProvider> {
    resolve_resident_compute_owner(handle).map_err(|decline| decline.with_domain_error(error()))
}

/// Validate the complete pre-effect source set before admitting migration.
/// Integrity/assignment errors (including a required same-owner mismatch)
/// dominate a policy decline on any sibling. No input or provider is cloned.
pub fn require_resident_compute_owners(
    handles: &[&GpuTensorHandle],
    require_same_owner: bool,
    error: impl FnOnce() -> crate::RuntimeError,
) -> crate::BuiltinResult<()> {
    require_resident_compute_set(handles.iter().copied(), require_same_owner, error)
}

/// Direct admitted data roles only; container/identity inspection is not a
/// compute ingress and deliberately does not use this bridge.
pub fn require_resident_compute_values(
    values: &[&Value],
    require_same_owner: bool,
    error: impl FnOnce() -> crate::RuntimeError,
) -> crate::BuiltinResult<()> {
    require_resident_compute_value_iter(values.iter().copied(), require_same_owner, error)
}

pub fn require_resident_compute_value_iter<'a>(
    values: impl Iterator<Item = &'a Value>,
    require_same_owner: bool,
    error: impl FnOnce() -> crate::RuntimeError,
) -> crate::BuiltinResult<()> {
    require_resident_compute_set(
        values.filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        }),
        require_same_owner,
        error,
    )
}

fn require_resident_compute_set<'a>(
    handles: impl Iterator<Item = &'a GpuTensorHandle>,
    require_same_owner: bool,
    error: impl FnOnce() -> crate::RuntimeError,
) -> crate::BuiltinResult<()> {
    let mut policy_denied = false;
    let mut invalid = None;
    let mut first_device = None;
    for handle in handles {
        match resolve_resident_compute_owner(handle) {
            Ok(_) => {}
            Err(ResidentComputeDecline::PolicyDenied) => policy_denied = true,
            Err(decline) => {
                invalid.get_or_insert(decline);
            }
        }
        if require_same_owner && first_device.is_some_and(|first| first != handle.device_id()) {
            invalid.get_or_insert(ResidentComputeDecline::InvalidOwner);
        }
        first_device.get_or_insert(handle.device_id());
    }
    if let Some(decline) = invalid {
        Err(decline.with_domain_error(error()))
    } else if policy_denied {
        Err(ResidentComputeDecline::PolicyDenied.with_domain_error(error()))
    } else {
        Ok(())
    }
}

/// Resolve one exact owner for two resident operands.
///
/// Device identifiers are provider-local, so matching identifiers alone do not
/// establish shared ownership. Both handles must resolve to the same registered
/// provider instance and the same device before a binary hook may consume them.
pub fn exact_provider_for_binary_inputs(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Result<&'static dyn AccelProvider, String> {
    let left_owner = exact_provider_for_handle(left)
        .ok_or_else(|| "no provider owns the left resident operand".to_string())?;
    let right_owner = exact_provider_for_handle(right)
        .ok_or_else(|| "no provider owns the right resident operand".to_string())?;
    if !runmat_accelerate_api::same_provider_instance(left_owner, right_owner)
        || left.device_id() != right.device_id()
    {
        return Err("resident operands must share one owning provider and device".to_string());
    }
    Ok(left_owner)
}

/// Select one owning provider for a set of resident inputs, with explicit
/// gpuArray provenance taking precedence over automatic residency.
pub fn select_resident_output_source(
    handles: impl IntoIterator<Item = GpuTensorHandle>,
    builtin: &str,
) -> crate::BuiltinResult<Option<GpuTensorHandle>> {
    let mut selected: Option<GpuTensorHandle> = None;
    for handle in handles {
        let ownership_error = |source: anyhow::Error| {
            build_runtime_error(format!(
                "{builtin}: no acceleration provider owns a resident input"
            ))
            .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(runmat_accelerate_api::ProviderExecutionError::from(source))
            .build()
        };
        let readback =
            runmat_accelerate_api::numeric_readback_for_handle(&handle).map_err(ownership_error)?;
        if let Some(current) = &selected {
            let current_readback = runmat_accelerate_api::numeric_readback_for_handle(current)
                .map_err(ownership_error)?;
            if current.device_id() != handle.device_id()
                || !current_readback
                    .same_owner(&readback)
                    .map_err(ownership_error)?
            {
                return Err(build_runtime_error(format!(
                    "{builtin}: resident inputs must share one owning provider"
                ))
                .with_identifier("RunMat:gpu:MixedProviders")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build());
            }
            if !runmat_accelerate_api::handle_is_explicit(current)
                && runmat_accelerate_api::handle_is_explicit(&handle)
            {
                selected = Some(handle);
            }
        } else {
            selected = Some(handle);
        }
    }
    Ok(selected)
}

pub use runmat_accelerate_api::ResidencyRestoreGuard;

mod adoption;
pub use adoption::{complex_gpu_value, logical_gpu_value, resident_gpu_value};
pub(crate) use adoption::{propagate_output_provenance, propagate_owned_output_provenance};
mod rejection;
pub(crate) use rejection::{reject_gpu_output, reject_gpu_outputs};
#[path = "gpu_helpers/structural.rs"]
mod structural;
pub(crate) use structural::{
    flip_output_matches, permutation_output_matches, replication_output_matches, replication_shape,
    reshape_output_matches,
};

#[cfg(test)]
#[path = "gpu_helpers/owned_encoding_tests.rs"]
mod owned_encoding_tests;

pub fn expected_gpu_class_identity(
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
    logical: bool,
) -> Option<ClassIdentity> {
    if logical {
        return Some(standard::LOGICAL.owned());
    }
    if let Some(integer) = integer {
        return Some(NumericElementType::from(integer).class_identity());
    }
    precision.map(|precision| match precision {
        ProviderPrecision::F32 => standard::SINGLE.owned(),
        ProviderPrecision::F64 => standard::DOUBLE.owned(),
    })
}

pub fn gpu_class_metadata_matches(
    handle: &GpuTensorHandle,
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
    logical: bool,
) -> bool {
    let expected = expected_gpu_class_identity(precision, integer, logical);
    runmat_accelerate_api::handle_class_identity(handle)
        .is_none_or(|actual| expected.as_ref() == Some(&actual))
}

fn integer_element_type(storage: &IntegerStorage) -> IntegerElementType {
    match storage {
        IntegerStorage::I8(_) => IntegerElementType::I8,
        IntegerStorage::I16(_) => IntegerElementType::I16,
        IntegerStorage::I32(_) => IntegerElementType::I32,
        IntegerStorage::I64(_) => IntegerElementType::I64,
        IntegerStorage::U8(_) => IntegerElementType::U8,
        IntegerStorage::U16(_) => IntegerElementType::U16,
        IntegerStorage::U32(_) => IntegerElementType::U32,
        IntegerStorage::U64(_) => IntegerElementType::U64,
    }
}

pub(crate) fn expected_handle_numeric_element_type(
    handle: &GpuTensorHandle,
) -> Result<NumericElementType, String> {
    handle
        .allocation_lease()
        .validate_live()
        .map_err(|error| error.to_string())?;
    // Physical class and logical interpretation are immutable owner-admitted
    // facts. In particular, logical U8 is a supported physical encoding.
    Ok(handle.physical().element_type())
}

pub(crate) fn numeric_descriptor_matches(
    handle: &GpuTensorHandle,
    element_type: NumericElementType,
    storage: GpuTensorStorage,
) -> bool {
    handle.descriptor().element_type == Some(element_type)
        && handle.descriptor().storage == Some(storage)
}

pub(crate) fn numeric_descriptor_matches_source(
    source: &GpuTensorHandle,
    output: &GpuTensorHandle,
) -> bool {
    expected_handle_numeric_element_type(source).is_ok_and(|element_type| {
        numeric_descriptor_matches(
            output,
            element_type,
            runmat_accelerate_api::handle_storage(source),
        )
    })
}

fn deinterleave<T: Copy>(values: &[T]) -> (Vec<T>, Vec<T>) {
    let mut real = Vec::with_capacity(values.len() / 2);
    let mut imag = Vec::with_capacity(values.len() / 2);
    for pair in values.chunks_exact(2) {
        real.push(pair[0]);
        imag.push(pair[1]);
    }
    (real, imag)
}

pub(crate) fn value_from_numeric_download(
    host: HostNumericTensorOwned,
    logical: bool,
) -> Result<Value, String> {
    host.validate().map_err(|error| error.to_string())?;
    if logical {
        if host.storage != GpuTensorStorage::Real {
            return Err("logical payload must use real storage".into());
        }
        let bits = match host.data {
            HostNumericDataOwned::F64(values) => {
                runmat_value::exact_capacity_vec_with(values.len(), |index| {
                    u8::from(values[index] != 0.0)
                })
                .map_err(|error| error.to_string())?
            }
            HostNumericDataOwned::F32(values) => {
                runmat_value::exact_capacity_vec_with(values.len(), |index| {
                    u8::from(values[index] != 0.0)
                })
                .map_err(|error| error.to_string())?
            }
            HostNumericDataOwned::U8(values) => {
                runmat_value::exact_capacity_vec_with(values.len(), |index| {
                    u8::from(values[index] != 0)
                })
                .map_err(|error| error.to_string())?
            }
            HostNumericDataOwned::I8(_)
            | HostNumericDataOwned::I16(_)
            | HostNumericDataOwned::I32(_)
            | HostNumericDataOwned::I64(_)
            | HostNumericDataOwned::U16(_)
            | HostNumericDataOwned::U32(_)
            | HostNumericDataOwned::U64(_) => {
                return Err("logical payload cannot use native integer storage".into());
            }
        };
        return LogicalArray::new(bits, host.shape).map(Value::LogicalArray);
    }

    if host.storage == GpuTensorStorage::Real {
        let storage = match host.data {
            HostNumericDataOwned::F64(values) => NumericStorage::F64(values),
            HostNumericDataOwned::F32(values) => NumericStorage::F32(values),
            HostNumericDataOwned::I8(values) => NumericStorage::I8(values),
            HostNumericDataOwned::I16(values) => NumericStorage::I16(values),
            HostNumericDataOwned::I32(values) => NumericStorage::I32(values),
            HostNumericDataOwned::I64(values) => NumericStorage::I64(values),
            HostNumericDataOwned::U8(values) => NumericStorage::U8(values),
            HostNumericDataOwned::U16(values) => NumericStorage::U16(values),
            HostNumericDataOwned::U32(values) => NumericStorage::U32(values),
            HostNumericDataOwned::U64(values) => NumericStorage::U64(values),
        };
        return Tensor::from_numeric_storage(storage, host.shape).map(Value::Tensor);
    }

    macro_rules! integer_complex {
        ($values:expr, $variant:ident) => {{
            let (real, imag) = deinterleave(&$values);
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::$variant(real),
                IntegerStorage::$variant(imag),
            )?)
        }};
    }
    let storage = match host.data {
        HostNumericDataOwned::F64(values) => {
            let (real, imag) = deinterleave(&values);
            ComplexStorage::F64(real.into_iter().zip(imag).collect())
        }
        HostNumericDataOwned::F32(values) => {
            let (real, imag) = deinterleave(&values);
            ComplexStorage::F32(real.into_iter().zip(imag).collect())
        }
        HostNumericDataOwned::I8(values) => integer_complex!(values, I8),
        HostNumericDataOwned::I16(values) => integer_complex!(values, I16),
        HostNumericDataOwned::I32(values) => integer_complex!(values, I32),
        HostNumericDataOwned::I64(values) => integer_complex!(values, I64),
        HostNumericDataOwned::U8(values) => integer_complex!(values, U8),
        HostNumericDataOwned::U16(values) => integer_complex!(values, U16),
        HostNumericDataOwned::U32(values) => integer_complex!(values, U32),
        HostNumericDataOwned::U64(values) => integer_complex!(values, U64),
    };
    ComplexTensor::from_complex_storage(storage, host.shape).map(Value::ComplexTensor)
}

pub fn same_gpu_handle(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    left.view_identity() == right.view_identity()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOutputAliasPolicy {
    AllowInput,
    /// A separate immutable view, possibly sharing the physical allocation.
    RequireDistinct,
    /// Fresh computational storage; another view of the input is not sufficient.
    RequireDistinctAllocation,
}

impl GpuOutputAliasPolicy {
    fn accepts(self, output: &GpuTensorHandle, input: &GpuTensorHandle) -> bool {
        match self {
            Self::AllowInput => true,
            Self::RequireDistinct => !same_gpu_handle(output, input),
            Self::RequireDistinctAllocation => !output.shares_allocation_with(input),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryGpuOutputContract {
    pub storage: GpuTensorStorage,
    pub precision: Option<ProviderPrecision>,
    pub integer: Option<IntegerElementType>,
    pub logical: bool,
    pub alias: GpuOutputAliasPolicy,
}

pub fn unary_gpu_output_matches(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    contract: UnaryGpuOutputContract,
) -> bool {
    unary_gpu_output_matches_shape(output, input, provider, input.shape(), contract)
}

/// Validate a unary result whose operation changes logical shape, such as
/// transpose or reshape. Ownership and numerical class checks are identical
/// to shape-preserving unary operations.
/// The output allocation must remain live; callers separately admit borrowed inputs.
pub fn unary_gpu_output_matches_shape(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    shape: &[usize],
    contract: UnaryGpuOutputContract,
) -> bool {
    output.allocation_lease().validate_live().is_ok()
        && output.shape() == shape
        && output.device_id() == input.device_id()
        && contract.alias.accepts(output, input)
        // Device namespace was checked above and by exact_provider_for_handle.
        // Compare the retained facade address, not codegen-dependent dyn vtables.
        && exact_provider_for_handle(output).is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == contract.storage
        && runmat_accelerate_api::handle_precision(output) == contract.precision
        && runmat_accelerate_api::handle_integer_type(output) == contract.integer
        && runmat_accelerate_api::handle_is_logical(output) == contract.logical
        && gpu_class_metadata_matches(
            output,
            contract.precision,
            contract.integer,
            contract.logical,
        )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryGpuOutputContract {
    pub shape: Vec<usize>,
    pub storage: GpuTensorStorage,
    pub precision: Option<ProviderPrecision>,
    pub integer: Option<IntegerElementType>,
    pub logical: bool,
    pub alias: GpuOutputAliasPolicy,
}

pub fn binary_gpu_output_matches(
    output: &GpuTensorHandle,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    contract: &BinaryGpuOutputContract,
) -> bool {
    output.allocation_lease().validate_live().is_ok()
        && output.shape() == contract.shape
        && output.device_id() == provider.device_id()
        && contract.alias.accepts(output, left)
        && contract.alias.accepts(output, right)
        // Registered live owners have distinct device namespaces. The address
        // additionally rejects a substituted facade in that same namespace;
        // dyn vtable equality is not stable across async/codegen boundaries.
        && exact_provider_for_handle(output).is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == contract.storage
        && runmat_accelerate_api::handle_precision(output) == contract.precision
        && runmat_accelerate_api::handle_integer_type(output) == contract.integer
        && runmat_accelerate_api::handle_is_logical(output) == contract.logical
        && gpu_class_metadata_matches(
            output,
            contract.precision,
            contract.integer,
            contract.logical,
        )
}

/// Return whether an optional provider hook reported the API's typed
/// unsupported-operation result. Other provider failures must remain visible
/// instead of silently changing the execution path.
pub fn provider_hook_is_unsupported(error: &anyhow::Error) -> bool {
    runmat_accelerate_api::is_unsupported_provider_operation(error)
}

/// Admit an unsupported provider hook's host implementation in the current
/// invocation. Classification alone is not authority to download or execute
/// host work; callers retain the original error when fallback is forbidden.
pub fn provider_hook_allows_host_fallback(error: &anyhow::Error) -> bool {
    provider_hook_is_unsupported(error)
        && crate::context::current_acceleration_policy()
            .is_none_or(|policy| policy.allow_host_fallback)
}

/// Download a GPU tensor handle to host memory, returning a dense `Tensor`.
///
/// This helper routes through the dispatcher so residency hooks and provider
/// semantics stay consistent with the rest of the runtime.
pub async fn gather_tensor_async(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> crate::BuiltinResult<Tensor> {
    let value = Value::GpuTensor(handle.clone());
    let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
    match gathered {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| build_runtime_error(format!("gather: {e}")).build()),
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(data, la.shape.clone())
                .map_err(|e| build_runtime_error(format!("gather: {e}")).build())
        }
        other => {
            Err(build_runtime_error(format!("gather: unexpected value kind {other:?}")).build())
        }
    }
}

/// Gather an arbitrary value, returning a host-side `Value`.
pub async fn gather_value_async(value: &Value) -> crate::BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(value).await
}

/// Download a handle through its owner without changing the handle's residency or metadata.
pub async fn download_value_preserving_residency_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> crate::BuiltinResult<Value> {
    download_value_preserving_residency_with_admission_async(
        provider,
        handle,
        NumericDownloadAdmission::None,
    )
    .await
}

/// A selected finite readback carries the provider mapping callback's exact
/// retained claim across cancellation; the native payload has a separate
/// selected host-workspace grant.
pub(crate) async fn download_value_preserving_residency_admitted_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    completion_claim: std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
) -> crate::BuiltinResult<Value> {
    download_value_preserving_residency_with_admission_async(
        provider,
        handle,
        NumericDownloadAdmission::Callback(completion_claim),
    )
    .await
}

/// A provider-issued stage proves selected physical scratch/queue ownership
/// before download. This shares the ordinary decoder and source checks.
pub async fn download_value_preserving_residency_staged_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    stage: PreparedNumericDownloadStage,
) -> crate::BuiltinResult<Value> {
    download_value_preserving_residency_with_admission_async(
        provider,
        handle,
        NumericDownloadAdmission::Stage(stage),
    )
    .await
}

/// Preserve the provider's owned native payload and shape for an admitted
/// consumer that can retain those exact backings. The stage is physically
/// settled on successful return; provider implementations must retire their
/// callback staging lease before completing this future.
pub async fn download_numeric_preserving_residency_staged_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    stage: PreparedNumericDownloadStage,
) -> crate::BuiltinResult<HostNumericTensorOwned> {
    download_numeric_preserving_residency_with_admission_async(
        provider,
        handle,
        NumericDownloadAdmission::Stage(stage),
    )
    .await
}

enum NumericDownloadAdmission {
    None,
    Callback(std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>),
    Stage(PreparedNumericDownloadStage),
}

async fn download_value_preserving_residency_with_admission_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    admission: NumericDownloadAdmission,
) -> crate::BuiltinResult<Value> {
    let host =
        download_numeric_preserving_residency_with_admission_async(provider, handle, admission)
            .await?;
    value_from_numeric_download(host, runmat_accelerate_api::handle_is_logical(handle)).map_err(
        |error| {
            build_runtime_error(format!("gpu download: {error}"))
                .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        },
    )
}

async fn download_numeric_preserving_residency_with_admission_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    admission: NumericDownloadAdmission,
) -> crate::BuiltinResult<HostNumericTensorOwned> {
    let _source_guard = ResidencyRestoreGuard::new(handle);
    if provider.device_id() != handle.device_id() {
        return Err(
            build_runtime_error("gpu download: provider does not own the input handle")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
    let expected_element = expected_handle_numeric_element_type(handle).map_err(|error| {
        build_runtime_error(format!("gpu download: {error}"))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let native = match admission {
        NumericDownloadAdmission::Callback(claim) => {
            // The provider future's own captured claim would drop before its
            // Box frame is deallocated. Keep an outer clone alive until that
            // frame drops; the provider callback retains another clone if it
            // outlives a cancelled waiter.
            let frame_funding = claim.clone();
            let future = provider.download_numeric_admitted(handle, claim);
            let result = future.await;
            drop(frame_funding);
            result
        }
        NumericDownloadAdmission::None => provider.download_numeric(handle).await,
        NumericDownloadAdmission::Stage(stage) => {
            let frame_funding = stage.completion_claim();
            let future = provider.download_numeric_admitted_stage(handle, stage);
            let result = future.await;
            drop(frame_funding);
            result
        }
    };
    finish_native_download(
        handle,
        expected_element,
        native,
        NumericDownloadDiagnostic::Gpu,
    )
}

/// The operation owning a plain provider download failure. Typed provider
/// RuntimeErrors still retain their original identity, retry and cause chain.
#[derive(Clone, Copy)]
pub(crate) enum NumericDownloadDiagnostic {
    Gpu,
    Gather,
}

/// Exact-source data egress without exposing compute/provider authority.
pub(crate) async fn download_numeric_readback_preserving_residency_async(
    readback: &runmat_accelerate_api::NumericReadback<'_>,
    stage: Option<PreparedNumericDownloadStage>,
    diagnostic: NumericDownloadDiagnostic,
) -> crate::BuiltinResult<HostNumericTensorOwned> {
    let handle = readback.source();
    let _source_guard = ResidencyRestoreGuard::new(handle);
    let expected_element = expected_handle_numeric_element_type(handle).map_err(|error| {
        build_runtime_error(format!("gpu download: {error}"))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let native = if let Some(stage) = stage {
        // Keep funding until the boxed provider future itself is dropped.
        let frame_funding = stage.completion_claim();
        let result = readback.download_numeric_admitted_stage(stage).await;
        drop(frame_funding);
        result
    } else {
        readback.download_numeric().await
    };
    finish_native_download(handle, expected_element, native, diagnostic)
}

fn finish_native_download(
    handle: &GpuTensorHandle,
    expected_element: NumericElementType,
    native: anyhow::Result<HostNumericTensorOwned>,
    diagnostic: NumericDownloadDiagnostic,
) -> crate::BuiltinResult<HostNumericTensorOwned> {
    let host = native.map_err(|error| {
        let (operation, identifier) = match diagnostic {
            NumericDownloadDiagnostic::Gpu => ("gpu download", "RunMat:gpu:DownloadFailed"),
            NumericDownloadDiagnostic::Gather => ("gather", "RunMat:gather:DownloadFailed"),
        };
        let fallback = build_runtime_error(format!("{operation}: {error}"))
            .with_identifier(identifier)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        crate::runtime_error::with_provider_execution_source(fallback, error.into())
    })?;
    if host.shape != handle.shape()
        || Some(host.storage) != handle.descriptor().storage
        || host.data.element_type() != expected_element
    {
        return Err(provider_payload_mismatch(
            handle,
            &host.shape,
            format!(
                "{:?} {:?}, expected {:?} {:?}",
                host.data.element_type(),
                host.storage,
                expected_element,
                handle.descriptor().storage
            ),
        ));
    }
    Ok(host)
}

/// Exact native scalar values decoded from a provider-owned real numeric payload.
pub struct HostNativeTensorOwned {
    pub data: Vec<NumericScalar>,
    pub shape: Vec<usize>,
}

/// Download a real numeric handle without changing its native class or source residency.
pub async fn download_native_values_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> crate::BuiltinResult<HostNativeTensorOwned> {
    let value = download_value_preserving_residency_async(provider, handle).await?;
    match value {
        Value::Tensor(tensor) => {
            // The fallible `Result<Vec<_>>` collector erases the exact lower
            // size hint and can grow from a minimum-capacity allocation even
            // for one scalar. Reserve the validated tensor length explicitly
            // so scalar probe admission can quote the actual requested slots.
            let mut data = Vec::with_capacity(tensor.len());
            for index in 0..tensor.len() {
                data.push(tensor.numeric_value_at(index).ok_or_else(|| {
                    build_runtime_error("native download: numeric payload storage is inconsistent")
                        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?);
            }
            Ok(HostNativeTensorOwned {
                data,
                shape: tensor.shape().to_vec(),
            })
        }
        Value::LogicalArray(logical) => {
            let mut data = Vec::with_capacity(logical.data.len());
            data.extend(logical.data.into_iter().map(NumericScalar::U8));
            Ok(HostNativeTensorOwned {
                data,
                shape: logical.shape.to_vec(),
            })
        }
        Value::Num(value) => Ok(HostNativeTensorOwned {
            data: vec![NumericScalar::F64(value)],
            shape: vec![1, 1],
        }),
        Value::Bool(value) => Ok(HostNativeTensorOwned {
            data: vec![NumericScalar::U8(u8::from(value))],
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(_) | Value::Complex(_, _) => Err(build_runtime_error(
            "native download: provider output must use real storage",
        )
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()),
        other => Err(build_runtime_error(format!(
            "native download: unexpected provider value {other:?}"
        ))
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()),
    }
}

/// Exact host truth values decoded from a provider-owned real numeric payload.
pub struct HostTruthTensorOwned {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
}

/// Download a real numeric handle and classify each native value as zero or
/// nonzero without entering a floating representation.
pub async fn download_truth_values_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> crate::BuiltinResult<HostTruthTensorOwned> {
    let native = download_native_values_async(provider, handle).await?;
    Ok(HostTruthTensorOwned {
        data: native
            .data
            .into_iter()
            .map(|value| u8::from(!value.is_zero()))
            .collect(),
        shape: native.shape,
    })
}

/// Project a provider-owned floating payload into binary64 host lanes for an
/// operation whose declared computation boundary is floating point.
///
/// This is intentionally not a generic gather API. It accepts physical `f64`
/// or `f32` storage, validates the complete handle/payload contract, preserves
/// source metadata and residency, and rejects every native integer class.
pub async fn download_floating_projection_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> crate::BuiltinResult<HostTensorOwned> {
    let _source_guard = ResidencyRestoreGuard::new(handle);
    if provider.device_id() != handle.device_id() {
        return Err(build_runtime_error(
            "floating projection: provider does not own the input handle",
        )
        .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    let expected_element = expected_handle_numeric_element_type(handle).map_err(|error| {
        build_runtime_error(format!("floating projection: {error}"))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let host = provider.download_numeric(handle).await.map_err(|error| {
        let fallback = build_runtime_error(format!("floating projection: {error}"))
            .with_identifier("RunMat:gpu:DownloadFailed")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        crate::runtime_error::with_provider_execution_source(fallback, error.into())
    })?;
    host.validate().map_err(|error| {
        build_runtime_error(format!("floating projection: {error}"))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    if host.shape != handle.shape()
        || Some(host.storage) != handle.descriptor().storage
        || host.data.element_type() != expected_element
    {
        return Err(provider_payload_mismatch(
            handle,
            &host.shape,
            format!(
                "{:?} {:?}, expected {:?} {:?}",
                host.data.element_type(),
                host.storage,
                expected_element,
                handle.descriptor().storage
            ),
        ));
    }
    let data = match host.data {
        HostNumericDataOwned::F64(values) => values,
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
        HostNumericDataOwned::I8(_)
        | HostNumericDataOwned::I16(_)
        | HostNumericDataOwned::I32(_)
        | HostNumericDataOwned::I64(_)
        | HostNumericDataOwned::U8(_)
        | HostNumericDataOwned::U16(_)
        | HostNumericDataOwned::U32(_)
        | HostNumericDataOwned::U64(_) => {
            return Err(build_runtime_error(
                "floating projection cannot consume native integer storage",
            )
            .with_identifier("RunMat:gpu:IntegerFloatingProjection")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
    };
    Ok(HostTensorOwned {
        data,
        shape: host.shape,
        storage: host.storage,
    })
}

pub(crate) fn provider_payload_mismatch(
    handle: &GpuTensorHandle,
    actual_shape: &[usize],
    detail: String,
) -> crate::RuntimeError {
    build_runtime_error(format!(
        "gpu download: provider payload mismatch ({detail}; shape {actual_shape:?}, expected {:?})",
        handle.shape()
    ))
    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .with_source(crate::dispatcher::ProviderDownloadPayloadMismatch)
    .build()
}

fn interleave<T: Copy>(real: &[T], imag: &[T]) -> Vec<T> {
    let mut values = Vec::with_capacity(real.len().saturating_mul(2));
    for (&real, &imag) in real.iter().zip(imag) {
        values.push(real);
        values.push(imag);
    }
    values
}

fn interleaved_complex_data(tensor: &ComplexTensor) -> Result<HostNumericDataOwned, String> {
    macro_rules! integer_interleaved {
        ($real:expr, $imag:expr, $variant:ident) => {
            HostNumericDataOwned::$variant(interleave($real, $imag))
        };
    }
    match tensor.complex_storage() {
        ComplexStorage::F64(values) => {
            complex_transfer::interleave(values).map(HostNumericDataOwned::F64)
        }
        ComplexStorage::F32(values) => {
            complex_transfer::interleave(values).map(HostNumericDataOwned::F32)
        }
        ComplexStorage::Integer(storage) => match (storage.real(), storage.imag()) {
            (IntegerStorage::I8(real), IntegerStorage::I8(imag)) => {
                Ok(integer_interleaved!(real, imag, I8))
            }
            (IntegerStorage::I16(real), IntegerStorage::I16(imag)) => {
                Ok(integer_interleaved!(real, imag, I16))
            }
            (IntegerStorage::I32(real), IntegerStorage::I32(imag)) => {
                Ok(integer_interleaved!(real, imag, I32))
            }
            (IntegerStorage::I64(real), IntegerStorage::I64(imag)) => {
                Ok(integer_interleaved!(real, imag, I64))
            }
            (IntegerStorage::U8(real), IntegerStorage::U8(imag)) => {
                Ok(integer_interleaved!(real, imag, U8))
            }
            (IntegerStorage::U16(real), IntegerStorage::U16(imag)) => {
                Ok(integer_interleaved!(real, imag, U16))
            }
            (IntegerStorage::U32(real), IntegerStorage::U32(imag)) => {
                Ok(integer_interleaved!(real, imag, U32))
            }
            (IntegerStorage::U64(real), IntegerStorage::U64(imag)) => {
                Ok(integer_interleaved!(real, imag, U64))
            }
            _ => Err("complex integer components must have matching classes".into()),
        },
    }
}

/// Upload a host complex tensor through the shared native numeric transfer contract.
pub fn upload_complex_tensor(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
) -> crate::BuiltinResult<GpuTensorHandle> {
    upload_complex_tensor_protecting(provider, tensor, &[])
}

/// Upload without adopting or freeing a provider result that aliases a borrowed input.
pub fn upload_complex_tensor_protecting(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
    protected: &[&GpuTensorHandle],
) -> crate::BuiltinResult<GpuTensorHandle> {
    let data = interleaved_complex_data(tensor)
        .map_err(|error| build_runtime_error(format!("gpu upload: {error}")).build())?;
    let transfer = HostNumericTensorView {
        data: data.as_view(),
        shape: tensor.shape(),
        storage: GpuTensorStorage::ComplexInterleaved,
    };
    runmat_accelerate_api::upload_numeric_protecting(provider, &transfer, protected).map_err(
        |error| {
            build_runtime_error(format!("gpu upload: {error}"))
                .with_source(error)
                .build()
        },
    )
}

pub(crate) fn tensor_numeric_view(tensor: &Tensor) -> HostNumericDataView<'_> {
    if let Some(values) = tensor.as_f64_slice() {
        return HostNumericDataView::F64(values);
    }
    if let Some(values) = tensor.as_f32_slice() {
        return HostNumericDataView::F32(values);
    }
    match tensor
        .integer_storage()
        .expect("non-floating tensor has integer storage")
    {
        IntegerStorage::I8(values) => HostNumericDataView::I8(values),
        IntegerStorage::I16(values) => HostNumericDataView::I16(values),
        IntegerStorage::I32(values) => HostNumericDataView::I32(values),
        IntegerStorage::I64(values) => HostNumericDataView::I64(values),
        IntegerStorage::U8(values) => HostNumericDataView::U8(values),
        IntegerStorage::U16(values) => HostNumericDataView::U16(values),
        IntegerStorage::U32(values) => HostNumericDataView::U32(values),
        IntegerStorage::U64(values) => HostNumericDataView::U64(values),
    }
}

/// Upload a host tensor through the shared native numeric transfer contract.
pub fn upload_tensor(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
) -> crate::BuiltinResult<GpuTensorHandle> {
    upload_tensor_protecting(provider, tensor, &[])
}

/// Upload without adopting or freeing a provider result that aliases a borrowed input.
pub fn upload_tensor_protecting(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
    protected: &[&GpuTensorHandle],
) -> crate::BuiltinResult<GpuTensorHandle> {
    upload_tensor_protecting_typed(provider, tensor, protected).map_err(|error| {
        build_runtime_error(format!("gpu upload: {error}"))
            .with_source(error)
            .build()
    })
}

/// Preserve provider error types for callers that distinguish unsupported
/// representations from allocation failures and malformed upload results.
pub fn upload_tensor_protecting_typed(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
    protected: &[&GpuTensorHandle],
) -> runmat_accelerate_api::NumericUploadResult<GpuTensorHandle> {
    let transfer = HostNumericTensorView {
        data: tensor_numeric_view(tensor),
        shape: tensor.shape(),
        storage: GpuTensorStorage::Real,
    };
    runmat_accelerate_api::upload_numeric_protecting(provider, &transfer, protected)
}

/// Side-effect-free admission for a real sequence's exact native-class upload.
/// A legacy preferred-precision `upload` adapter is not evidence that another
/// floating class can be published; providers must own the exact transfer query.
pub fn real_sequence_upload_feasible(
    provider: &dyn AccelProvider,
    precision: ProviderPrecision,
    shape: &[usize],
) -> bool {
    let Some(shape) = shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()
    else {
        return false;
    };
    let representation = runmat_accelerate_api::ProviderRepresentation {
        element_type: match precision {
            ProviderPrecision::F32 => runmat_accelerate_api::ProviderElementType::F32,
            ProviderPrecision::F64 => runmat_accelerate_api::ProviderElementType::F64,
        },
        storage: runmat_accelerate_api::ProviderStorage::DenseReal,
        layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: runmat_accelerate_api::ProviderResidency::Device,
    };
    runmat_accelerate_api::class_preserving_upload_query(vec![representation])
        .is_some_and(|query| provider.query_feasibility(&query).is_supported())
}

mod restore;
pub(crate) use restore::restore_real_scalar_after_host;
pub use restore::{
    restore_class_preserving_value, restore_class_preserving_value_protecting,
    restore_class_preserving_value_with_operands, RestoreOperandContext,
};

/// Upload a finite integral scalar in the native integer class of `prototype`.
/// Returns `None` when preserving typed-integer scalar semantics would require
/// the host extended-precision path instead. Provider failures remain errors so
/// callers cannot mistake an unavailable device for an inapplicable fast path.
pub fn upload_exact_integer_scalar_like(
    provider: &dyn AccelProvider,
    prototype: &GpuTensorHandle,
    scalar: f64,
) -> runmat_accelerate_api::NumericUploadResult<Option<GpuTensorHandle>> {
    if !scalar.is_finite() || scalar.fract() != 0.0 {
        return Ok(None);
    }
    let Some(element_type) = runmat_accelerate_api::handle_integer_type(prototype) else {
        return Ok(None);
    };
    let shape = [1usize, 1usize];
    macro_rules! upload {
        ($value:expr, $variant:ident) => {{
            let values = [$value];
            runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::$variant(&values),
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                },
                &[prototype],
            )
            .map(Some)
        }};
    }
    match element_type {
        IntegerElementType::I8 if scalar >= i8::MIN as f64 && scalar <= i8::MAX as f64 => {
            upload!(scalar as i8, I8)
        }
        IntegerElementType::I16 if scalar >= i16::MIN as f64 && scalar <= i16::MAX as f64 => {
            upload!(scalar as i16, I16)
        }
        IntegerElementType::I32 if scalar >= i32::MIN as f64 && scalar <= i32::MAX as f64 => {
            upload!(scalar as i32, I32)
        }
        IntegerElementType::I64
            if scalar >= i64::MIN as f64 && scalar < 9_223_372_036_854_775_808.0 =>
        {
            upload!(scalar as i64, I64)
        }
        IntegerElementType::U8 if scalar >= 0.0 && scalar <= u8::MAX as f64 => {
            upload!(scalar as u8, U8)
        }
        IntegerElementType::U16 if scalar >= 0.0 && scalar <= u16::MAX as f64 => {
            upload!(scalar as u16, U16)
        }
        IntegerElementType::U32 if scalar >= 0.0 && scalar <= u32::MAX as f64 => {
            upload!(scalar as u32, U32)
        }
        IntegerElementType::U64 if (0.0..18_446_744_073_709_551_616.0).contains(&scalar) => {
            upload!(scalar as u64, U64)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
#[path = "gpu_helpers/preserving_download_tests.rs"]
mod preserving_download_tests;

#[cfg(test)]
#[path = "gpu_helpers/output_admission_tests.rs"]
mod output_admission_tests;

#[cfg(all(test, feature = "wgpu"))]
#[path = "gpu_helpers/integer_upload_wgpu_tests.rs"]
mod tests;
