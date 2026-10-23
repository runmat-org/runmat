//! Explicit request-local admission at the existing typed transfer boundary.
//! Registration owns retained metadata; native encoding is temporary storage.
use super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, provider_registration_host_quote, HostAllocationError,
    HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::indexing::plan::admission::{Owner, Scratch};
use runmat_accelerate_api::{HostNumericTensorView, HostNumericUploadScratch};
use runmat_execution::resource::ProviderHostMetadataAdmission;

fn allocation(context: Option<&RuntimeContext>, cause: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:Indexing:UploadAllocation",
        "resident upload ownership",
        cause,
    )
}

// The prepared permit retires before its preparation receipt. Published
// records/views retain their own original claims independently of this owner.
struct Registration {
    permit: Option<Box<dyn ProviderHostMetadataAdmission>>,
    _receipt: Option<HostAllocationReceipt>,
}
impl Registration {
    fn prepare(
        context: Option<&RuntimeContext>,
        provider: &dyn AccelProvider,
        rank: usize,
        temporary: u64,
    ) -> Result<Self, RuntimeError> {
        if HostAllocationReceipt::policy(context).map_err(|e| allocation(context, e))?
            == HostAllocationPolicy::Unbounded
        {
            return Ok(Self {
                permit: None,
                _receipt: None,
            });
        }
        let runtime =
            context.ok_or_else(|| allocation(context, HostAllocationError::MissingManagedOwner))?;
        let quote = provider_registration_host_quote(provider, rank, 1, 1)
            .map_err(|e| allocation(context, e))?
            .ok_or_else(|| {
                failure(
                    "resident upload registration",
                    runmat_accelerate_api::unsupported_provider_operation(
                        "provider has no finite indexing registration owner",
                    )
                    .into(),
                )
            })?;
        let preparation = quote
            .preparation_bytes
            .checked_add(temporary)
            .ok_or_else(|| allocation(context, HostAllocationError::Overflow))?;
        let receipt = HostAllocationReceipt::admit(
            context,
            HostAllocationPlan::new(preparation, quote.retained_bytes)
                .map_err(|e| allocation(context, e))?,
        )
        .map_err(|e| allocation(context, e))?
        .ok_or_else(|| allocation(context, HostAllocationError::MissingManagedOwner))?;
        let retained = receipt
            .retained_view_permit(runtime)
            .map_err(|e| allocation(context, e))?;
        let permit = provider
            .prepare_output_host_metadata(Box::new(retained), 1)
            .map_err(|cause| failure("resident upload registration", cause.into()))?;
        Ok(Self {
            permit: Some(permit),
            _receipt: Some(receipt),
        })
    }
}

pub(super) fn numeric(
    context: Option<&RuntimeContext>,
    provider: &dyn AccelProvider,
    view: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, RuntimeError> {
    let owner = Owner::new(context)?;
    // Validate source geometry before preparing any registration or scratch.
    view.validate()
        .map_err(|cause| failure("indexed result", cause.into()))?;
    let registration = Registration::prepare(context, provider, view.shape.len(), 0)?;
    let mut scratch = if registration.permit.is_some() {
        let words = provider
            .numeric_upload_encoding_words(view.data.element_type(), view.data.len())
            .ok_or_else(|| {
                failure(
                    "resident upload encoding",
                    runmat_accelerate_api::unsupported_provider_operation(
                        "provider has no finite native upload encoding bound",
                    )
                    .into(),
                )
            })?;
        if words == 0 {
            None
        } else {
            Some(Scratch::<u32>::new(context, words)?)
        }
    } else {
        None
    };
    owner.check()?;
    let output = runmat_accelerate_api::upload_numeric_protecting_admitted_with_scratch(
        provider,
        view,
        protected,
        registration.permit.as_deref(),
        scratch.as_mut().map(|scratch| HostNumericUploadScratch {
            words: scratch.as_mut_slice(),
        }),
    )
    .map_err(|cause| failure("indexed result", cause.into_execution_error()))?;
    if let Err(cause) = owner.check() {
        return Err(gpu_helpers::reject_gpu_output(output, cause));
    }
    Ok(output)
}

fn indexing_workspace(
    context: Option<&RuntimeContext>,
    provider: &dyn AccelProvider,
    source: &GpuTensorHandle,
    rhs: Option<&GpuTensorHandle>,
) -> Result<u64, RuntimeError> {
    if HostAllocationReceipt::policy(context).map_err(|e| allocation(context, e))?
        == HostAllocationPolicy::Unbounded
    {
        return Ok(0);
    }
    match provider.indexing_host_workspace_bound(source, rhs) {
        runmat_execution::ResourceByteBound::Known(bytes) => Ok(bytes),
        runmat_execution::ResourceByteBound::Unavailable(_) => Err(failure(
            "resident indexing metadata",
            runmat_accelerate_api::unsupported_provider_operation(
                "provider has no finite indexing temporary metadata bound",
            )
            .into(),
        )),
    }
}

pub(super) fn gather(
    context: Option<&RuntimeContext>,
    source: &GpuTensorHandle,
    indices: &[u32],
    shape: &[usize],
) -> Result<GpuTensorHandle, RuntimeError> {
    let owner = Owner::new(context)?;
    let provider = super::owner(source)?;
    let workspace = indexing_workspace(context, provider, source, None)?;
    let registration = Registration::prepare(context, provider, shape.len(), workspace)?;
    let result = match registration.permit.as_deref() {
        Some(permit) => provider.gather_linear_admitted(source, indices, shape, permit),
        None => provider
            .gather_linear(source, indices, shape)
            .map_err(Into::into),
    }
    .map_err(|cause| failure("indexed gather", cause))?;
    if let Err(cause) = owner.check() {
        return Err(gpu_helpers::reject_gpu_output(result, cause));
    }
    Ok(result)
}

pub(super) async fn scatter(
    context: Option<&RuntimeContext>,
    target: &GpuTensorHandle,
    indices: &[u32],
    values: &GpuTensorHandle,
    mapping: runmat_accelerate_api::ProviderScatterValueMapping,
) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
    let run = async {
        let owner = Owner::new(context)?;
        let provider = super::owner(target)?;
        let workspace = indexing_workspace(context, provider, target, Some(values))?;
        let registration =
            Registration::prepare(context, provider, target.shape().len(), workspace)?;
        let result = match registration.permit.as_deref() {
            Some(permit) => provider
                .scatter_linear_functional_admitted(target, indices, values, mapping, permit),
            None if mapping == runmat_accelerate_api::ProviderScatterValueMapping::Linear => {
                owner
                    .await_stage(provider.scatter_linear_functional(target, indices, values))
                    .await?
            }
            None => Err(runmat_accelerate_api::unsupported_provider_operation(
                "ordinary functional scatter requires expanded values",
            )
            .into()),
        }
        .map_err(|cause| failure("functional indexed scatter", cause))?;
        if let Err(cause) = owner.check() {
            return Err(gpu_helpers::reject_gpu_output(result, cause));
        }
        Ok(result)
    };
    run.await.map_err(ProviderExecutionError::from_source)
}
