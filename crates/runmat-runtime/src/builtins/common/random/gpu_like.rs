//! Semantic intent of a resident `Like` prototype, shared by random constructors.
//! Automatic residency is not an instruction to draw from the independent GPU
//! stream. The caller still owns its output-class contract and presentation.

use crate::builtins::common::gpu_helpers::host_upload::GpuHostUpload;
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use crate::context::host_presentation::{
    select_host_computed_result, ExactHostComputedResult, ExactHostComputedUpload,
    HostComputedSelection, HostSinkReceipt,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostNumericDataView,
    HostNumericTensorView, NumericElementType, ProviderPrecision,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostLogicalTensorAllocation, ExactHostTensorAllocation,
    IntValue, NumericDType, Value,
};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[path = "gpu_like/workspace.rs"]
mod workspace;
pub(crate) use workspace::{
    exact_host_workspace, host_random_mode, ExactHostRandomMode, ExactHostRandomWorkspace,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum GpuLikeClass {
    Floating {
        precision: ProviderPrecision,
        complex: bool,
    },
    Integer {
        class: runmat_types::IntegerClass,
        complex: bool,
    },
    Logical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuLikeIntent {
    AutomaticHostStream,
    ExplicitGpuStream,
}

pub(crate) enum GpuLikePresentation<'a> {
    Host(
        Option<ClaimedBox<HostSinkReceipt>>,
        ExactHostRandomMode,
        Option<crate::context::RuntimeContext>,
    ),
    Upload(
        ClaimedBox<GpuHostUpload<'a>>,
        ExactHostRandomMode,
        Option<crate::context::RuntimeContext>,
    ),
}

/// The selected route's box is admitted before allocation and remains charged
/// until its payload is retired. The claim drops after the box.
pub(crate) struct ClaimedBox<T> {
    value: Box<T>,
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

impl<T> ClaimedBox<T> {
    fn requested_bytes(label: &'static str) -> BuiltinResult<u64> {
        let payload = u64::try_from(std::mem::size_of::<T>()).map_err(|source| {
            build_runtime_error(format!("{label}: selected route box size overflow"))
                .with_builtin(label)
                .with_source(source)
                .build()
        })?;
        HostAllocationPlan::retained_metadata_bytes(payload).map_err(|source| {
            build_runtime_error(format!("{label}: selected route box quote failed"))
                .with_builtin(label)
                .with_source(source)
                .build()
        })
    }

    fn new(
        label: &'static str,
        runtime: Option<&crate::context::RuntimeContext>,
        value: T,
    ) -> BuiltinResult<Self> {
        let payload = u64::try_from(std::mem::size_of::<T>()).map_err(|source| {
            build_runtime_error(format!("{label}: selected route box size overflow"))
                .with_builtin(label)
                .with_source(source)
                .build()
        })?;
        let plan = HostAllocationPlan::new(0, Self::requested_bytes(label)?).map_err(|source| {
            build_runtime_error(format!("{label}: selected route box plan failed"))
                .with_builtin(label)
                .with_source(source)
                .build()
        })?;
        let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(|source| {
            build_runtime_error(format!("{label}: selected route box admission failed"))
                .with_builtin(label)
                .with_source(source)
                .build()
        })?;
        let claim = receipt
            .as_ref()
            .map(|receipt| {
                receipt
                    .claim_retained_metadata(runtime.expect("managed receipt has runtime"), payload)
            })
            .transpose()
            .map_err(|source| {
                build_runtime_error(format!("{label}: selected route box claim failed"))
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            })?;
        Ok(Self {
            value: Box::new(value),
            _claim: claim,
        })
    }

    fn into_inner(self) -> T {
        *self.value
    }
}

impl<T> Deref for ClaimedBox<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for ClaimedBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl GpuLikeClass {
    pub(crate) const fn native_storage(self) -> GpuTensorStorage {
        match self {
            Self::Floating { complex: true, .. } | Self::Integer { complex: true, .. } => {
                GpuTensorStorage::ComplexInterleaved
            }
            Self::Floating { complex: false, .. }
            | Self::Integer { complex: false, .. }
            | Self::Logical => GpuTensorStorage::Real,
        }
    }

    pub(crate) const fn native_element(self) -> Option<NumericElementType> {
        match self {
            Self::Floating {
                precision: ProviderPrecision::F64,
                complex: false,
            } => Some(NumericElementType::F64),
            Self::Floating {
                precision: ProviderPrecision::F32,
                complex: false,
            } => Some(NumericElementType::F32),
            Self::Floating {
                precision: ProviderPrecision::F64,
                complex: true,
            } => Some(NumericElementType::F64),
            Self::Floating {
                precision: ProviderPrecision::F32,
                complex: true,
            } => Some(NumericElementType::F32),
            Self::Integer {
                class: runmat_types::IntegerClass::Int8,
                complex: false,
            } => Some(NumericElementType::I8),
            Self::Integer {
                class: runmat_types::IntegerClass::Int16,
                complex: false,
            } => Some(NumericElementType::I16),
            Self::Integer {
                class: runmat_types::IntegerClass::Int32,
                complex: false,
            } => Some(NumericElementType::I32),
            Self::Integer {
                class: runmat_types::IntegerClass::Int64,
                complex: false,
            } => Some(NumericElementType::I64),
            Self::Integer {
                class: runmat_types::IntegerClass::UInt8,
                complex: false,
            } => Some(NumericElementType::U8),
            Self::Integer {
                class: runmat_types::IntegerClass::UInt16,
                complex: false,
            } => Some(NumericElementType::U16),
            Self::Integer {
                class: runmat_types::IntegerClass::UInt32,
                complex: false,
            } => Some(NumericElementType::U32),
            Self::Integer {
                class: runmat_types::IntegerClass::UInt64,
                complex: false,
            } => Some(NumericElementType::U64),
            Self::Integer { complex: true, .. } | Self::Logical => None,
        }
    }
}

/// Match the output's already selected allocation mode: a managed tensor has
/// two independent retained claims, while an unbounded tensor owns the same
/// numeric and shape backing without claim controls.
pub(crate) fn host_result_bytes(
    label: &'static str,
    shape: &[usize],
    class: GpuLikeClass,
    mode: ExactHostRandomMode,
) -> BuiltinResult<ResourceByteBound> {
    let dtype = match class {
        GpuLikeClass::Floating { precision, .. } => Some(if precision == ProviderPrecision::F32 {
            NumericDType::F32
        } else {
            NumericDType::F64
        }),
        GpuLikeClass::Integer {
            class,
            complex: false,
        } => Some(NumericDType::from(class)),
        GpuLikeClass::Logical => None,
        GpuLikeClass::Integer { complex: true, .. } => {
            return Err(
                build_runtime_error("random integer-complex host output is unsupported")
                    .with_builtin(label)
                    .build(),
            );
        }
    };
    let bytes = match mode {
        ExactHostRandomMode::Managed => match class {
            GpuLikeClass::Logical => HostAllocationPlan::retained_logical_tensor_bytes(shape),
            GpuLikeClass::Floating { complex, .. } => {
                let dtype = dtype.expect("floating output has a numeric dtype");
                if complex {
                    HostAllocationPlan::retained_complex_tensor_bytes(dtype, shape)
                } else {
                    HostAllocationPlan::retained_tensor_bytes(dtype, shape)
                }
            }
            GpuLikeClass::Integer { .. } => HostAllocationPlan::retained_tensor_bytes(
                dtype.expect("real integer output has a numeric dtype"),
                shape,
            ),
        }
        .ok(),
        ExactHostRandomMode::Unbounded => {
            let requested = match class {
                GpuLikeClass::Logical => {
                    ExactHostLogicalTensorAllocation::unmanaged_requested_bytes(shape)
                }
                GpuLikeClass::Floating { complex, .. } => {
                    let dtype = dtype.expect("floating output has a numeric dtype");
                    if complex {
                        ExactHostComplexTensorAllocation::unmanaged_requested_bytes(dtype, shape)
                    } else {
                        ExactHostTensorAllocation::unmanaged_requested_bytes(dtype, shape)
                    }
                }
                GpuLikeClass::Integer { .. } => {
                    ExactHostTensorAllocation::unmanaged_requested_bytes(
                        dtype.expect("real integer output has a numeric dtype"),
                        shape,
                    )
                }
            }
            .ok();
            requested.and_then(|bytes| u64::try_from(bytes).ok())
        }
    }
    .ok_or_else(|| {
        build_runtime_error("random host result extent overflow")
            .with_builtin(label)
            .build()
    })?;
    Ok(ResourceByteBound::Known(bytes))
}

pub(crate) const fn unquoted_host_scratch() -> ResourceByteBound {
    ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
}

pub(crate) fn complex_upload_workspace_bytes(
    label: &'static str,
    shape: &[usize],
    precision: ProviderPrecision,
) -> BuiltinResult<ResourceByteBound> {
    let len = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| {
            build_runtime_error("complex upload extent overflow")
                .with_builtin(label)
                .build()
        })?;
    let dtype = if precision == ProviderPrecision::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let bytes =
        crate::context::host_allocation::HostAllocationPlan::numeric_workspace_bytes(dtype, len)
            .map_err(|source| {
                build_runtime_error("complex upload workspace quote failed")
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            })?;
    Ok(ResourceByteBound::Known(bytes))
}

/// Decide presentation before acquiring the ordered CPU random turn. Each
/// native class needs its own physical upload evidence; placement never
/// changes which random stream is drawn.
pub(crate) fn select_presentation<'a>(
    label: &'static str,
    source: &'a GpuTensorHandle,
    shape: &'a [usize],
    output_class: GpuLikeClass,
    host_result_bytes: ResourceByteBound,
    host_scratch: ResourceByteBound,
    upload_scratch: ResourceByteBound,
    upload_owned_scratch: u64,
    host_mode: ExactHostRandomMode,
) -> BuiltinResult<GpuLikePresentation<'a>> {
    let runtime = crate::context::legacy::active();
    if host_mode != host_random_mode(runtime.as_ref()) {
        return Err(
            build_runtime_error("random host allocation mode changed before selection")
                .with_identifier("RunMat:Placement:StaleResourceAuthority")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
    if runtime
        .as_ref()
        .and_then(|runtime| runtime.service_ports().placement())
        .is_none()
    {
        return Ok(GpuLikePresentation::Host(None, host_mode, runtime));
    }
    let host_box_bytes = ClaimedBox::<HostSinkReceipt>::requested_bytes(label)?;
    let upload_box_bytes = ClaimedBox::<GpuHostUpload<'a>>::requested_bytes(label)?;
    let host_scratch = add_route_box_quote(label, host_scratch, host_box_bytes)?;
    let upload_scratch = add_route_box_quote(label, upload_scratch, upload_box_bytes)?;
    let quote = match (upload_scratch, host_result_bytes) {
        (ResourceByteBound::Known(scratch), ResourceByteBound::Known(retained)) => {
            let output_storage = output_class.native_storage();
            let lanes = if output_storage == GpuTensorStorage::ComplexInterleaved {
                2_u64
            } else {
                1_u64
            };
            let payload = shape
                .iter()
                .try_fold(1_u64, |count, &dim| {
                    count.checked_mul(u64::try_from(dim).ok()?)
                })
                .and_then(|count| count.checked_mul(lanes))
                .and_then(|count| {
                    count.checked_mul(
                        output_class
                            .native_element()
                            .map_or(1, |class| class.element_size()) as u64,
                    )
                })
                .ok_or_else(|| {
                    build_runtime_error("random upload payload extent overflow")
                        .with_builtin(label)
                        .build()
                })?;
            let metadata = retained.checked_sub(payload).ok_or_else(|| {
                build_runtime_error("random upload output metadata quote underflow")
                    .with_builtin(label)
                    .build()
            })?;
            let scratch = scratch.checked_add(metadata).ok_or_else(|| {
                build_runtime_error("random upload host peak overflow")
                    .with_builtin(label)
                    .build()
            })?;
            if output_class == GpuLikeClass::Logical {
                GpuHostUpload::quote_logical_candidate(label, source, shape, scratch)?
            } else if let Some(class) = output_class.native_element() {
                GpuHostUpload::quote_candidate_with_storage(
                    label,
                    source,
                    shape,
                    class,
                    output_storage,
                    scratch,
                )?
            } else {
                None
            }
        }
        _ => None,
    };
    let selection = select_host_computed_result(
        runtime.as_ref(),
        ExactHostComputedResult {
            operation: label,
            variant: "automatic-like-host-random",
            source,
            domain_facts: Digest::sha256_serialized(&(shape, output_class))
                .expect("typed random output serializes"),
            host_result_bytes,
            host_compute_scratch: host_scratch,
            upload: quote.as_ref().map(|quote| ExactHostComputedUpload {
                candidate: &quote.candidate,
                query: &quote.query,
                output: &quote.output,
                requested_bytes: quote.requested_bytes,
            }),
        },
    )?;
    match selection {
        HostComputedSelection::Host(receipt) => Ok(GpuLikePresentation::Host(
            receipt
                .map(|receipt| ClaimedBox::new(label, runtime.as_ref(), receipt))
                .transpose()?,
            host_mode,
            runtime,
        )),
        HostComputedSelection::Upload(selection) => {
            let quoted_bytes = quote
                .as_ref()
                .expect("selected upload has quote")
                .requested_bytes;
            let scratch = match upload_scratch {
                ResourceByteBound::Known(_) => upload_owned_scratch,
                ResourceByteBound::Unavailable(_) => {
                    selection.finish(Err(
                        runmat_accelerate_api::observation::ProviderObservationFailure::ResourceLimit,
                    ));
                    return Err(build_runtime_error(
                        "selected random upload has no host scratch bound",
                    )
                    .with_builtin(label)
                    .build());
                }
            };
            let prepared_result = match output_class {
                GpuLikeClass::Logical => {
                    GpuHostUpload::prepare_borrowed_logical_host_output(label, source, shape)
                }
                _ if output_class.native_storage() == GpuTensorStorage::ComplexInterleaved => {
                    GpuHostUpload::prepare_borrowed_complex_host_output(
                        label,
                        source,
                        shape,
                        output_class
                            .native_element()
                            .expect("selected complex class"),
                        scratch,
                    )
                }
                _ => GpuHostUpload::prepare_borrowed_host_output(
                    label,
                    source,
                    shape,
                    output_class
                        .native_element()
                        .expect("selected numeric class"),
                    scratch,
                ),
            };
            let prepared = match prepared_result {
                Ok(prepared) => prepared,
                Err(cause) => {
                    selection.finish(Err(
                        runmat_accelerate_api::observation::ProviderObservationFailure::ResourceLimit,
                    ));
                    return Err(cause);
                }
            };
            Ok(GpuLikePresentation::Upload(
                ClaimedBox::new(
                    label,
                    runtime.as_ref(),
                    prepared.with_selected_host_route(selection, quoted_bytes)?,
                )?,
                host_mode,
                runtime,
            ))
        }
    }
}

fn add_route_box_quote(
    label: &'static str,
    bound: ResourceByteBound,
    requested: u64,
) -> BuiltinResult<ResourceByteBound> {
    match bound {
        ResourceByteBound::Known(bytes) => bytes
            .checked_add(requested)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                build_runtime_error(format!("{label}: selected route box quote overflow"))
                    .with_builtin(label)
                    .build()
            }),
        ResourceByteBound::Unavailable(cause) => Ok(ResourceByteBound::Unavailable(cause)),
    }
}

impl GpuLikePresentation<'_> {
    pub(crate) fn mode(&self) -> ExactHostRandomMode {
        match self {
            Self::Host(_, mode, _) | Self::Upload(_, mode, _) => *mode,
        }
    }

    pub(crate) fn runtime(&self) -> Option<&crate::context::RuntimeContext> {
        match self {
            Self::Host(_, _, runtime) | Self::Upload(_, _, runtime) => runtime.as_ref(),
        }
    }

    pub(crate) fn checkpoint(&self) -> BuiltinResult<()> {
        if self.mode() != host_random_mode(self.runtime()) {
            return Err(
                build_runtime_error("random host allocation mode changed after selection")
                    .with_identifier("RunMat:Placement:StaleResourceAuthority")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
            );
        }
        match self {
            Self::Host(_, _, _) => Ok(()),
            Self::Upload(upload, _, _) => upload.checkpoint(),
        }
    }

    pub(crate) fn checkpoint_for_turn(&self, turn: &super::RandomTurn) -> BuiltinResult<()> {
        self.checkpoint()?;
        let same_owner = match (self.runtime(), turn.owner()) {
            (Some(selected), Some(actual)) => selected.local_identity() == actual.local_identity(),
            (None, None) => true,
            _ => false,
        };
        if same_owner {
            Ok(())
        } else {
            Err(
                build_runtime_error("random turn owner changed after placement selection")
                    .with_identifier("RunMat:Placement:StaleResourceAuthority")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
            )
        }
    }

    pub(crate) fn finish_result(
        self,
        label: &'static str,
        result: BuiltinResult<Value>,
        shape: &[usize],
    ) -> BuiltinResult<Value> {
        match result {
            Ok(value) => self
                .publish(value, shape)
                .map_err(|source| super::gpu_effect::after_stream_commit(label, source)),
            Err(source) => {
                if let Self::Host(Some(receipt), _, _) = self {
                    receipt.into_inner().finish(false);
                }
                // A cooperative generator can fail before or after consuming
                // its reserved CPU range. The generic result carrier cannot
                // prove which; keep the original typed cause but never certify
                // a pre-effect replay from this point.
                Err(crate::runtime_error::with_provider_execution_source(
                    build_runtime_error(source.message.clone())
                        .with_builtin(label)
                        .build(),
                    runmat_accelerate_api::ProviderExecutionError::from_source(source)
                        .with_unproven_effects(),
                ))
            }
        }
    }

    pub(crate) fn publish(self, value: Value, shape: &[usize]) -> BuiltinResult<Value> {
        match self {
            Self::Host(receipt, _, _) => {
                if let Some(receipt) = receipt {
                    receipt.into_inner().finish(true);
                }
                Ok(value)
            }
            Self::Upload(mut upload, _, _) => {
                let output = match &value {
                    Value::Tensor(tensor) => upload.publish(&HostNumericTensorView {
                        data: crate::builtins::common::gpu_helpers::tensor_numeric_view(tensor),
                        shape,
                        storage: GpuTensorStorage::Real,
                    })?,
                    Value::Num(number) => upload.publish(&HostNumericTensorView {
                        data: HostNumericDataView::F64(std::slice::from_ref(number)),
                        shape,
                        storage: GpuTensorStorage::Real,
                    })?,
                    Value::Complex(real, imaginary) => {
                        let lanes = [*real, *imaginary];
                        upload.publish(&HostNumericTensorView {
                            data: HostNumericDataView::F64(&lanes),
                            shape,
                            storage: GpuTensorStorage::ComplexInterleaved,
                        })?
                    }
                    Value::ComplexTensor(tensor) => upload.publish_complex(tensor, shape)?,
                    Value::LogicalArray(array) => {
                        upload.publish_logical(&runmat_accelerate_api::HostLogicalTensorView {
                            data: &array.data,
                            shape,
                        })?
                    }
                    Value::Bool(value) => {
                        upload.publish_logical(&runmat_accelerate_api::HostLogicalTensorView {
                            data: &[u8::from(*value)],
                            shape,
                        })?
                    }
                    Value::Int(IntValue::I8(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::I8(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::I16(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::I16(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::I32(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::I32(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::I64(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::I64(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::U8(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::U8(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::U16(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::U16(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::U32(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::U32(std::slice::from_ref(number)),
                        shape,
                    )?,
                    Value::Int(IntValue::U64(number)) => upload_scalar(
                        &mut upload,
                        HostNumericDataView::U64(std::slice::from_ref(number)),
                        shape,
                    )?,
                    _ => {
                        return Err(build_runtime_error(
                            "selected random upload received non-real host result",
                        )
                        .build())
                    }
                };
                runmat_accelerate_api::mark_residency(&output);
                Ok(Value::GpuTensor(output))
            }
        }
    }
}

fn upload_scalar(
    upload: &mut GpuHostUpload<'_>,
    data: HostNumericDataView<'_>,
    shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    upload.publish(&HostNumericTensorView {
        data,
        shape,
        storage: GpuTensorStorage::Real,
    })
}

pub(crate) fn class(prototype: &GpuTensorHandle) -> Option<GpuLikeClass> {
    let complex = runmat_accelerate_api::handle_storage(prototype)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
    if runmat_accelerate_api::handle_is_logical(prototype) {
        return (!complex).then_some(GpuLikeClass::Logical);
    }
    if let Some(class) = runmat_accelerate_api::handle_integer_class(prototype) {
        return Some(GpuLikeClass::Integer { class, complex });
    }
    runmat_accelerate_api::handle_precision(prototype)
        .map(|precision| GpuLikeClass::Floating { precision, complex })
}

pub(crate) fn intent(
    label: &'static str,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<GpuLikeIntent> {
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(prototype)
        .ok_or_else(|| {
            build_runtime_error(format!(
                "{label}: no live provider owns the gpuArray prototype"
            ))
            .with_builtin(label)
            .build()
        })?;
    prototype
        .allocation_lease()
        .validate_live()
        .map_err(|source| {
            build_runtime_error(format!("{label}: gpuArray prototype is not live"))
                .with_builtin(label)
                .with_source(source)
                .build()
        })?;
    if owner.device_id() != prototype.device_id() {
        return Err(
            build_runtime_error(format!("{label}: gpuArray prototype owner changed"))
                .with_builtin(label)
                .build(),
        );
    }
    match runmat_accelerate_api::handle_provenance(prototype) {
        Some(GpuHandleProvenance::Automatic) => Ok(GpuLikeIntent::AutomaticHostStream),
        Some(GpuHandleProvenance::Explicit) => Ok(GpuLikeIntent::ExplicitGpuStream),
        None => Err(build_runtime_error(format!(
            "{label}: gpuArray prototype has no authenticated residency intent"
        ))
        .with_builtin(label)
        .build()),
    }
}

#[cfg(test)]
#[path = "gpu_like/host_result_quote_tests.rs"]
mod host_result_quote_tests;
