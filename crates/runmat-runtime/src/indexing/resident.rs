//! Owning resident indexing boundaries shared by reads and mutations.
use crate::builtins::common::gpu_helpers;
use crate::{runtime_error::semantic_error, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderExecutionError};

mod transfer;

pub(super) fn owner(handle: &GpuTensorHandle) -> Result<&'static dyn AccelProvider, RuntimeError> {
    gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        semantic_error(
            "AccelerationProviderUnavailable",
            "No acceleration provider owns the indexed gpuArray",
        )
    })
}

pub(super) fn failure(context: &str, source: ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        semantic_error(
            "AccelerationOperationFailed",
            format!("{context}: {source}"),
        ),
        source,
    )
}

pub(super) async fn functional_scatter_in_context(
    context: Option<&crate::context::RuntimeContext>,
    target: &GpuTensorHandle,
    indices: &[u32],
    values: &GpuTensorHandle,
    mapping: runmat_accelerate_api::ProviderScatterValueMapping,
) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
    transfer::scatter(context, target, indices, values, mapping).await
}

pub(super) fn read_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    input: &GpuTensorHandle,
    plan: &crate::indexing::plan::IndexPlan,
) -> Result<runmat_value::Value, RuntimeError> {
    let authority = crate::indexing::plan::admission::Owner::new(context)?;
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(input);
    if plan.indices.is_empty() {
        return empty_read_in_context(context, input, &plan.output_shape);
    }
    let output = transfer::gather(context, input, &plan.indices, &plan.output_shape)?;
    if let Err(cause) = authority.check() {
        return Err(gpu_helpers::reject_gpu_output(output, cause));
    }
    admit_read(output, input, &plan.output_shape)
}

pub(super) fn expand_scalar_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source: &GpuTensorHandle,
    plan: &crate::indexing::plan::IndexPlan,
) -> Result<GpuTensorHandle, RuntimeError> {
    // Only after the typed scalar port declines: legacy providers require a
    // full values handle. Admit its actual native selector backing.
    let indices =
        crate::indexing::plan::admission::Scratch::<u32>::new(context, plan.indices.len())?;
    let output = transfer::gather(context, source, indices.as_slice(), &plan.output_shape)?;
    match admit_read(output, source, &plan.output_shape)? {
        runmat_value::Value::GpuTensor(output) => Ok(output),
        _ => unreachable!("resident scalar expansion"),
    }
}

pub(super) async fn gather_in_context(
    context: Option<&crate::context::RuntimeContext>,
    input: &GpuTensorHandle,
) -> Result<runmat_value::Value, RuntimeError> {
    let provider = owner(input)?;
    // Keep physical transfer work out of every host scalar-assignment future.
    // This boundary is reached only for an actual resident operand. Its one
    // concrete child allocation is admitted before constructing the download.
    let result = crate::execution::rooted_continuation::admit_rooted_future_with(
        context.cloned(),
        "resident indexing transfer",
        crate::builtin::builtin_control_error,
        move |context| async move {
            let value = gpu_helpers::download_value_preserving_residency_in_context(
                context.as_ref(),
                provider,
                input,
                "RunMat:Indexing:Ownership",
                "resident indexing transfer admission failed",
            )
            .await?;
            crate::execution::RootedValue::in_optional_context(value, context.as_ref())
        },
    )
    .await?;
    Ok(result.with_owned(|value| value))
}

/// Preserve dispatcher gather semantics for resident selectors and assignment
/// operands without embedding its recursive future in every host indexing
/// frame. Callers enter this boundary only when a resident operand is present.
pub(super) async fn gather_dispatch_value_in_context(
    value: &runmat_value::Value,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<runmat_value::Value, RuntimeError> {
    let result = crate::execution::rooted_continuation::admit_rooted_future_with(
        context.cloned(),
        "resident indexing gather",
        crate::builtin::builtin_control_error,
        move |context| async move {
            let value =
                crate::dispatcher::gather_if_needed_in_context_async(value, context.as_ref())
                    .await?;
            crate::execution::RootedValue::in_optional_context(value, context.as_ref())
        },
    )
    .await?;
    Ok(result.with_owned(|value| value))
}

pub(super) async fn gather_numeric_in_context(
    context: Option<&crate::context::RuntimeContext>,
    input: &GpuTensorHandle,
) -> Result<runmat_value::Value, RuntimeError> {
    match gather_in_context(context, input).await? {
        runmat_value::Value::LogicalArray(array) => {
            // The legacy scalar numeric grammar needs a numeric working
            // carrier. Planned mutation above retains Logical directly. Fund
            // this actual conversion owner rather than an ordinary f64 mirror.
            let owner = crate::indexing::plan::admission::Owner::new(context)?;
            let shape =
                crate::indexing::plan::admission::shape(context, array.shape.len(), |axis| {
                    Ok(array.shape[axis])
                })?;
            crate::indexing::read_slice::payload::output::real(
                context,
                runmat_value::NumericDType::F64,
                shape,
                |output| {
                    for (index, &byte) in array.data.iter().enumerate() {
                        if index % 4096 == 0 {
                            owner.check()?;
                        }
                        output.set(
                            index,
                            runmat_value::NumericScalar::F64(if byte != 0 { 1.0 } else { 0.0 }),
                        )?;
                    }
                    Ok(())
                },
            )
            .map(runmat_value::Value::Tensor)
        }
        value => Ok(value),
    }
}

pub(super) fn restore_in_context(
    context: Option<&crate::context::RuntimeContext>,
    input: &GpuTensorHandle,
    value: runmat_value::Value,
    other: &[&GpuTensorHandle],
) -> Result<runmat_value::Value, RuntimeError> {
    use runmat_value::Value;
    let authority = crate::indexing::plan::admission::Owner::new(context)?;
    let provider = owner(input)?;
    let protected_len = other
        .len()
        .checked_add(1)
        .ok_or_else(|| semantic_error("IndexOutOfBounds", "resident protection extent overflow"))?;
    let protected = crate::indexing::plan::admission::array(context, protected_len, |index| {
        Ok(if index == 0 { input } else { other[index - 1] })
    })?;
    let output = match value {
        Value::Tensor(tensor) if runmat_accelerate_api::handle_is_logical(input) => {
            let array = crate::indexing::value::primitive_output::logical(
                context,
                tensor.shape(),
                |bytes| {
                    for (index, byte) in bytes.iter_mut().enumerate() {
                        if index % 4096 == 0 {
                            authority.check()?;
                        }
                        *byte = u8::from(
                            tensor
                                .numeric_value_at(index)
                                .expect("validated tensor")
                                .materialize_f64()
                                != 0.0,
                        );
                    }
                    Ok(())
                },
            )?;
            restore_logical(context, input, &array, &protected)?
        }
        Value::Tensor(tensor) => transfer::numeric(
            context,
            provider,
            &runmat_accelerate_api::HostNumericTensorView {
                data: gpu_helpers::tensor_numeric_view(&tensor),
                shape: tensor.shape(),
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            },
            &protected,
        )?,
        Value::ComplexTensor(_) if runmat_accelerate_api::handle_is_logical(input) => {
            return Err(semantic_error(
                "AssignmentTypeMismatch",
                "Logical assignment requires real numeric or logical values",
            ));
        }
        Value::ComplexTensor(tensor) => restore_complex(context, provider, &tensor, &protected)?,
        Value::LogicalArray(array) => restore_logical(context, input, &array, &protected)?,
        _ => {
            return Err(semantic_error(
                "AccelerationOperationFailed",
                "Indexed mutation produced an unsupported host carrier",
            ))
        }
    };
    if let Err(cause) = authority.check() {
        return Err(gpu_helpers::reject_gpu_output(output, cause));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

fn restore_logical(
    context: Option<&crate::context::RuntimeContext>,
    input: &GpuTensorHandle,
    array: &runmat_value::LogicalArray,
    protected: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, RuntimeError> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    if HostAllocationReceipt::policy(context).map_err(|source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            context,
            "RunMat:Indexing:Ownership",
            "logical indexed transfer",
            source,
        )
    })? == HostAllocationPolicy::Unbounded
    {
        return runmat_accelerate_api::upload_logical_protecting(
            owner(input)?,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &array.data,
                shape: &array.shape,
            },
            protected,
        )
        .map_err(|cause| failure("logical indexed result", cause.into_execution_error()));
    }
    use gpu_helpers::host_upload::GpuHostUpload;
    let mut upload = GpuHostUpload::prepare_borrowed_logical_output_in_optional_context(
        context,
        "indexing",
        input,
        &array.shape,
    )?;
    upload.publish_logical_protecting(
        &runmat_accelerate_api::HostLogicalTensorView {
            data: &array.data,
            shape: &array.shape,
        },
        protected,
    )
}

fn restore_complex(
    context: Option<&crate::context::RuntimeContext>,
    provider: &dyn AccelProvider,
    tensor: &runmat_value::ComplexTensor,
    protected: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, RuntimeError> {
    use crate::indexing::plan::admission::{Owner, Scratch};
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
    use runmat_value::{ComplexStorage, IntegerStorage};
    let owner = Owner::new(context)?;
    let lanes = tensor
        .len()
        .checked_mul(2)
        .ok_or_else(|| semantic_error("IndexOutOfBounds", "complex upload extent overflow"))?;
    macro_rules! upload {
        ($type:ty, $variant:ident, $real:expr, $imag:expr) => {{
            let mut interleaved = Scratch::<$type>::new(context, lanes)?;
            for (index, (&real, &imag)) in $real.iter().zip($imag.iter()).enumerate() {
                if index % 4096 == 0 {
                    owner.check()?;
                }
                interleaved.as_mut_slice()[2 * index] = real;
                interleaved.as_mut_slice()[2 * index + 1] = imag;
            }
            transfer::numeric(
                context,
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::$variant(interleaved.as_slice()),
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::ComplexInterleaved,
                },
                protected,
            )
        }};
    }
    macro_rules! pairs {
        ($type:ty, $variant:ident, $pairs:expr) => {{
            let mut interleaved = Scratch::<$type>::new(context, lanes)?;
            for (index, pair) in $pairs.iter().enumerate() {
                if index % 4096 == 0 {
                    owner.check()?;
                }
                interleaved.as_mut_slice()[2 * index] = pair.0;
                interleaved.as_mut_slice()[2 * index + 1] = pair.1;
            }
            transfer::numeric(
                context,
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::$variant(interleaved.as_slice()),
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::ComplexInterleaved,
                },
                protected,
            )
        }};
    }
    match tensor.complex_storage() {
        ComplexStorage::F32(values) => pairs!(f32, F32, values),
        ComplexStorage::F64(values) => pairs!(f64, F64, values),
        ComplexStorage::Integer(storage) => match (storage.real(), storage.imag()) {
            (IntegerStorage::I8(real), IntegerStorage::I8(imag)) => upload!(i8, I8, real, imag),
            (IntegerStorage::I16(real), IntegerStorage::I16(imag)) => upload!(i16, I16, real, imag),
            (IntegerStorage::I32(real), IntegerStorage::I32(imag)) => upload!(i32, I32, real, imag),
            (IntegerStorage::I64(real), IntegerStorage::I64(imag)) => upload!(i64, I64, real, imag),
            (IntegerStorage::U8(real), IntegerStorage::U8(imag)) => upload!(u8, U8, real, imag),
            (IntegerStorage::U16(real), IntegerStorage::U16(imag)) => upload!(u16, U16, real, imag),
            (IntegerStorage::U32(real), IntegerStorage::U32(imag)) => upload!(u32, U32, real, imag),
            (IntegerStorage::U64(real), IntegerStorage::U64(imag)) => upload!(u64, U64, real, imag),
            _ => Err(semantic_error(
                "AccelerationOperationFailed",
                "complex integer components must have matching classes",
            )),
        },
    }
}

pub(super) fn admit_read(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    shape: &[usize],
) -> Result<runmat_value::Value, RuntimeError> {
    let provider = owner(input)?;
    let logical = runmat_accelerate_api::handle_is_logical(input);
    let valid = output.shape() == shape
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|actual| std::ptr::eq(actual, provider))
        && !output.shares_allocation_with(input)
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::handle_storage(input)
        && runmat_accelerate_api::handle_is_logical(&output) == logical
        && (logical || output.physical().element_type() == input.physical().element_type());
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            semantic_error(
                "AccelerationOperationFailed",
                "Indexing provider returned an invalid class, shape, owner or alias",
            ),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

pub(super) fn empty_read_in_context(
    context: Option<&crate::context::RuntimeContext>,
    input: &GpuTensorHandle,
    shape: &[usize],
) -> Result<runmat_value::Value, RuntimeError> {
    use runmat_accelerate_api::{HostNumericDataView as Data, NumericElementType as Element};
    let provider = owner(input)?;
    if runmat_accelerate_api::handle_is_logical(input) {
        let array = crate::indexing::value::primitive_output::logical(context, shape, |_| Ok(()))?;
        restore_logical(context, input, &array, &[input])
            .and_then(|output| admit_read(output, input, shape))
    } else {
        let data = match input.physical().element_type() {
            Element::F64 => Data::F64(&[]),
            Element::F32 => Data::F32(&[]),
            Element::I8 => Data::I8(&[]),
            Element::I16 => Data::I16(&[]),
            Element::I32 => Data::I32(&[]),
            Element::I64 => Data::I64(&[]),
            Element::U8 => Data::U8(&[]),
            Element::U16 => Data::U16(&[]),
            Element::U32 => Data::U32(&[]),
            Element::U64 => Data::U64(&[]),
        };
        transfer::numeric(
            context,
            provider,
            &runmat_accelerate_api::HostNumericTensorView {
                data,
                shape,
                storage: runmat_accelerate_api::handle_storage(input),
            },
            &[input],
        )
        .and_then(|output| admit_read(output, input, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn logical_restore_preserves_low_rank_shape_and_owner() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for shape in [Vec::new(), vec![2]] {
                let count = shape.iter().product::<usize>();
                let input = runmat_accelerate_api::upload_logical_protecting(
                    provider,
                    &runmat_accelerate_api::HostLogicalTensorView {
                        data: &vec![0; count],
                        shape: &shape,
                    },
                    &[],
                )
                .expect("logical input");
                let host = runmat_value::Tensor::new(vec![1.0; count], shape.clone()).unwrap();
                let runmat_value::Value::GpuTensor(output) =
                    restore_in_context(None, &input, runmat_value::Value::Tensor(host), &[])
                        .unwrap()
                else {
                    panic!("resident logical output expected")
                };
                assert_eq!(output.shape(), shape.as_slice());
                assert!(runmat_accelerate_api::handle_is_logical(&output));
                assert!(!output.shares_allocation_with(&input));
                assert!(runmat_accelerate_api::same_provider_instance(
                    owner(&output).unwrap(),
                    owner(&input).unwrap(),
                ));
                let gathered =
                    futures::executor::block_on(gather_in_context(None, &output)).unwrap();
                let runmat_value::Value::LogicalArray(gathered) = gathered else {
                    panic!("logical host output expected")
                };
                assert_eq!(gathered.shape, shape);
                assert_eq!(gathered.data.as_slice(), vec![1; count]);
            }
        });
    }

    #[test]
    fn scalar_assignment_preserves_single_and_paired_integer_payloads() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let single = runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(&[1.0, 2.0]),
                    shape: &[1, 2],
                    storage: GpuTensorStorage::Real,
                },
                &[],
            )
            .unwrap();
            let output =
                futures::executor::block_on(crate::indexing::write_linear::assign_gpu_scalar(
                    &single,
                    &[1],
                    &runmat_value::Value::Num(7.0),
                    false,
                ))
                .unwrap();
            let runmat_value::Value::GpuTensor(output) = output else {
                panic!("resident");
            };
            assert_eq!(
                output.physical().element_type(),
                single.physical().element_type()
            );
            let actual = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
            assert_eq!(
                actual.data,
                runmat_accelerate_api::HostNumericDataOwned::F32(vec![7.0, 2.0])
            );
            assert!(!output.shares_allocation_with(&single));
            let original = futures::executor::block_on(provider.download_numeric(&single)).unwrap();
            assert_eq!(
                original.data,
                runmat_accelerate_api::HostNumericDataOwned::F32(vec![1.0, 2.0])
            );

            let complex = runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::U64(&[u64::MAX, 9_007_199_254_740_993, 4, 5]),
                    shape: &[1, 2],
                    storage: GpuTensorStorage::ComplexInterleaved,
                },
                &[],
            )
            .unwrap();
            let output =
                futures::executor::block_on(crate::indexing::write_linear::assign_gpu_scalar(
                    &complex,
                    &[2],
                    &runmat_value::Value::Int(runmat_value::IntValue::U64(7)),
                    false,
                ))
                .unwrap();
            let runmat_value::Value::GpuTensor(output) = output else {
                panic!("resident complex");
            };
            let actual = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
            assert_eq!(actual.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(
                actual.data,
                runmat_accelerate_api::HostNumericDataOwned::U64(vec![
                    u64::MAX,
                    9_007_199_254_740_993,
                    7,
                    0
                ])
            );
        });
    }
    #[test]
    fn empty_reads_preserve_single_complex_and_logical_producer_classes() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
                let data = if storage == GpuTensorStorage::Real {
                    &[3.0_f32][..]
                } else {
                    &[3.0_f32, 4.0][..]
                };
                let input = runmat_accelerate_api::upload_numeric_protecting(
                    provider,
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(data),
                        shape: &[1, 1],
                        storage,
                    },
                    &[],
                )
                .unwrap();
                let runmat_value::Value::GpuTensor(output) =
                    empty_read_in_context(None, &input, &[1, 0]).unwrap()
                else {
                    panic!("resident result");
                };
                assert_eq!(output.shape(), &[1, 0]);
                assert_eq!(
                    output.physical().element_type(),
                    input.physical().element_type()
                );
                assert_eq!(runmat_accelerate_api::handle_storage(&output), storage);
                assert!(!output.shares_allocation_with(&input));
            }
            let input = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .unwrap();
            let runmat_value::Value::GpuTensor(output) =
                empty_read_in_context(None, &input, &[0, 1]).unwrap()
            else {
                panic!("resident logical");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&output));
            assert_eq!(output.shape(), &[0, 1]);
            assert!(!output.shares_allocation_with(&input));
        });
    }
}
