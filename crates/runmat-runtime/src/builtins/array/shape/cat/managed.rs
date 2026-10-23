//! Exact retained numerical concatenation for selected finite host execution.

use super::*;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use runmat_value::{ExactHostLogicalTensorAllocation, NumericScalar, Value};

const CHECKPOINT_CHUNK: usize = 4096;

mod character;
mod complex;
pub(super) use character::character;
pub(super) use complex::complex;

pub(super) fn admission_error(
    context: &RuntimeContext,
    cause: HostAllocationError,
) -> RuntimeError {
    build_runtime_error(format!("cat: host allocation: {cause}"))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

fn source_shape(value: &Value) -> BuiltinResult<&[usize]> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape()),
        Value::ComplexTensor(tensor) => Ok(tensor.shape()),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(&[]),
        _ => Err(cat_err(
            "cat: selected numerical owner received a nonnumerical input",
        )),
    }
}

fn source_len(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.len()),
        Value::ComplexTensor(tensor) => Ok(tensor.len()),
        Value::LogicalArray(array) => Ok(array.data.len()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(1),
        _ => Err(cat_err(
            "cat: selected numerical owner received a nonnumerical input",
        )),
    }
}

fn checked_extent_product(extents: &[usize]) -> Option<usize> {
    if extents.contains(&0) {
        Some(0)
    } else {
        checked_product(extents)
    }
}

fn extent(value: &Value, axis: usize) -> BuiltinResult<usize> {
    Ok(source_shape(value)?.get(axis).copied().unwrap_or(1))
}

fn active(value: &Value, has_nonempty: bool) -> BuiltinResult<bool> {
    Ok(!has_nonempty || !source_shape(value)?.contains(&0))
}

fn output_extent(
    values: FunctionArgumentView<'_>,
    axis: usize,
    concat_axis: usize,
    has_nonempty: bool,
) -> BuiltinResult<usize> {
    let mut first = None;
    let mut sum = 0usize;
    for (index, value) in values.iter().enumerate() {
        if !active(value, has_nonempty)? {
            continue;
        }
        let current = extent(value, axis)?;
        if axis == concat_axis {
            sum = sum
                .checked_add(current)
                .ok_or_else(|| cat_err("cat: concatenated dimension exceeds maximum size"))?;
        } else if let Some(expected) = first {
            if current != expected {
                return Err(cat_err(format!(
                    "cat: dimension {} mismatch between input 1 and input {}",
                    axis + 1,
                    index + 1
                )));
            }
        } else {
            first = Some(current);
        }
    }
    Ok(if axis == concat_axis {
        sum
    } else {
        first.unwrap_or(1)
    })
}

fn source_scalar(value: &Value, index: usize) -> BuiltinResult<NumericScalar> {
    let scalar = match value {
        Value::Tensor(tensor) => tensor.host_buffer().value_at(index),
        Value::LogicalArray(array) => array
            .data
            .get(index)
            .copied()
            .map(|value| NumericScalar::F64(f64::from(value))),
        Value::Num(value) if index == 0 => Some(NumericScalar::F64(*value)),
        Value::Int(value) if index == 0 => Some(NumericScalar::from(value.clone())),
        Value::Bool(value) if index == 0 => Some(NumericScalar::F64(f64::from(u8::from(*value)))),
        _ => None,
    };
    scalar.ok_or_else(|| cat_err("cat: source geometry changed during numerical assembly"))
}

fn convert_scalar(
    scalar: NumericScalar,
    dtype: NumericDType,
    integer: Option<IntegerClass>,
) -> NumericScalar {
    if let Some(target) = integer {
        let converted = match scalar.into_int_value() {
            Some(value) => target.cast_int(&value),
            None => target.cast_scalar(scalar.materialize_f64()),
        };
        NumericScalar::from(converted)
    } else if dtype == NumericDType::F32 {
        NumericScalar::F32(scalar.materialize_f64() as f32)
    } else {
        NumericScalar::F64(scalar.materialize_f64())
    }
}

struct CatGeometry {
    rank: usize,
    output_rank: usize,
    len: usize,
    has_nonempty: bool,
}

fn inspect_geometry(
    dim_zero: usize,
    values: FunctionArgumentView<'_>,
) -> BuiltinResult<CatGeometry> {
    let mut has_nonempty = false;
    let mut rank = dim_zero
        .checked_add(1)
        .ok_or_else(|| cat_err("cat: dimension exceeds maximum size"))?;
    for value in values {
        let shape = source_shape(value)?;
        has_nonempty |= !shape.contains(&0);
        rank = rank.max(shape.len());
    }
    let mut has_zero = false;
    for axis in 0..rank {
        has_zero |= output_extent(values, axis, dim_zero, has_nonempty)? == 0;
    }
    let mut len = 1usize;
    if has_zero {
        len = 0;
    } else {
        for axis in 0..rank {
            len = len
                .checked_mul(output_extent(values, axis, dim_zero, has_nonempty)?)
                .ok_or_else(|| cat_err("cat: resulting array exceeds maximum size"))?;
        }
    }
    for value in values {
        let shape = source_shape(value)?;
        let count = if shape.contains(&0) {
            0
        } else {
            checked_product(shape)
                .ok_or_else(|| cat_err("cat: input dimensions exceed maximum size"))?
        };
        if count != source_len(value)? {
            return Err(cat_err(
                "cat: input shape does not match its numerical payload",
            ));
        }
    }
    let minimum_rank = (dim_zero + 1).max(2).min(rank);
    let mut output_rank = minimum_rank;
    for axis in minimum_rank..rank {
        if output_extent(values, axis, dim_zero, has_nonempty)? != 1 {
            output_rank = axis + 1;
        }
    }
    Ok(CatGeometry {
        rank,
        output_rank,
        len,
        has_nonempty,
    })
}

fn fill_elements(
    context: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    dim_zero: usize,
    values: FunctionArgumentView<'_>,
    shape: &[usize],
    geometry: &CatGeometry,
    mut write: impl FnMut(usize, &Value, usize) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let inner = if dim_zero == 0 {
        1
    } else {
        checked_extent_product(&shape[..dim_zero])
            .ok_or_else(|| cat_err("cat: inner dimension overflow"))?
    };
    let outer = if dim_zero + 1 >= geometry.rank {
        1
    } else {
        checked_extent_product(&shape[dim_zero + 1..])
            .ok_or_else(|| cat_err("cat: outer dimension overflow"))?
    };
    let mut destination = 0usize;
    for outer_index in 0..outer {
        for value in values {
            if !active(value, geometry.has_nonempty)? {
                continue;
            }
            let chunk = extent(value, dim_zero)?
                .checked_mul(inner)
                .ok_or_else(|| cat_err("cat: source chunk size overflow"))?;
            let offset = outer_index
                .checked_mul(chunk)
                .ok_or_else(|| cat_err("cat: source offset overflow"))?;
            let end = offset
                .checked_add(chunk)
                .ok_or_else(|| cat_err("cat: source range overflow"))?;
            for source_index in offset..end {
                if destination.is_multiple_of(CHECKPOINT_CHUNK) {
                    receipt
                        .checkpoint(context)
                        .map_err(|cause| admission_error(context, cause))?;
                }
                write(destination, value, source_index)?;
                destination += 1;
            }
        }
    }
    if destination != geometry.len {
        return Err(cat_err("cat: assembly did not fill admitted output"));
    }
    receipt
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))
}

pub(super) fn numeric(
    context: &RuntimeContext,
    dim_zero: usize,
    values: FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let geometry = inspect_geometry(dim_zero, values)?;
    let integer = leftmost_integer_target(values);
    let dtype = integer.map(NumericDType::from).unwrap_or_else(|| {
        if values.iter().any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        }) {
            NumericDType::F32
        } else {
            NumericDType::F64
        }
    });
    let scratch = HostTypedWorkspace::<usize>::quoted_bytes(geometry.rank)
        .map_err(|cause| admission_error(context, cause))?;
    let retained = HostAllocationPlan::retained_tensor_bytes_for_dimensions(
        dtype,
        geometry.len,
        geometry.output_rank,
    )
    .map_err(|cause| admission_error(context, cause))?;
    let plan = HostAllocationPlan::new(scratch, retained)
        .map_err(|cause| admission_error(context, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|cause| admission_error(context, cause))?
        .ok_or_else(|| admission_error(context, HostAllocationError::MissingManagedOwner))?;
    let mut shape = receipt
        .temporary_typed::<usize>(context, geometry.rank)
        .map_err(|cause| admission_error(context, cause))?;
    for axis in 0..geometry.rank {
        shape.as_mut_slice()[axis] = output_extent(values, axis, dim_zero, geometry.has_nonempty)?;
    }
    let mut output = receipt
        .exact_output_tensor(context, dtype, &shape.as_slice()[..geometry.output_rank])
        .map_err(|cause| admission_error(context, cause))?;
    fill_elements(
        context,
        &receipt,
        dim_zero,
        values,
        shape.as_slice(),
        &geometry,
        |destination, value, source_index| {
            output
                .set_value(
                    destination,
                    convert_scalar(source_scalar(value, source_index)?, dtype, integer),
                )
                .map_err(cat_err)
        },
    )?;
    output
        .into_tensor()
        .map(tensor::tensor_into_value)
        .map_err(|cause| cat_err(format!("cat: {cause}")))
}

pub(super) fn logical(
    context: &RuntimeContext,
    dim_zero: usize,
    values: FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let geometry = inspect_geometry(dim_zero, values)?;
    let raw = ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(
        geometry.len,
        geometry.output_rank,
    )
    .map_err(|cause| admission_error(context, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| admission_error(context, HostAllocationError::Overflow))?,
        ExactHostLogicalTensorAllocation::retained_claim_count(),
    )
    .map_err(|cause| admission_error(context, cause))?;
    let scratch = HostTypedWorkspace::<usize>::quoted_bytes(geometry.rank)
        .map_err(|cause| admission_error(context, cause))?;
    let plan = HostAllocationPlan::new(scratch, retained)
        .map_err(|cause| admission_error(context, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|cause| admission_error(context, cause))?
        .ok_or_else(|| admission_error(context, HostAllocationError::MissingManagedOwner))?;
    let mut shape = receipt
        .temporary_typed::<usize>(context, geometry.rank)
        .map_err(|cause| admission_error(context, cause))?;
    for axis in 0..geometry.rank {
        shape.as_mut_slice()[axis] = output_extent(values, axis, dim_zero, geometry.has_nonempty)?;
    }
    let mut output = receipt
        .with_output_allocation(context, |permit| {
            ExactHostLogicalTensorAllocation::new(&shape.as_slice()[..geometry.output_rank], permit)
        })
        .map_err(|cause| admission_error(context, cause))?;
    fill_elements(
        context,
        &receipt,
        dim_zero,
        values,
        shape.as_slice(),
        &geometry,
        |destination, value, source_index| {
            let element = match value {
                Value::Bool(element) if source_index == 0 => u8::from(*element),
                Value::LogicalArray(array) => *array.data.get(source_index).ok_or_else(|| {
                    cat_err("cat: logical source geometry changed during assembly")
                })?,
                _ => {
                    return Err(cat_err(
                        "cat: selected logical owner received nonlogical input",
                    ))
                }
            };
            output.as_mut_slice()[destination] = element;
            Ok(())
        },
    )?;
    output
        .into_array()
        .map(Value::LogicalArray)
        .map_err(|cause| admission_error(context, HostAllocationError::Numeric(cause)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use runmat_value::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor,
    };
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    fn runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
            )
    }

    #[test]
    fn exact_dense_cat_categories_retain_claims_and_refund() {
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let context = runtime(&ledger);
        let single = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let numerical_value = numeric(
            &context,
            1,
            FunctionArgumentView::from_slice(&[single.clone(), single]),
        )
        .unwrap();
        let Value::Tensor(numerical) = &numerical_value else {
            panic!("single tensor output")
        };
        assert_eq!(numerical.shape(), &[2, 2]);
        assert_eq!(
            numerical.as_f32_slice(),
            Some([1.0, 2.0, 1.0, 2.0].as_slice())
        );

        let logical_input = Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap());
        let logical_value = logical(
            &context,
            1,
            FunctionArgumentView::from_slice(&[logical_input.clone(), logical_input]),
        )
        .unwrap();
        let Value::LogicalArray(logical) = &logical_value else {
            panic!("logical output")
        };
        assert_eq!(logical.data.as_slice(), &[1, 0, 1, 0]);

        let paired = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![7]),
                )
                .unwrap(),
                vec![1, 1],
            )
            .unwrap(),
        );
        let complex_value = complex(
            &context,
            1,
            FunctionArgumentView::from_slice(&[paired.clone(), paired]),
        )
        .unwrap();
        let Value::ComplexTensor(complex_output) = &complex_value else {
            panic!("paired output")
        };
        let storage = complex_output
            .integer_storage()
            .expect("exact paired output");
        assert_eq!(
            storage.real(),
            &IntegerStorage::U64(vec![u64::MAX, u64::MAX])
        );
        assert_eq!(storage.imag(), &IntegerStorage::U64(vec![7, 7]));

        let characters = character(
            &context,
            1,
            FunctionArgumentView::from_slice(&[
                Value::CharArray(
                    CharArray::from_code_units(
                        [0xD800, 0xDC00]
                            .map(runmat_value::Utf16CodeUnit::from_raw)
                            .to_vec(),
                        1,
                        2,
                    )
                    .unwrap(),
                ),
                Value::Num(65535.0),
            ]),
        )
        .unwrap();
        let Value::CharArray(chars) = &characters else {
            panic!("character output")
        };
        assert_eq!(chars.shape(), &[1, 3]);
        assert_eq!(
            chars
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            vec![0xD800, 0xDC00, 0xFFFF]
        );
        assert_eq!(std::mem::size_of_val(chars.data()), 6);
        let rank_one = CharArray::from_unicode_scalars_with_shape(vec!['x', 'y'], vec![2]).unwrap();
        let empty = CharArray::from_unicode_scalars_with_shape(Vec::new(), Vec::new()).unwrap();
        let normalized = character(
            &context,
            1,
            FunctionArgumentView::from_slice(&[
                Value::CharArray(empty.clone()),
                Value::CharArray(rank_one.clone()),
                Value::CharArray(CharArray::new_row("z")),
            ]),
        )
        .unwrap();
        let Value::CharArray(normalized_chars) = &normalized else {
            panic!("normalized character output")
        };
        assert_eq!(normalized_chars.shape(), &[1, 3]);
        assert_eq!(
            runmat_value::decode_utf16_code_units(normalized_chars.data())
                .expect("valid UTF-16 fixture text"),
            "xyz"
        );
        let ordinary = super::super::cat_host_values(
            1,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::CharArray(empty),
                Value::CharArray(rank_one),
                Value::CharArray(CharArray::new_row("z")),
            ]),
            &super::super::LikeSpec::default(),
            None,
        )
        .unwrap();
        assert_eq!(ordinary, normalized);
        let char_alias = characters.clone();
        let selected = futures::executor::block_on(super::super::cat_builtin(
            Some(runtime(&ledger)),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::CharArray(
                    CharArray::from_code_units(
                        [0xD800, 0xDC00]
                            .map(runmat_value::Utf16CodeUnit::from_raw)
                            .to_vec(),
                        1,
                        2,
                    )
                    .unwrap(),
                ),
                Value::Num(65535.0),
            ]),
        ))
        .unwrap();
        assert!(
            matches!(&selected, Value::CharArray(chars) if chars.data().iter().map(|unit|unit.raw()).collect::<Vec<_>>() == vec![0xD800,0xDC00,0xFFFF])
        );

        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop((
            numerical_value,
            logical_value,
            complex_value,
            characters,
            normalized,
            selected,
        ));
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(char_alias);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let denied = ManagedHostEffectLedger::new(0);
        let denied_context = runtime(&denied);
        assert!(numeric(
            &denied_context,
            0,
            FunctionArgumentView::from_slice(&[Value::Num(1.0), Value::Num(2.0)])
        )
        .is_err());
        assert!(character(
            &denied_context,
            1,
            FunctionArgumentView::from_slice(&[Value::CharArray(CharArray::new_row("a"))])
        )
        .is_err());
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);
    }
}
