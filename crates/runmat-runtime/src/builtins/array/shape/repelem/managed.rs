//! Admitted factor metadata and exact retained dense replication outputs.

use super::*;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation,
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, NumericDType,
};

const CHECKPOINT_CHUNK: usize = 4096;

mod character;

pub(super) fn character(
    context: &RuntimeContext,
    value: &CharArray,
    controls: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    character::assemble(context, value, controls)
}

pub(super) fn admission(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    build_runtime_error(format!("repelem: host allocation: {cause}"))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

fn source_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(&[]),
        _ => None,
    }
}

fn source_len(value: &Value) -> usize {
    match value {
        Value::Tensor(tensor) => tensor.len(),
        Value::ComplexTensor(tensor) => tensor.len(),
        Value::LogicalArray(array) => array.data.len(),
        _ => 1,
    }
}

fn source_pair(
    value: &Value,
    index: usize,
) -> crate::BuiltinResult<(NumericScalar, NumericScalar)> {
    let zero = NumericScalar::F64(0.0);
    match value {
        Value::Tensor(tensor) => tensor.numeric_value_at(index).map(|real| (real, zero)),
        Value::ComplexTensor(tensor) => tensor.numeric_value_at(index),
        Value::LogicalArray(array) => array
            .data
            .get(index)
            .map(|&real| (NumericScalar::F64(f64::from(real)), zero)),
        Value::Num(real) if index == 0 => Some((NumericScalar::F64(*real), zero)),
        Value::Int(real) if index == 0 => Some((NumericScalar::from(real.clone()), zero)),
        Value::Bool(real) if index == 0 => {
            Some((NumericScalar::F64(f64::from(u8::from(*real))), zero))
        }
        Value::Complex(real, imag) if index == 0 => {
            Some((NumericScalar::F64(*real), NumericScalar::F64(*imag)))
        }
        _ => None,
    }
    .ok_or_else(|| repelem_internal("repelem: source payload does not match its shape"))
}

fn fill_axis_table(
    context: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    control: Option<(&Value, usize)>,
    size: usize,
    table: &mut [usize],
) -> crate::BuiltinResult<()> {
    if let Some((control, position)) = control {
        let count = control_len(control, position)?;
        if count == 1 {
            let repetitions = control_at(control, 0, position)?;
            if repetitions != 0 {
                for (destination, slot) in table.iter_mut().enumerate() {
                    if destination % CHECKPOINT_CHUNK == 0 {
                        receipt
                            .checkpoint(context)
                            .map_err(|cause| admission(context, cause))?;
                    }
                    *slot = destination / repetitions;
                }
            }
        } else {
            let mut cursor = 0usize;
            for source_index in 0..size {
                if source_index % CHECKPOINT_CHUNK == 0 {
                    receipt
                        .checkpoint(context)
                        .map_err(|cause| admission(context, cause))?;
                }
                let repetitions = control_at(control, source_index, position)?;
                for slot in &mut table[cursor..cursor + repetitions] {
                    *slot = source_index;
                }
                cursor += repetitions;
            }
        }
    } else {
        for (index, slot) in table.iter_mut().enumerate() {
            if index % CHECKPOINT_CHUNK == 0 {
                receipt
                    .checkpoint(context)
                    .map_err(|cause| admission(context, cause))?;
            }
            *slot = index;
        }
    }
    Ok(())
}

fn control_len(control: &Value, position: usize) -> crate::BuiltinResult<usize> {
    match control {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(1),
        Value::Tensor(tensor) => {
            if tensor.len() != 1 {
                ensure_vector_shape(tensor.shape(), position)?;
            }
            Ok(tensor.len())
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                ensure_vector_shape(&array.shape, position)?;
            }
            Ok(array.data.len())
        }
        other => Err(repelem_invalid_factors(format!(
            "repelem: replication argument {position} must be numeric, got {other:?}"
        ))),
    }
}

fn control_at(control: &Value, index: usize, position: usize) -> crate::BuiltinResult<usize> {
    match control {
        Value::Num(value) if index == 0 => coerce_count(*value, position),
        Value::Int(value) if index == 0 => coerce_integer_count(value, position),
        Value::Bool(value) if index == 0 => Ok(usize::from(*value)),
        Value::Tensor(tensor) => match tensor.numeric_value_at(index) {
            Some(scalar) => match scalar.into_int_value() {
                Some(value) => coerce_integer_count(&value, position),
                None => coerce_count(scalar.materialize_f64(), position),
            },
            None => Err(repelem_invalid_factors(
                "repelem: control payload changed during preparation",
            )),
        },
        Value::LogicalArray(array) => array
            .data
            .get(index)
            .map(|&value| usize::from(value != 0))
            .ok_or_else(|| {
                repelem_invalid_factors("repelem: control payload changed during preparation")
            }),
        _ => Err(repelem_invalid_factors("repelem: invalid control class")),
    }
}

fn axis_control(
    controls: crate::call::FunctionArgumentView<'_>,
    single_axis: Option<usize>,
    axis: usize,
) -> Option<(&Value, usize)> {
    match single_axis {
        Some(selected) if selected == axis => controls.first().map(|value| (value, 1)),
        Some(_) => None,
        None => controls.get(axis).map(|value| (value, axis + 1)),
    }
}

fn axis_extent(
    context: &RuntimeContext,
    controls: crate::call::FunctionArgumentView<'_>,
    single_axis: Option<usize>,
    axis: usize,
    source_extent: usize,
) -> crate::BuiltinResult<usize> {
    let Some((control, position)) = axis_control(controls, single_axis, axis) else {
        return Ok(source_extent);
    };
    let count = control_len(control, position)?;
    if count == 1 {
        return source_extent
            .checked_mul(control_at(control, 0, position)?)
            .ok_or_else(|| {
                repelem_invalid_factors("repelem: requested output exceeds maximum size")
            });
    }
    if count != source_extent {
        return Err(repelem_invalid_factors(format!(
            "repelem: replication vector at dimension {} has length {} but the input dimension has size {}",
            axis + 1, count, source_extent
        )));
    }
    let mut total = 0usize;
    for index in 0..count {
        if index % CHECKPOINT_CHUNK == 0 {
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| admission(context, cause))?;
        }
        total = total
            .checked_add(control_at(control, index, position)?)
            .ok_or_else(|| {
                repelem_invalid_factors("repelem: requested output exceeds maximum size")
            })?;
    }
    Ok(total)
}

enum Output {
    Real(ExactHostTensorAllocation),
    Logical(ExactHostLogicalTensorAllocation),
    Complex(ExactHostComplexTensorAllocation),
    Paired(ExactHostIntegerComplexTensorAllocation),
}

impl Output {
    fn set(
        &mut self,
        index: usize,
        real: NumericScalar,
        imag: NumericScalar,
    ) -> crate::BuiltinResult<()> {
        match self {
            Self::Real(output) => output.set_value(index, real).map_err(repelem_internal),
            Self::Logical(output) => {
                output.as_mut_slice()[index] = u8::from(real.materialize_f64() != 0.0);
                Ok(())
            }
            Self::Complex(output) => output
                .set(index, real.materialize_f64(), imag.materialize_f64())
                .map_err(|error| repelem_internal(format!("repelem: {error}"))),
            Self::Paired(output) => output.set(index, real, imag).map_err(repelem_internal),
        }
    }

    fn into_value(self) -> crate::BuiltinResult<Value> {
        match self {
            Self::Real(output) => output.into_tensor().map(tensor::tensor_into_value),
            Self::Logical(output) => output.into_array().map(Value::LogicalArray),
            Self::Complex(output) => output.into_tensor().map(complex_tensor_into_value),
            Self::Paired(output) => output.into_tensor().map(Value::ComplexTensor),
        }
        .map_err(|error| repelem_internal(format!("repelem: {error}")))
    }
}

pub(super) fn dense(
    context: &RuntimeContext,
    value: &Value,
    controls: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    let original = source_shape(value)
        .ok_or_else(|| repelem_unsupported("repelem: unsupported finite dense input"))?;
    let rank = original.len().max(2).max(if controls.len() == 1 {
        0
    } else {
        controls.len()
    });
    let source_extent = |axis: usize| original.get(axis).copied().unwrap_or(1);
    let source_total = if original.contains(&0) {
        0
    } else {
        original
            .iter()
            .try_fold(1usize, |total, &extent| total.checked_mul(extent))
            .ok_or_else(|| repelem_invalid_factors("repelem: source geometry overflow"))?
    };
    if source_total != source_len(value) {
        return Err(repelem_internal(
            "repelem: source payload does not match its shape",
        ));
    }
    let single_axis = if controls.len() == 1 {
        let mut found = None;
        for axis in 0..rank {
            if source_extent(axis) > 1 {
                if found.is_some() {
                    return Err(repelem_invalid_factors(
                        "repelem: one replication control requires vector input",
                    ));
                }
                found = Some(axis);
            }
        }
        Some(
            found.unwrap_or(if source_total == 0 && source_extent(0) != 1 {
                0
            } else {
                1
            }),
        )
    } else {
        None
    };
    let mut has_zero_extent = false;
    let mut table_len = 0usize;
    for axis in 0..rank {
        let extent = axis_extent(context, controls, single_axis, axis, source_extent(axis))?;
        table_len = table_len
            .checked_add(extent)
            .ok_or_else(|| repelem_invalid_factors("repelem: metadata extent overflow"))?;
        has_zero_extent |= extent == 0;
    }
    let mut output_len: usize = if has_zero_extent { 0 } else { 1 };
    if !has_zero_extent {
        for axis in 0..rank {
            let extent = axis_extent(context, controls, single_axis, axis, source_extent(axis))?;
            output_len = output_len.checked_mul(extent).ok_or_else(|| {
                repelem_invalid_factors("repelem: requested output exceeds maximum size")
            })?;
        }
    }
    let dtype = match value {
        Value::Tensor(tensor) => tensor.numeric_dtype(),
        Value::Int(integer) => NumericDType::from(integer.integer_class()),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype(),
        Value::LogicalArray(_) | Value::Bool(_) => NumericDType::F64,
        _ => NumericDType::F64,
    };
    let (raw, claims) = match value {
        Value::LogicalArray(_) | Value::Bool(_) => (
            ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(output_len, rank),
            ExactHostLogicalTensorAllocation::retained_claim_count(),
        ),
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => (
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype, output_len, rank,
            ),
            ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
        ),
        Value::ComplexTensor(_) | Value::Complex(_, _) => (
            ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype, output_len, rank,
            ),
            ExactHostComplexTensorAllocation::retained_claim_count(),
        ),
        _ => (
            ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, output_len, rank),
            ExactHostTensorAllocation::retained_claim_count(),
        ),
    };
    let raw = raw.map_err(|cause| admission(context, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| admission(context, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| admission(context, cause))?;
    let metadata_len = rank
        .checked_mul(3)
        .and_then(|count| count.checked_add(table_len))
        .ok_or_else(|| repelem_invalid_factors("repelem: metadata extent overflow"))?;
    let scratch = HostTypedWorkspace::<usize>::quoted_bytes(metadata_len)
        .map_err(|cause| admission(context, cause))?;
    let plan =
        HostAllocationPlan::new(scratch, retained).map_err(|cause| admission(context, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|cause| admission(context, cause))?
        .ok_or_else(|| admission(context, HostAllocationError::MissingManagedOwner))?;
    let mut metadata = receipt
        .temporary_typed::<usize>(context, metadata_len)
        .map_err(|cause| admission(context, cause))?;
    let (output_shape, rest) = metadata.as_mut_slice().split_at_mut(rank);
    let (strides, rest) = rest.split_at_mut(rank);
    let (offsets, tables) = rest.split_at_mut(rank);
    let mut next_table = 0usize;
    let mut stride = 1usize;
    for axis in 0..rank {
        let size = source_extent(axis);
        let extent = axis_extent(context, controls, single_axis, axis, size)?;
        output_shape[axis] = extent;
        strides[axis] = stride;
        stride = if source_total == 0 {
            0
        } else {
            stride
                .checked_mul(size)
                .ok_or_else(|| repelem_invalid_factors("repelem: source stride overflow"))?
        };
        offsets[axis] = next_table;
        let table = &mut tables[next_table..next_table + extent];
        fill_axis_table(
            context,
            &receipt,
            axis_control(controls, single_axis, axis),
            size,
            table,
        )?;
        next_table += extent;
    }
    let mut output = receipt
        .with_output_allocation(context, |permit| match value {
            Value::LogicalArray(_) | Value::Bool(_) => {
                ExactHostLogicalTensorAllocation::new(output_shape, permit).map(Output::Logical)
            }
            Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
                ExactHostIntegerComplexTensorAllocation::new(dtype, output_shape, permit)
                    .map(Output::Paired)
            }
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                ExactHostComplexTensorAllocation::new(dtype, output_shape, permit)
                    .map(Output::Complex)
            }
            _ => ExactHostTensorAllocation::new(dtype, output_shape, permit).map(Output::Real),
        })
        .map_err(|cause| admission(context, cause))?;
    for destination in 0..output_len {
        if destination % CHECKPOINT_CHUNK == 0 {
            receipt
                .checkpoint(context)
                .map_err(|cause| admission(context, cause))?;
        }
        let mut quotient = destination;
        let mut source = 0usize;
        for axis in 0..rank {
            let coordinate = quotient % output_shape[axis];
            quotient /= output_shape[axis];
            let source_axis = tables[offsets[axis] + coordinate];
            source = source
                .checked_add(
                    source_axis
                        .checked_mul(strides[axis])
                        .ok_or_else(|| repelem_internal("repelem: source index overflow"))?,
                )
                .ok_or_else(|| repelem_internal("repelem: source index overflow"))?;
        }
        let (real, imag) = source_pair(value, source)?;
        output.set(destination, real, imag)?;
    }
    receipt
        .checkpoint(context)
        .map_err(|cause| admission(context, cause))?;
    output.into_value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use runmat_value::{IntegerStorage, Tensor};
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
    fn exact_vector_counts_keep_class_and_refund_after_output_drop() {
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let context = runtime(&ledger);
        let source = Value::Tensor(Tensor::from_f32(vec![3.0, 5.0], vec![2, 1]).unwrap());
        let counts = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![2, 1]).unwrap(),
        );
        let result = dense(
            &context,
            &source,
            crate::call::FunctionArgumentView::from_slice(&[counts]),
        )
        .unwrap();
        let Value::Tensor(tensor) = &result else {
            panic!("native single output")
        };
        assert_eq!(tensor.shape(), &[3, 1]);
        assert_eq!(tensor.as_f32_slice(), Some([3.0, 5.0, 5.0].as_slice()));
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(result);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let chars = CharArray::new_row("ab");
        let repeated = character(
            &context,
            &chars,
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)]),
        )
        .unwrap();
        let Value::CharArray(output) = &repeated else {
            panic!("character output")
        };
        assert_eq!(output.shape(), &[1, 4]);
        assert_eq!(
            runmat_value::decode_utf16_code_units(output.data()).expect("valid UTF-16 fixture"),
            "aabb"
        );
        let selected = futures::executor::block_on(super::super::repelem_builtin(
            Some(runtime(&ledger)),
            Value::CharArray(CharArray::new_row("ab")),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .unwrap();
        assert!(
            matches!(&selected, Value::CharArray(chars) if runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture") == "aabb")
        );
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop((repeated, selected));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

        let denied = ManagedHostEffectLedger::new(0);
        assert!(dense(
            &runtime(&denied),
            &source,
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)])
        )
        .is_err());
        assert!(character(
            &runtime(&denied),
            &chars,
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)])
        )
        .is_err());
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);
    }
}
