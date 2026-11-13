//! Exact metadata children and the existing selected object publication owner.
use super::*;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{ExactArrayInitializationError, Utf8Text};

pub(super) fn publish(
    spec: &DecompositionSpec,
    runtime: &RuntimeContext,
) -> BuiltinResult<crate::execution::RootedValue> {
    let real = spec
        .source_is_real
        .ok_or_else(|| internal("decomposition source facts were not selected"))?;
    let text = decomposition_type_name(spec.decomp_type);
    let dtype = matrix_datatype(&spec.matrix);
    let triangle = match spec.triangle {
        Some(TriangularFlag::Upper) => "upper",
        Some(TriangularFlag::Lower) => "lower",
        None => "",
    };
    let mut bytes = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 2])
        .map_err(|e| native::allocation(e, runtime))?;
    for text in [Some(text), Some(dtype), spec.triangle.map(|_| triangle)]
        .into_iter()
        .flatten()
    {
        let raw = Utf8Text::requested_bytes_for_capacity(text.len())
            .map_err(|e| native::allocation(HostAllocationError::Numeric(e), runtime))?;
        let next = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw)
                .map_err(|_| native::allocation(HostAllocationError::Overflow, runtime))?,
            usize::from(!text.is_empty()),
        )
        .map_err(|e| native::allocation(e, runtime))?;
        bytes = bytes
            .checked_add(next)
            .ok_or_else(|| native::allocation(HostAllocationError::Overflow, runtime))?;
    }
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, bytes).map_err(|e| native::allocation(e, runtime))?,
    )
    .map_err(|e| native::allocation(e, runtime))?;
    let string = |text: &str| -> BuiltinResult<Value> {
        let Some(r) = receipt.as_ref() else {
            return Ok(Value::string(text.to_owned()));
        };
        let permit = r
            .output_permit(runtime)
            .map_err(|e| native::allocation(e, runtime))?;
        let text = Utf8Text::try_copy_str_exact_with(text, &permit, |_| r.checkpoint(runtime))
            .map_err(|e| match e {
                ExactArrayInitializationError::Allocation(e) => {
                    native::allocation(HostAllocationError::Numeric(e), runtime)
                }
                ExactArrayInitializationError::Element(e) => native::allocation(e, runtime),
            })?;
        r.checkpoint(runtime)
            .map_err(|e| native::allocation(e, runtime))?;
        Ok(Value::String(text.into()))
    };
    let (m, n) = matrix_dims(&spec.matrix);
    let (m, n) = if spec.is_conjugate_transposed {
        (n, m)
    } else {
        (m, n)
    };
    let size = if let Some(r) = receipt.as_ref() {
        let mut size = r
            .exact_output_tensor(runtime, NumericDType::F64, &[1, 2])
            .map_err(|e| native::allocation(e, runtime))?;
        size.as_f64_slice_mut()
            .expect("double metadata")
            .copy_from_slice(&[m as f64, n as f64]);
        Value::Tensor(
            size.into_tensor()
                .map_err(|e| native::allocation(HostAllocationError::Numeric(e), runtime))?,
        )
    } else {
        Value::Tensor(Tensor::new(vec![m as f64, n as f64], vec![1, 2]).map_err(internal)?)
    };
    let triangle_value = spec.triangle.map(|_| string(triangle)).transpose()?;
    let rank_value = spec.rank_tolerance.map(Value::Num);
    let fields = [
        (MATRIX_FIELD, spec.matrix.clone()),
        (TYPE_FIELD, string(text)?),
        (MATRIX_SIZE_FIELD, size),
        (CHECK_CONDITION_FIELD, Value::Bool(spec.check_condition)),
        (DATATYPE_FIELD, string(dtype)?),
        (
            IS_CONJUGATE_TRANSPOSED_FIELD,
            Value::Bool(spec.is_conjugate_transposed),
        ),
        (IS_REAL_FIELD, Value::Bool(real && spec.scale.1 == 0.0)),
        (IS_SPARSE_FIELD, Value::Bool(spec.is_sparse)),
        (SCALE_FACTOR_FIELD, complex_scalar_to_value(spec.scale)),
        (SOURCE_REAL_FIELD, Value::Bool(real)),
    ];
    if let Some(r) = receipt.as_ref() {
        r.checkpoint(runtime)
            .map_err(|e| native::allocation(e, runtime))?;
    }
    let root = crate::execution::object::construct_object_from_fields_with_native(
        Some(runtime),
        DECOMPOSITION_CLASS_IDENTITY.display_name(),
        fields
            .iter()
            .map(|(key, value)| (*key, value))
            .chain(rank_value.iter().map(|value| (RANK_TOLERANCE_FIELD, value)))
            .chain(triangle_value.iter().map(|value| (TRIANGLE_FIELD, value))),
        spec.factor_data
            .as_ref()
            .map(crate::execution::object::ObjectNativeInitialization::NumericFactor),
    )?;
    if let Some(r) = receipt.as_ref() {
        r.checkpoint(runtime)
            .map_err(|e| native::allocation(e, runtime))?;
    }
    Ok(root)
}
