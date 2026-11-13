use super::*;

pub fn concatenate(dimension: usize, inputs: &[Value]) -> Result<Value, RuntimeError> {
    let runtime = crate::context::legacy::active();
    let _roots = match runtime.as_ref() {
        Some(runtime) => {
            crate::execution::RootedValueLease::for_value_slices_in_context(&[inputs], runtime)?
        }
        None => {
            HostAllocationReceipt::capture_owner(None).map_err(|e| allocation(None, e))?;
            crate::execution::RootedValueLease::for_values(inputs.iter())
                .map_err(|e| e.into_runtime_error())?
        }
    };
    concatenate_iter(dimension, inputs.iter(), inputs.len(), runtime.as_ref())
}

pub(crate) fn concatenate_arguments(
    dimension: usize,
    inputs: &crate::call::BuiltinArguments,
    omit_empty: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<Value, RuntimeError> {
    let selected = || {
        inputs.iter().filter(move |value| {
            !omit_empty || !crate::builtins::common::validation::value_is_empty(value)
        })
    };
    let count = selected().count();
    concatenate_iter(dimension, selected(), count, runtime)
}

fn concatenate_iter<'a, I>(
    dimension: usize,
    inputs: I,
    input_count: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<Value, RuntimeError>
where
    I: Iterator<Item = &'a Value> + Clone,
{
    let first = inputs
        .clone()
        .next()
        .ok_or_else(|| invalid("structure concatenation requires an input"))?;
    let rank = inputs
        .clone()
        .try_fold(2usize, |rank, value| {
            Ok::<_, RuntimeError>(rank.max(shape(value)?.len()))
        })?
        .max(
            dimension
                .checked_add(1)
                .ok_or_else(|| invalid("concatenation rank overflow"))?,
        )
        .max(2);
    // Prefix axis boundaries locate each source in logarithmic time without
    // allocating a per-output mapping or materializing per-element structures.
    let slots = rank
        .checked_add(input_count)
        .ok_or_else(|| invalid("concatenation metadata overflow"))?;
    let mut scratch = Scratch::new(runtime, slots)?;
    let (output, prefix) = scratch.values_mut().split_at_mut(rank);
    let first_shape = shape(first)?;
    for (axis, extent) in output.iter_mut().enumerate() {
        *extent = first_shape.get(axis).copied().unwrap_or(1);
    }
    let mut sum = 0usize;
    for (input, boundary) in inputs.clone().zip(prefix) {
        let input_shape = shape(input)?;
        for (axis, &extent) in output.iter().enumerate() {
            if axis != dimension && input_shape.get(axis).copied().unwrap_or(1) != extent {
                return Err(invalid("structure concatenation dimensions do not agree"));
            }
        }
        validate_fields(first, input)?;
        sum = sum
            .checked_add(input_shape.get(dimension).copied().unwrap_or(1))
            .ok_or_else(|| invalid("concatenation extent overflow"))?;
        *boundary = sum;
    }
    output[dimension] = sum;
    scratch.checkpoint()?;
    let (output, prefix) = scratch.values().split_at(rank);
    let dimension_stride = count(&output[..dimension])?;
    let element = |name: &str, index: usize| {
        let coordinate = index / dimension_stride % output[dimension];
        let input_index = prefix.partition_point(|&boundary| boundary <= coordinate);
        let input = inputs
            .clone()
            .nth(input_index)
            .ok_or_else(|| invalid("concatenation source is missing"))?;
        let local = coordinate
            - if input_index == 0 {
                0
            } else {
                prefix[input_index - 1]
            };
        let input_shape = shape(input)?;
        let mut remainder = index;
        let mut source = 0usize;
        let mut stride = 1usize;
        for (axis, &extent) in output.iter().enumerate() {
            let coordinate = remainder % extent;
            remainder /= extent;
            let coordinate = if axis == dimension { local } else { coordinate };
            source = source
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("concatenation index overflow"))?,
                )
                .ok_or_else(|| invalid("concatenation index overflow"))?;
            stride = stride
                .checked_mul(input_shape.get(axis).copied().unwrap_or(1))
                .ok_or_else(|| invalid("concatenation stride overflow"))?;
        }
        field(input, name, source)
    };
    let result = match first {
        Value::Struct(value) => {
            construct_generated_structure(runtime, value.field_names(), output, element)
        }
        Value::StructArray(value) => {
            construct_generated_structure(runtime, value.field_names(), output, element)
        }
        _ => Err(invalid("structure concatenation requires structure inputs")),
    }?;
    Ok(raw(result))
}
fn shape(value: &Value) -> Result<&[usize], RuntimeError> {
    match value {
        Value::Struct(_) => Ok(&[1, 1]),
        Value::StructArray(value) => Ok(value.shape()),
        _ => Err(invalid("structure concatenation requires structure inputs")),
    }
}
fn field<'a>(value: &'a Value, name: &str, index: usize) -> Result<&'a Value, RuntimeError> {
    let field = match value {
        Value::Struct(value) if index == 0 => value.fields.get(name),
        Value::StructArray(value) => value
            .field_values(name)
            .and_then(|values| values.get(index)),
        _ => None,
    };
    field.ok_or_else(|| invalid("structure concatenation field or index is missing"))
}
fn validate_fields(first: &Value, input: &Value) -> Result<(), RuntimeError> {
    let contains = |name: &str| match input {
        Value::Struct(value) => value.fields.contains_key(name),
        Value::StructArray(value) => value.column_owner(name).is_some(),
        _ => false,
    };
    let expected = match first {
        Value::Struct(value) => value.fields.len(),
        Value::StructArray(value) => value.field_names().len(),
        _ => return Err(invalid("structure input expected")),
    };
    let actual = match input {
        Value::Struct(value) => value.fields.len(),
        Value::StructArray(value) => value.field_names().len(),
        _ => return Err(invalid("structure input expected")),
    };
    let valid = match first {
        Value::Struct(value) => value.field_names().all(contains),
        Value::StructArray(value) => value.field_names().all(contains),
        _ => false,
    };
    if actual != expected || !valid {
        return Err(invalid("structure concatenation requires one field set"));
    }
    Ok(())
}
