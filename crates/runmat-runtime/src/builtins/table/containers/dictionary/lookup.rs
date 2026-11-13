use super::allocation::{add, allocation, exact_bytes, exact_error, Indices, Work};
use super::stream::{Element, Query};
use super::*;
use crate::builtins::common::uniform_scalar_output::{UniformScalarCollector, UniformScalarError};
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use runmat_value::{ExactArrayStorage, OutputListStorage, SharedShape, StringElement};

fn uniform_error(error: UniformScalarError) -> crate::RuntimeError {
    match error.into_failure() {
        Err(error) => error,
        Ok(error) => invalid_variable(format!("dictionary: inconsistent scalar output: {error:?}")),
    }
}
#[derive(Clone, Copy, PartialEq)]
enum Class {
    Numeric(NumericDType),
    Logical,
}
fn class(value: &Value) -> Option<Class> {
    Some(match value {
        Value::Num(_) => Class::Numeric(NumericDType::F64),
        Value::Int(value) => Class::Numeric(NumericScalar::from(value.clone()).numeric_dtype()),
        Value::Bool(_) => Class::Logical,
        Value::Tensor(value) if value.len() == 1 && value.numeric_dtype() == NumericDType::F32 => {
            Class::Numeric(NumericDType::F32)
        }
        _ => return None,
    })
}

pub(super) fn lookup(
    runtime: Option<&RuntimeContext>,
    object: &ObjectInstance,
    query: Query<'_>,
) -> BuiltinResult<RootedValue> {
    let _query_roots = query.roots(runtime)?;
    let _source_root = rooted(runtime, Value::Object(object.clone()))?;
    let count = query.count()?;
    let shape = query.shape(count)?;
    let keys = dictionary_cells(object, "Keys")?;
    let values = dictionary_cells(object, "Values")?;
    if keys.data().len() != values.data().len() {
        return Err(invalid_variable("dictionary: key/value storage diverged"));
    }
    let prototype = keys.data().first().map(Element::Borrowed);
    let mut indices = Indices::new(runtime, count)?;
    for index in 0..count {
        indices.check()?;
        let key = query.at(index)?.normalize(prototype)?;
        let mut found = None;
        for (position, value) in keys.data().iter().enumerate() {
            indices.check()?;
            if key.equal(Element::Borrowed(value))? {
                found = Some(position);
                break;
            }
        }
        let found = found.ok_or_else(|| invalid_index("dictionary: key not found"))?;
        indices.as_mut_slice()[index] = found;
    }
    let at = |index| &values.data()[indices.as_slice()[index]];
    if count == 1 {
        let output = rooted(runtime, at(0).clone())?;
        indices.check()?;
        return Ok(output);
    }
    let output = output(runtime, count, shape.as_slice(), at, || indices.check())?;
    let output = rooted(runtime, output)?;
    indices.check()?;
    Ok(output)
}

// Classification precedes construction. A quota/cancellation error is never
// reinterpreted as heterogeneous data and cannot trigger Cell fallback.
pub(super) fn output<'a>(
    runtime: Option<&RuntimeContext>,
    count: usize,
    shape: &[usize],
    at: impl Fn(usize) -> &'a Value,
    check: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Value> {
    let first = if count == 0 { None } else { class(at(0)) };
    let mut uniform = first.is_some();
    let mut strings = count != 0;
    for index in 0..count {
        check()?;
        uniform &= class(at(index)) == first;
        strings &= matches!(at(index), Value::String(_));
    }
    if uniform {
        let mut collector = match runtime {
            Some(runtime) => {
                UniformScalarCollector::in_context(runtime, count).map_err(uniform_error)?
            }
            None => UniformScalarCollector::new(),
        };
        for index in 0..count {
            check()?;
            collector.push(at(index)).map_err(uniform_error)?;
        }
        let result = collector.finish(shape).map_err(uniform_error)?;
        check()?;
        return Ok(result);
    }
    // Query values are MATLAB-linear. Cell payloads retain row-major 2-D
    // pages, so initialize each final physical slot from its linear source.
    // This is an index projection, not another collection or allocation.
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let page = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid_variable("dictionary: cell page size overflow"))?;
    let cell_at = |physical: usize| {
        let linear = if page == 0 {
            physical
        } else {
            let within = physical % page;
            physical / page * page + within / cols + within % cols * rows
        };
        at(linear)
    };
    let raw_list = if strings {
        ExactArrayStorage::<StringElement>::requested_bytes(count)
    } else {
        OutputListStorage::requested_bytes(count)
    }
    .map_err(|cause| allocation(runtime, cause.into()))?;
    let raw_shape = SharedShape::requested_bytes_for_rank(shape.len())
        .map_err(|cause| allocation(runtime, cause.into()))?;
    let mut retained = exact_bytes(raw_list, usize::from(count != 0), runtime)?;
    retained = add(retained, exact_bytes(raw_shape, 1, runtime)?, runtime)?;
    if strings {
        for index in 0..count {
            check()?;
            let Value::String(text) = at(index) else {
                unreachable!("preflight strings")
            };
            if let Some(text) = text.text_ref().filter(|text| !text.has_retained_claim()) {
                retained = add(
                    retained,
                    exact_bytes(
                        StringElement::requested_bytes_for_capacity(text.len())
                            .map_err(|cause| allocation(runtime, cause.into()))?,
                        usize::from(!text.is_empty()),
                        runtime,
                    )?,
                    runtime,
                )?;
            }
        }
    }
    let work = Work::new(runtime, 0, retained)?;
    let output = work.construct(|permit| {
        let Some(permit) = permit else {
            if strings {
                let data = (0..count)
                    .map(|index| {
                        let Value::String(text) = at(index) else {
                            unreachable!("preflight strings")
                        };
                        text.clone()
                    })
                    .collect();
                return StringArray::from_elements_unbounded(data, shape.to_vec())
                    .map(Value::StringArray)
                    .map_err(invalid_variable);
            }
            return crate::make_cell_with_shape(
                (0..count).map(|index| cell_at(index).clone()).collect(),
                shape.to_vec(),
            )
            .map_err(invalid_variable);
        };
        let shape = SharedShape::new_admitted_from_slice(shape, permit)
            .map_err(|cause| allocation(runtime, cause.into()))?;
        if strings {
            let data = ExactArrayStorage::try_initialize(count, permit, |index| {
                check()?;
                work.check()?;
                let Value::String(text) = at(index) else {
                    unreachable!("preflight strings")
                };
                if text.has_retained_claim() {
                    Ok(text.clone())
                } else {
                    text.try_copy_exact_with(permit, |_| work.check())
                        .map_err(|cause| exact_error(runtime, cause))
                }
            })
            .map_err(|cause| exact_error(runtime, cause))?;
            StringArray::from_admitted(data, shape)
                .map(Value::StringArray)
                .map_err(invalid_variable)
        } else {
            let data = OutputListStorage::try_initialize(count, permit, |index| {
                check()?;
                work.check()?;
                Ok::<_, crate::RuntimeError>(cell_at(index).clone())
            })
            .map_err(|cause| exact_error(runtime, cause))?;
            CellArray::from_admitted(data, shape)
                .map(Value::Cell)
                .map_err(invalid_variable)
        }
    })?;
    check()?;
    Ok(output)
}

pub(super) fn rooted(runtime: Option<&RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => RootedValue::new(value).map_err(|error| error.into_runtime_error()),
    }
}
