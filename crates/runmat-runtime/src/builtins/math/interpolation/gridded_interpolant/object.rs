use super::*;

pub(super) fn spec_to_object(
    runtime: Option<&crate::context::RuntimeContext>,
    spec: InterpolantSpec,
) -> BuiltinResult<crate::execution::RootedValue> {
    let grid_cells = spec
        .grid_vectors
        .iter()
        .map(|grid| {
            Tensor::new(grid.clone(), vec![grid.len(), 1])
                .map(Value::Tensor)
                .map_err(internal)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let fields = [
        (
            GRID_VECTORS,
            Value::Cell(
                CellArray::new_with_shape(grid_cells, vec![1, spec.grid_vectors.len()])
                    .map_err(internal)?,
            ),
        ),
        (VALUES, Value::Tensor(spec.values)),
        (
            METHOD,
            Value::string(interp_method_name(spec.method).to_string()),
        ),
        (
            EXTRAPOLATION_METHOD,
            Value::string(extrap_method_name(spec.extrapolation).to_string()),
        ),
    ];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        CLASS_IDENTITY.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

pub(super) fn object_to_spec(value: &Value) -> BuiltinResult<InterpolantSpec> {
    let Value::Object(object) = value else {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    };
    if !object.is_class(CLASS_IDENTITY) {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    }
    let grid_vectors = match object.properties().get(GRID_VECTORS) {
        Some(Value::Cell(cell)) => {
            ensure_integer_extension(
                &Value::Cell(cell.clone()),
                &GRIDDED_INTEGER_GRID_EXTENSION,
                "sample grid",
            )?;
            cell.data()
                .iter()
                .map(|value| numeric_vector(value, GRID_VECTORS))
                .collect::<BuiltinResult<Vec<_>>>()?
        }
        _ => return Err(internal("object is missing GridVectors")),
    };
    let values = match object.properties().get(VALUES) {
        Some(Value::Tensor(tensor)) => {
            let value = Value::Tensor(tensor.clone());
            ensure_integer_extension(&value, &GRIDDED_INTEGER_VALUES_EXTENSION, "sample values")?;
            numeric_tensor(value, VALUES)?
        }
        _ => return Err(internal("object is missing Values")),
    };
    let method = match object
        .properties()
        .get(METHOD)
        .map(text_from_value)
        .transpose()?
        .flatten()
    {
        Some(name) => parse_interp_method(MethodKeyword::parse(&name).ok_or_else(|| {
            invalid(format!(
                "unknown interpolation method '{}'",
                name.to_ascii_lowercase()
            ))
        })?)?,
        None => return Err(internal("object is missing Method")),
    };
    let extrapolation = match object
        .properties()
        .get(EXTRAPOLATION_METHOD)
        .map(text_from_value)
        .transpose()?
        .flatten()
    {
        Some(name) => parse_extrap_method(MethodKeyword::parse(&name).ok_or_else(|| {
            invalid(format!(
                "unknown extrapolation method '{}'",
                name.to_ascii_lowercase()
            ))
        })?)?,
        None => return Err(internal("object is missing ExtrapolationMethod")),
    };
    validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
    Ok(InterpolantSpec {
        grid_vectors,
        values,
        method,
        extrapolation,
    })
}

pub(super) fn gridded_member(obj: Value, payload: Value) -> BuiltinResult<Value> {
    let field = text_from_value(&payload)?.ok_or_else(|| invalid("property name must be text"))?;
    let Value::Object(object) = obj else {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    };
    if !object.is_class(CLASS_IDENTITY) {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    }
    object
        .properties()
        .get(&field)
        .cloned()
        .ok_or_else(|| invalid(format!("unknown griddedInterpolant property '{field}'")))
}

pub(super) fn payload_to_args(
    payload: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    match payload {
        Value::Cell(cell) => {
            let values = match runtime {
                Some(runtime) => crate::execution::RootedValues::from_output_storage_in_context(
                    cell.into_parts().0,
                    runtime,
                )?,
                None => {
                    use crate::context::host_allocation::{
                        HostAllocationPolicy, HostAllocationReceipt,
                    };
                    if HostAllocationReceipt::policy(None).map_err(|source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            None,
                            "RunMat:griddedInterpolant:Allocation",
                            "query argument policy is unavailable",
                            source,
                        )
                    })? != HostAllocationPolicy::Unbounded
                    {
                        return Err(invalid(
                            "query arguments require their selected runtime context",
                        ));
                    }
                    crate::execution::RootedValues::from_output_storage(cell.into_parts().0)
                        .map_err(|source| source.into_runtime_error())?
                }
            };
            Ok(values.into())
        }
        other => Err(invalid(format!(
            "indexing payload must be a cell array, got {other:?}"
        ))),
    }
}
