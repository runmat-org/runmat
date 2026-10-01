use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PiecewiseForm {
    Polynomial,
}

impl PiecewiseForm {
    fn from_keyword(value: &str) -> Option<Self> {
        match value {
            "pp" => Some(Self::Polynomial),
            _ => None,
        }
    }

    const fn keyword(self) -> &'static str {
        match self {
            Self::Polynomial => "pp",
        }
    }
}

pub fn pp_to_value(pp: PiecewisePolynomial, name: &'static str) -> BuiltinResult<Value> {
    let PiecewisePolynomial {
        breaks,
        coefs,
        pieces,
        order,
        dim,
        component_shape,
    } = pp;
    let break_count = breaks.len();
    let breaks = Tensor::new(breaks, vec![1, break_count])
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let rows = pieces
        .checked_mul(dim)
        .ok_or_else(|| interp_error(name, format!("{name}: coefficient shape overflow")))?;
    let coefs = Tensor::new(coefs, vec![rows, order])
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let mut st = StructValue::new();
    st.insert(
        "form",
        Value::String(PiecewiseForm::Polynomial.keyword().to_string()),
    );
    st.insert("breaks", Value::Tensor(breaks));
    st.insert("coefs", Value::Tensor(coefs));
    st.insert("pieces", Value::Num(pieces as f64));
    st.insert("order", Value::Num(order as f64));
    let dim = if component_shape.len() > 1 {
        Value::Tensor(
            Tensor::new(
                component_shape
                    .iter()
                    .map(|&dimension| dimension as f64)
                    .collect(),
                vec![1, component_shape.len()],
            )
            .map_err(|err| interp_error(name, format!("{name}: {err}")))?,
        )
    } else {
        Value::Num(dim as f64)
    };
    st.insert("dim", dim);
    Ok(Value::Struct(st))
}

pub async fn pp_from_value(value: Value, name: &'static str) -> BuiltinResult<PiecewisePolynomial> {
    let value = materialize_if_resident(value).await?;
    let Value::Struct(mut st) = value else {
        return Err(interp_error(
            name,
            format!("{name}: first argument must be a pp struct"),
        ));
    };
    let form = st
        .fields
        .get("form")
        .and_then(keyword_of)
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing form")))?;
    if PiecewiseForm::from_keyword(&form) != Some(PiecewiseForm::Polynomial) {
        return Err(interp_error(
            name,
            format!("{name}: struct form must be 'pp'"),
        ));
    }
    let breaks_value = st
        .fields
        .remove("breaks")
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing breaks")))?;
    let breaks = vector_from_value(breaks_value, "breaks", name).await?;
    validate_breaks(&breaks, name)?;

    let coefs_value = st
        .fields
        .remove("coefs")
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing coefs")))?;
    let coefs_host = materialize_if_resident(coefs_value).await?;
    let coefs_tensor = tensor::value_into_tensor_for(name, coefs_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;

    let pieces = scalar_field(&st, "pieces", name).await?;
    let order = scalar_field(&st, "order", name).await?;
    let component_shape = component_shape_field(&st, name).await?;
    let dim = component_shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| interp_error(name, format!("{name}: malformed pp dimensions")))?;
    if pieces.checked_add(1) != Some(breaks.len()) {
        return Err(interp_error(name, format!("{name}: malformed pp breaks")));
    }
    if order == 0 || dim == 0 {
        return Err(interp_error(
            name,
            format!("{name}: malformed pp dimensions"),
        ));
    }
    let coefs_len = tensor_len(&coefs_tensor);
    if pieces
        .checked_mul(dim)
        .and_then(|count| count.checked_mul(order))
        != Some(coefs_len)
    {
        return Err(interp_error(
            name,
            format!("{name}: malformed pp coefficients"),
        ));
    }
    let coefs = tensor::tensor_into_values_f64(coefs_tensor);
    Ok(PiecewisePolynomial {
        breaks,
        coefs,
        pieces,
        order,
        dim,
        component_shape,
    })
}

async fn component_shape_field(st: &StructValue, name: &'static str) -> BuiltinResult<Vec<usize>> {
    let value = st
        .fields
        .get("dim")
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing dim")))?;
    let gathered = materialize_if_resident(value.clone()).await?;
    ensure_exact_integer_metadata(&gathered, name, "dim")?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    if !is_vector_shape(&tensor.shape) {
        return Err(interp_error(
            name,
            format!("{name}: malformed pp dimensions"),
        ));
    }
    let dimensions = tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .map(|raw| {
            if !raw.is_finite() || raw < 1.0 || raw.fract() != 0.0 {
                return Err(interp_error(
                    name,
                    format!("{name}: malformed pp dimensions"),
                ));
            }
            let dimension = raw as u128;
            if dimension > usize::MAX as u128 || dimension as f64 != raw {
                return Err(interp_error(
                    name,
                    format!("{name}: malformed pp dimensions"),
                ));
            }
            Ok(dimension as usize)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if dimensions.is_empty() {
        return Err(interp_error(
            name,
            format!("{name}: malformed pp dimensions"),
        ));
    }
    Ok(if dimensions == [1] {
        Vec::new()
    } else {
        dimensions
    })
}

async fn scalar_field(st: &StructValue, field: &str, name: &'static str) -> BuiltinResult<usize> {
    let value = st
        .fields
        .get(field)
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing {field}")))?;
    let gathered = materialize_if_resident(value.clone()).await?;
    ensure_exact_integer_metadata(&gathered, name, field)?;
    let Some(raw) = tensor::scalar_f64_from_value_async(&gathered)
        .await
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?
    else {
        return Err(interp_error(
            name,
            format!("{name}: pp field {field} must be scalar"),
        ));
    };
    if !raw.is_finite() || raw < 1.0 || raw.fract() != 0.0 {
        return Err(interp_error(
            name,
            format!("{name}: pp field {field} must be a positive integer"),
        ));
    }
    let count = raw as u128;
    if count > usize::MAX as u128 || count as f64 != raw {
        return Err(interp_error(
            name,
            format!("{name}: pp field {field} must be a positive integer"),
        ));
    }
    Ok(count as usize)
}

fn ensure_exact_integer_metadata(
    value: &Value,
    name: &'static str,
    field: &str,
) -> BuiltinResult<()> {
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    let representable = match value {
        Value::Int(integer) => exact(integer),
        Value::Tensor(tensor) => tensor.integer_storage().is_none_or(|storage| {
            (0..storage.len()).all(|index| {
                storage
                    .value_at(index)
                    .is_some_and(|integer| exact(&integer))
            })
        }),
        _ => true,
    };
    if representable {
        Ok(())
    } else {
        Err(interp_error(
            name,
            format!("{name}: pp field {field} must be exactly representable as double"),
        ))
    }
}
