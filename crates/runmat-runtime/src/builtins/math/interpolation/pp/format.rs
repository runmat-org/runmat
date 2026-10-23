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
    pp.validate_layout(name)?;
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
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert(
        "form",
        Value::string(PiecewiseForm::Polynomial.keyword().to_string()),
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
    Ok(Value::Struct(st.finish()))
}

pub async fn pp_from_value(value: Value, name: &'static str) -> BuiltinResult<PiecewisePolynomial> {
    let value = materialize_if_resident(value).await?;
    let Value::Struct(st) = value else {
        return Err(interp_error(
            name,
            format!("{name}: first argument must be a pp struct"),
        ));
    };
    let form = st
        .fields
        .get("form")
        .map(keyword_of)
        .transpose()?
        .flatten()
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing form")))?;
    if PiecewiseForm::from_keyword(&form) != Some(PiecewiseForm::Polynomial) {
        return Err(interp_error(
            name,
            format!("{name}: struct form must be 'pp'"),
        ));
    }
    let breaks_value = st
        .fields
        .get("breaks")
        .cloned()
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing breaks")))?;
    let breaks_host = materialize_if_resident(breaks_value).await?;
    ensure_exact_integer_field(&breaks_host, name, "breaks")?;
    let breaks = vector_from_value(breaks_host, "breaks", name).await?;
    validate_breaks(&breaks, name)?;

    let coefs_value = st
        .fields
        .get("coefs")
        .cloned()
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing coefs")))?;
    let coefs_host = materialize_if_resident(coefs_value).await?;
    ensure_exact_integer_field(&coefs_host, name, "coefs")?;
    let coefs_tensor = tensor::value_into_tensor_for(name, coefs_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;

    let pieces = scalar_field(&st, "pieces", name).await?;
    let order = scalar_field(&st, "order", name).await?;
    let component_shape = component_shape_field(&st, name).await?;
    let dim = PiecewisePolynomial::component_dim(&component_shape, name)?;
    let rows = pieces
        .checked_mul(dim)
        .ok_or_else(|| interp_error(name, format!("{name}: coefficient shape overflow")))?;
    if coefs_tensor.shape() != [rows, order] {
        return Err(interp_error(
            name,
            format!("{name}: pp coefficients must be a {rows}-by-{order} matrix"),
        ));
    }
    let coefs = tensor::tensor_into_values_f64(coefs_tensor);
    PiecewisePolynomial::from_parts(breaks, coefs, pieces, order, dim, component_shape, name)
}

async fn component_shape_field(st: &StructValue, name: &'static str) -> BuiltinResult<Vec<usize>> {
    let value = st
        .fields
        .get("dim")
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing dim")))?;
    let gathered = materialize_if_resident(value.clone()).await?;
    ensure_exact_integer_field(&gathered, name, "dim")?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    if !is_vector_shape(tensor.shape()) {
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
    ensure_exact_integer_field(&gathered, name, field)?;
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

fn ensure_exact_integer_field(value: &Value, name: &'static str, field: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        Ok(())
    } else {
        Err(interp_error(
            name,
            format!("{name}: pp field {field} must be exactly representable as double"),
        ))
    }
}
