//! Typed membership atoms and equality.

use super::cell_leaves::CellLeaves;
use super::{
    chars_to_string, host_value, integer_f64_order, invalid_argument_error, numeric_class_identity,
    tensor, BuiltinResult, Ordering, RuntimeError, SparseTensor, ValidationAtom, Value,
};
mod admission;
#[cfg(test)]
mod tests;

pub fn value_is_member(value: &Value, set: &Value) -> Result<bool, RuntimeError> {
    let values = atoms(value)?;
    let allowed = atoms(set)?;
    value_is_member_atoms_inner(&values, &allowed)
}

pub async fn value_is_member_async(value: &Value, set: &Value) -> BuiltinResult<bool> {
    admission::shape_and_kind(value)?;
    admission::shape_and_kind(set)?;
    ensure_member_class_compatibility(value, set)?;
    admission::resident_metadata(value)?;
    admission::resident_metadata(set)?;
    let values = host_atoms(value).await?;
    let allowed = host_atoms(set).await?;
    value_is_member_atoms_inner(&values, &allowed).map_err(Into::into)
}

fn ensure_member_class_compatibility(value: &Value, set: &Value) -> BuiltinResult<()> {
    let Some(value_class) = numeric_class_identity(value) else {
        return Ok(());
    };
    let Some(set_class) = numeric_class_identity(set) else {
        return Ok(());
    };
    if value_class == set_class
        || value_class.is(runmat_types::standard::DOUBLE)
        || set_class.is(runmat_types::standard::DOUBLE)
    {
        return Ok(());
    }
    Err(invalid_argument_error(
        "mustBeMember",
        "unlike nondouble numeric inputs must have the same class",
    )
    .into())
}

pub fn value_is_member_atoms(
    value: &Value,
    allowed: &[ValidationAtom],
) -> Result<bool, RuntimeError> {
    let values = atoms(value)?;
    value_is_member_atoms_inner(&values, allowed)
}

pub async fn value_is_member_atoms_async(
    value: &Value,
    allowed: &[ValidationAtom],
) -> BuiltinResult<bool> {
    admission::shape_and_kind(value)?;
    admission::resident_metadata(value)?;
    value_is_member_atoms_inner(&host_atoms(value).await?, allowed).map_err(Into::into)
}

async fn host_atoms(value: &Value) -> BuiltinResult<Vec<ValidationAtom>> {
    let mut result = Vec::new();
    for leaf in CellLeaves::new(value) {
        result.extend(atoms(&host_value(leaf).await?)?);
    }
    Ok(result)
}

fn value_is_member_atoms_inner(
    values: &[ValidationAtom],
    allowed: &[ValidationAtom],
) -> Result<bool, RuntimeError> {
    Ok(values
        .iter()
        .all(|value| allowed.iter().any(|allowed| atom_eq(value, allowed))))
}

pub fn atoms(value: &Value) -> Result<Vec<ValidationAtom>, RuntimeError> {
    match value {
        Value::Num(v) => Ok(vec![ValidationAtom::Number(*v)]),
        Value::Int(v) => Ok(vec![ValidationAtom::Integer(v.clone())]),
        Value::Bool(v) => Ok(vec![ValidationAtom::Bool(*v)]),
        Value::Complex(re, im) => Ok(vec![ValidationAtom::ComplexNumber(*re, *im)]),
        Value::String(s) => Ok(vec![s
            .as_str()
            .map(|text| ValidationAtom::Text(text.to_owned()))
            .unwrap_or(ValidationAtom::MissingString)]),
        Value::CharArray(c) if c.rows == 1 => Ok(vec![ValidationAtom::Text(chars_to_string(c)?)]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return Ok(storage
                    .exact_values()
                    .into_iter()
                    .map(ValidationAtom::Integer)
                    .collect());
            }
            Ok(tensor::tensor_values_f64_cow(t)
                .iter()
                .copied()
                .map(ValidationAtom::Number)
                .collect())
        }
        Value::ComplexTensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return Ok((0..storage.len())
                    .map(|index| {
                        ValidationAtom::ComplexInteger(
                            storage
                                .real()
                                .value_at(index)
                                .expect("validated real storage"),
                            storage
                                .imag()
                                .value_at(index)
                                .expect("validated imaginary storage"),
                        )
                    })
                    .collect());
            }
            Ok(t.materialize_f64()
                .iter()
                .map(|(re, im)| ValidationAtom::ComplexNumber(*re, *im))
                .collect())
        }
        Value::SparseTensor(t) => sparse_atoms(t),
        Value::LogicalArray(a) => Ok(a
            .data
            .iter()
            .map(|v| ValidationAtom::Bool(*v != 0))
            .collect()),
        Value::StringArray(s) => Ok(s
            .data()
            .iter()
            .map(|text| {
                text.as_str()
                    .map(|text| ValidationAtom::Text(text.to_owned()))
                    .unwrap_or(ValidationAtom::MissingString)
            })
            .collect()),
        Value::Cell(c) => {
            let mut out = Vec::new();
            for entry in c.data() {
                out.extend(atoms(entry)?);
            }
            Ok(out)
        }
        _ => Err(invalid_argument_error(
            "mustBeMember",
            "unsupported member value type",
        )),
    }
}

fn sparse_atoms(t: &SparseTensor) -> Result<Vec<ValidationAtom>, RuntimeError> {
    let numel = t.rows.saturating_mul(t.cols);
    let mut out = Vec::with_capacity(numel.min(t.nnz().saturating_add(1)));
    if let Some(storage) = t.integer_storage() {
        out.extend(
            storage
                .exact_values()
                .into_iter()
                .map(ValidationAtom::Integer),
        );
        if storage.len() < numel {
            let zero = storage
                .zeros_like(1)
                .value_at(0)
                .expect("one-element zero storage");
            out.push(ValidationAtom::Integer(zero));
        }
        return Ok(out);
    }
    if t.is_complex() {
        let values = t.materialize_complex_f64().expect("complex sparse storage");
        out.extend(
            values
                .iter()
                .map(|value| ValidationAtom::ComplexNumber(value.0, value.1)),
        );
        if values.len() < numel {
            out.push(ValidationAtom::ComplexNumber(0.0, 0.0));
        }
        return Ok(out);
    }
    let values = t.materialize_f64();
    out.extend(values.iter().copied().map(ValidationAtom::Number));
    if values.len() < numel {
        out.push(ValidationAtom::Number(0.0));
    }
    Ok(out)
}

fn atom_eq(left: &ValidationAtom, right: &ValidationAtom) -> bool {
    match (left, right) {
        (ValidationAtom::Number(a), ValidationAtom::Number(b)) => a == b,
        (ValidationAtom::Integer(a), ValidationAtom::Integer(b)) => a == b,
        (ValidationAtom::Integer(a), ValidationAtom::Number(b))
        | (ValidationAtom::Number(b), ValidationAtom::Integer(a)) => {
            integer_f64_order(a.clone(), *b) == Some(Ordering::Equal)
        }
        (ValidationAtom::ComplexNumber(ar, ai), ValidationAtom::ComplexNumber(br, bi)) => {
            ar == br && ai == bi
        }
        (ValidationAtom::ComplexInteger(ar, ai), ValidationAtom::ComplexInteger(br, bi)) => {
            ar == br && ai == bi
        }
        (ValidationAtom::ComplexInteger(ar, ai), ValidationAtom::ComplexNumber(br, bi))
        | (ValidationAtom::ComplexNumber(br, bi), ValidationAtom::ComplexInteger(ar, ai)) => {
            integer_f64_order(ar.clone(), *br) == Some(Ordering::Equal)
                && integer_f64_order(ai.clone(), *bi) == Some(Ordering::Equal)
        }
        (ValidationAtom::ComplexNumber(re, im), ValidationAtom::Number(value))
        | (ValidationAtom::Number(value), ValidationAtom::ComplexNumber(re, im)) => {
            *im == 0.0 && re == value
        }
        (ValidationAtom::ComplexInteger(re, im), ValidationAtom::Integer(value))
        | (ValidationAtom::Integer(value), ValidationAtom::ComplexInteger(re, im)) => {
            im.is_zero() && re == value
        }
        (ValidationAtom::ComplexInteger(re, im), ValidationAtom::Number(value))
        | (ValidationAtom::Number(value), ValidationAtom::ComplexInteger(re, im)) => {
            im.is_zero() && integer_f64_order(re.clone(), *value) == Some(Ordering::Equal)
        }
        (ValidationAtom::ComplexNumber(re, im), ValidationAtom::Integer(value))
        | (ValidationAtom::Integer(value), ValidationAtom::ComplexNumber(re, im)) => {
            *im == 0.0 && integer_f64_order(value.clone(), *re) == Some(Ordering::Equal)
        }
        (ValidationAtom::Text(a), ValidationAtom::Text(b)) => a == b,
        (ValidationAtom::Bool(a), ValidationAtom::Bool(b)) => a == b,
        (ValidationAtom::Bool(value), ValidationAtom::Number(number))
        | (ValidationAtom::Number(number), ValidationAtom::Bool(value)) => {
            *number == f64::from(*value)
        }
        (ValidationAtom::Bool(value), ValidationAtom::Integer(integer))
        | (ValidationAtom::Integer(integer), ValidationAtom::Bool(value)) => {
            integer_f64_order(integer.clone(), f64::from(*value)) == Some(Ordering::Equal)
        }
        _ => false,
    }
}
