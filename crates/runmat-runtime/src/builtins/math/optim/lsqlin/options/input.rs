//! Preserve the public owned options path; only actual borrowed compiled
//! consumers clone already admitted scalar controls after complete preflight.
use super::*;
use runmat_value::StructValue;
pub(super) enum Input<'a> {
    Owned(Option<StructValue>),
    Borrowed(Option<&'a StructValue>),
}
impl<'a> Input<'a> {
    pub fn owned(value: Value) -> BuiltinResult<Self> {
        Ok(Self::Owned(match value {
            Value::Struct(value) => Some(value),
            Value::Tensor(value) if value.is_empty() => None,
            _ => return Err(invalid("options must be a scalar struct or []")),
        }))
    }
    pub fn borrowed(value: Option<&'a Value>) -> BuiltinResult<Self> {
        Ok(Self::Borrowed(match value {
            Some(Value::Struct(value)) => Some(value),
            None => None,
            Some(Value::Tensor(value)) if value.is_empty() => None,
            _ => return Err(invalid("options must be a scalar struct or []")),
        }))
    }
    pub fn as_ref(&self) -> Option<&StructValue> {
        match self {
            Self::Owned(value) => value.as_ref(),
            Self::Borrowed(value) => *value,
        }
    }
    pub fn into_active_owned(
        self,
        algorithm: OptimAlgorithm,
        method: Option<super::reflective::Method>,
    ) -> Option<StructValue> {
        let active = |name: &str| {
            name.eq_ignore_ascii_case("Solver")
                || Field::parse_for_solver(name, OptimSolver::Lsqlin).is_some_and(|field| {
                    super::reflective::selected_active(field, algorithm, method)
                })
        };
        match self {
            Self::Owned(value) => value.map(|value| {
                StructValue::from_fields_unbounded(
                    value
                        .into_fields_unbounded()
                        .filter(|(name, _)| active(name)),
                )
            }),
            Self::Borrowed(value) => value.map(|value| {
                StructValue::from_fields_unbounded(
                    value
                        .fields
                        .iter()
                        .filter(|(name, _)| active(name))
                        .map(|(name, value)| (name.to_owned(), value.clone())),
                )
            }),
        }
    }
}
