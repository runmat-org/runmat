use super::ExtremaKind;
use crate::BuiltinResult;
use runmat_value::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::reduction) enum ExtremaDemand {
    ValuesOnly,
    ValuesAndIndices,
}

#[derive(Debug)]
pub(in crate::builtins::math::reduction) enum ExtremaResult {
    Values(Value),
    Pair { values: Value, indices: Value },
}

impl ExtremaResult {
    pub(in crate::builtins::math::reduction) fn from_pair(
        demand: ExtremaDemand,
        values: Value,
        indices: Value,
    ) -> Self {
        match demand {
            ExtremaDemand::ValuesOnly => Self::Values(values),
            ExtremaDemand::ValuesAndIndices => Self::Pair { values, indices },
        }
    }
    pub(in crate::builtins::math::reduction) fn into_value(self) -> Value {
        match self {
            Self::Values(values) | Self::Pair { values, .. } => values,
        }
    }
    pub(in crate::builtins::math::reduction) fn into_pair(
        self,
        kind: ExtremaKind,
    ) -> BuiltinResult<(Value, Value)> {
        match self {
            Self::Pair { values, indices } => Ok((values, indices)),
            Self::Values(_) => {
                Err(kind.contract_error("internal evaluation omitted requested indices"))
            }
        }
    }
}
