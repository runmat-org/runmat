//! Typed post-parse facts for the direct host-string comparison subroutes.
//! Other text carriers and gathered controls never receive this certificate.

use runmat_value::{NumericDType, Value};

use crate::dispatcher::direct_call_observation::HostCallDomainFacts;

#[derive(serde::Serialize)]
pub(crate) enum ComparisonMode {
    CaseInsensitive,
    PrefixSensitive,
    PrefixInsensitive,
}

#[derive(serde::Serialize)]
pub(crate) enum PrefixCarrier {
    Integer(NumericDType),
    Double,
    Logical,
}

impl PrefixCarrier {
    pub(crate) fn immediate(value: &Value) -> Option<Self> {
        match value {
            Value::Int(value) => Some(Self::Integer(value.numeric_dtype())),
            Value::Num(_) => Some(Self::Double),
            Value::Bool(_) => Some(Self::Logical),
            _ => None,
        }
    }
}

#[derive(serde::Serialize)]
pub(crate) struct PrefixFacts {
    pub(crate) limit_chars: usize,
    pub(crate) carrier: PrefixCarrier,
}

#[derive(serde::Serialize)]
pub(crate) struct ScalarComparisonWorkload {
    pub(crate) left_utf8_bytes: usize,
    pub(crate) right_utf8_bytes: usize,
    pub(crate) mode: ComparisonMode,
    pub(crate) prefix: Option<PrefixFacts>,
    pub(crate) requested_outputs: Option<usize>,
}

impl HostCallDomainFacts for ScalarComparisonWorkload {
    const ROUTE_ID: &'static [u8] = b"runmat.string-compare.host-string-scalar.v1";

    fn requested_outputs(&self) -> Option<usize> {
        self.requested_outputs
    }
}

pub(crate) fn host_string_lengths(a: &Value, b: &Value) -> Option<(usize, usize)> {
    match (a, b) {
        (Value::String(left), Value::String(right)) => {
            Some((left.as_str()?.len(), right.as_str()?.len()))
        }
        _ => None,
    }
}
