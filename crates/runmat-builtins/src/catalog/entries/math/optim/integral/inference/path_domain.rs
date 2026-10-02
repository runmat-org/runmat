use super::super::IntegralOptionName;
use crate::catalog::inference::literal_text;
use runmat_types::{CallRequest, NumericDomain, NumericFact, ValueFact, ValueKindFact};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum PathDomain {
    Real,
    Complex,
    Unknown,
}

impl PathDomain {
    fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Complex, _) | (_, Self::Complex) => Self::Complex,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            _ => Self::Real,
        }
    }
}

fn value_domain(value: &ValueFact) -> PathDomain {
    match value.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => PathDomain::Complex,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => PathDomain::Real,
        _ => PathDomain::Unknown,
    }
}

pub(in super::super) fn path_domain(request: &CallRequest) -> PathDomain {
    let bounds = request
        .arguments
        .get(1..3)
        .map_or(PathDomain::Unknown, |bounds| {
            value_domain(&bounds[0]).combine(value_domain(&bounds[1]))
        });
    let rest = request.arguments.len().saturating_sub(3);
    if rest == 1 {
        return bounds.combine(PathDomain::Unknown);
    }
    // Runtime option parsing is last-value-wins. Keep the bounds separate so
    // an earlier complex Waypoints value cannot outlive a later real one.
    let mut waypoints = None;
    for index in (3..request.arguments.len()).step_by(2) {
        match request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
            .and_then(|name| IntegralOptionName::parse(&name))
        {
            Some(IntegralOptionName::Waypoints) => {
                waypoints = Some(
                    request
                        .arguments
                        .get(index + 1)
                        .map_or(PathDomain::Unknown, value_domain),
                );
            }
            Some(_) => {}
            None => waypoints = Some(PathDomain::Unknown),
        }
    }
    bounds.combine(waypoints.unwrap_or(PathDomain::Real))
}

pub(super) fn result_domain(callback: NumericDomain, path: PathDomain) -> Option<NumericDomain> {
    match (callback, path) {
        (NumericDomain::Complex, _) | (_, PathDomain::Complex) => Some(NumericDomain::Complex),
        (NumericDomain::Real, PathDomain::Real) => Some(NumericDomain::Real),
        _ => None,
    }
}
