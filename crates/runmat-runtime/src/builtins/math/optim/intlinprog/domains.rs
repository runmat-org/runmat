//! Original zero-or-interval domains, distinct from their convex node hulls.
use super::super::integer_constraint::schema::{Property, ValidatedConstraint};
use super::*;
use num_rational::BigRational;
use num_traits::Zero;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Semi {
    pub column: usize,
    /// For semi-integers this is ceil(the supplied strictly positive bound).
    pub lower: f64,
    pub upper: f64,
}
impl Semi {
    pub(in crate::builtins::math::optim) fn from_bounds(
        column: usize,
        lower: f64,
        upper: f64,
        integer: bool,
    ) -> BuiltinResult<Self> {
        if !(lower > 0.0 && lower <= 1e5 && upper > 0.0 && upper <= 1e5) {
            return Err(invalid("semi-continuous and semi-integer bounds must both be strictly positive and at most 1e5"));
        }
        Ok(Self {
            column,
            lower: if integer { lower.ceil() } else { lower },
            upper,
        })
    }
    pub fn contains(&self, value: &BigRational) -> bool {
        value.is_zero() || (value >= &exact(self.lower) && value <= &exact(self.upper))
    }
}
fn exact(value: f64) -> BigRational {
    BigRational::from_float(value).expect("admitted finite semi bound")
}
pub(super) fn validate_input(
    value: &Value,
    runtime: &crate::context::RuntimeContext,
) -> BuiltinResult<Option<ValidatedConstraint>> {
    if let Value::Object(object) = value {
        let admitted = ValidatedConstraint::admit(
            object,
            &crate::execution::value_codec::ValidationWorkspaceOwner::capture(runtime),
        )
        .map_err(|error| {
            super::super::integer_constraint::validation_error(Some(runtime), error)
        })?;
        return Ok(Some(admitted));
    }
    Ok(None)
}

/// Consume the current validated object's indices only after the variable
/// geometry is known. The object owns no hidden/cached solver interpretation.
pub(super) async fn decode(
    value: Value,
    admitted: Option<ValidatedConstraint>,
    n: usize,
    lower: &mut [f64],
    upper: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Vec<usize>, Vec<Semi>)> {
    let mut integers = Vec::new();
    let mut semis = Vec::new();
    if let Some(view) = admitted {
        drop(value);
        for property in Property::ALL {
            let indices = view.indices(property).map_err(invalid)?;
            integers
                .try_reserve(indices.len())
                .map_err(|e| reserve_error(e, "integer indices"))?;
            if property != Property::Integer {
                semis
                    .try_reserve(indices.len())
                    .map_err(|e| reserve_error(e, "semi domains"))?;
            }
            for index in indices.iter() {
                let column = column(index, n)?;
                if property != Property::SemiContinuous {
                    integers.push(column);
                }
                if property != Property::Integer {
                    let l = lower[column];
                    let u = upper[column];
                    semis.push(Semi::from_bounds(
                        column,
                        l,
                        u,
                        property == Property::SemiInteger,
                    )?);
                    // Zero belongs to the original domain regardless of its
                    // positive active lower bound. This is the root LP hull.
                    lower[column] = 0.0;
                }
                cooperation.completed(1).await?;
            }
        }
    } else {
        for index in numeric::vector(value, true, false, cooperation).await? {
            integers
                .try_reserve(1)
                .map_err(|e| reserve_error(e, "integer indices"))?;
            integers.push(column(index, n)?);
            cooperation.completed(1).await?;
        }
    }
    integers.sort_unstable();
    integers.dedup();
    semis.sort_unstable_by_key(|domain| domain.column);
    Ok((integers, semis))
}
fn column(index: f64, n: usize) -> BuiltinResult<usize> {
    if !index.is_finite()
        || index < 1.0
        || index.fract() != 0.0
        || index >= 2f64.powi(usize::BITS as i32)
    {
        return Err(invalid(
            "intcon must contain exact positive variable indices",
        ));
    }
    let column = index as usize - 1;
    if column >= n {
        return Err(invalid("intcon index exceeds the variable count"));
    }
    Ok(column)
}
pub(super) async fn admitted(
    point: &[BigRational],
    semis: &[Semi],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for domain in semis {
        if point
            .get(domain.column)
            .is_none_or(|value| !domain.contains(value))
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
pub(super) async fn exact_split(
    point: &[BigRational],
    semis: &[Semi],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(usize, f64, f64)>> {
    for domain in semis {
        let Some(value) = point.get(domain.column) else {
            return Ok(None);
        };
        // Outside the hull is not a domain disjunction proposal. The node's
        // exact linear verifier is responsible for its convex constraints.
        if value > &BigRational::zero() && value < &exact(domain.lower) {
            return Ok(Some((domain.column, 0.0, domain.lower)));
        }
        cooperation.completed(1).await?;
    }
    Ok(None)
}
pub(super) fn numeric_split(
    point: &[f64],
    semis: &[Semi],
    mut progresses: impl FnMut(usize, f64, f64) -> bool,
) -> Option<(usize, f64, f64)> {
    semis.iter().find_map(|domain| {
        let value = *point.get(domain.column)?;
        (value.is_finite()
            && value > 0.0
            && value < domain.lower
            && progresses(domain.column, 0.0, domain.lower))
        .then_some((domain.column, 0.0, domain.lower))
    })
}
