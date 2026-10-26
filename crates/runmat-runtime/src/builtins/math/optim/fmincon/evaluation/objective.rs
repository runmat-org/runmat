//! One original rooted objective tuple decoder for scalar and vector models.
//! Public Jacobian columns retain objective identities before epigraph lifting.
use super::super::numeric::{self, Geometry, Numeric, Policy};
use super::{jacobian, selected_values};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;

#[derive(Clone, Copy)]
pub(super) enum Shape {
    Scalar,
    ScalarWithModel,
    Vector,
}
pub(super) enum Decoded {
    Scalar {
        value: f64,
        gradient: Option<Vec<f64>>,
        model: Option<crate::execution::RootedValue>,
    },
    Vector {
        values: Vec<f64>,
        shape: Vec<usize>,
        jacobian: Option<CscMatrix<f64>>,
    },
}

pub(super) async fn read(
    returned: crate::execution::RootedValueSequence,
    variables: usize,
    shape: Shape,
    analytic: bool,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Decoded>> {
    let geometry = match shape {
        Shape::Scalar | Shape::ScalarWithModel => Geometry::Scalar,
        Shape::Vector => Geometry::Array,
    };
    let first = &returned.as_slice()[0];
    if !numeric::preflight(
        first,
        "objective value",
        geometry,
        policy,
        cooperation.runtime(),
    )? {
        return Ok(None);
    }
    let objectives = numeric::shape(first)
        .expect("preflight numeric carrier")
        .iter()
        .try_fold(1usize, |n, &axis| n.checked_mul(axis))
        .ok_or_else(|| numeric::invalid(policy, "objective value", "extent overflow"))?;
    if objectives == 0 {
        return Err(numeric::invalid(
            policy,
            "objective value",
            "must be nonempty",
        ));
    }
    if analytic {
        let geometry = match shape {
            Shape::Scalar | Shape::ScalarWithModel => Geometry::Elements(variables),
            Shape::Vector => Geometry::Matrix(variables, objectives),
        };
        if !numeric::preflight(
            &returned.as_slice()[1],
            "objective Jacobian",
            geometry,
            policy,
            cooperation.runtime(),
        )? {
            return Ok(None);
        }
    }
    // The iterator owns the original roots across all asynchronous decodes.
    let mut selected = selected_values(returned, cooperation.runtime())?;
    let Numeric::Real {
        values,
        shape: value_shape,
    } = numeric::read(
        selected.next().expect("selected objective tuple"),
        "objective value",
        geometry,
        policy,
        cooperation,
    )
    .await?
    else {
        return Ok(None);
    };
    let decoded = match shape {
        Shape::Scalar | Shape::ScalarWithModel => {
            let gradient = if analytic {
                let Numeric::Real { values, .. } = numeric::read(
                    selected.next().expect("selected scalar gradient"),
                    "objective gradient",
                    Geometry::Elements(variables),
                    policy,
                    cooperation,
                )
                .await?
                else {
                    return Ok(None);
                };
                Some(values)
            } else {
                None
            };
            Decoded::Scalar {
                value: values[0],
                gradient,
                model: if matches!(shape, Shape::ScalarWithModel) {
                    Some(
                        crate::execution::RootedValue::in_optional_context(
                            selected
                                .next()
                                .expect("selected original third objective result"),
                            cooperation.runtime(),
                        )
                        .map_err(|mut error| {
                            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                            error
                        })?,
                    )
                } else {
                    None
                },
            }
        }
        Shape::Vector => {
            let jacobian = if analytic {
                let jacobian::Jacobian::Real(value) = jacobian::read(
                    selected.next().expect("selected objective Jacobian"),
                    variables,
                    objectives,
                    "objective Jacobian",
                    policy,
                    cooperation,
                )
                .await?
                else {
                    return Ok(None);
                };
                Some(value)
            } else {
                None
            };
            Decoded::Vector {
                values,
                shape: value_shape,
                jacobian,
            }
        }
    };
    Ok(Some(decoded))
}
