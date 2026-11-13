//! Uncapped original-row blocking for a domain-checked recession proposal.
use super::{rows, Outcome, Rows, Target};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;

pub(super) struct State<'a> {
    pub x: &'a mut Vec<f64>,
    pub active: &'a mut Vec<bool>,
    pub stationary: &'a mut usize,
}
pub(super) async fn advance<T: Target>(
    objective: &mut T,
    model: Rows<'_>,
    state: State<'_>,
    direction: Vec<f64>,
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Outcome>> {
    let State {
        x,
        active,
        stationary,
    } = state;
    if direction.len() != x.len()
        || direction.iter().any(|value| !value.is_finite())
        || !rows::feasible(model, x, tolerance, owner, c).await?
    {
        return Ok(Some(Outcome::Numerical {
            point: std::mem::take(x),
        }));
    }
    let Some(activity) = rows::product(model.a, x, owner, c).await? else {
        return Ok(Some(Outcome::Numerical {
            point: std::mem::take(x),
        }));
    };
    let Some(delta) = rows::product(model.a, &direction, owner, c).await? else {
        return Ok(Some(Outcome::Numerical {
            point: std::mem::take(x),
        }));
    };
    let mut block = None;
    let mut alpha = f64::INFINITY;
    for row in model.equalities..model.b.len() {
        if !active[row] && delta[row] > 0. {
            let ratio = ((model.b[row] - activity[row]) / delta[row]).max(0.);
            // A genuine blocker whose distance overflows is not evidence that
            // no blocker exists. Preserve numerical uncertainty instead.
            if !ratio.is_finite() {
                return Ok(Some(Outcome::Numerical {
                    point: std::mem::take(x),
                }));
            }
            if ratio < alpha {
                alpha = ratio;
                block = Some(row);
            }
        }
        c.completed(1).await?;
    }
    let Some(block) = block else {
        return Ok(Some(Outcome::Unbounded {
            point: std::mem::take(x),
            direction,
        }));
    };
    let mut finite = true;
    for (&value, &step) in x.iter().zip(&direction) {
        finite &= (value + alpha * step).is_finite();
        c.completed(1).await?;
    }
    if !finite {
        return Ok(Some(Outcome::Numerical {
            point: std::mem::take(x),
        }));
    }
    let mut moved = false;
    for (value, &step) in x.iter_mut().zip(&direction) {
        let next = *value + alpha * step;
        moved |= next != *value;
        *value = next;
        c.completed(1).await?;
    }
    *stationary = if moved {
        0
    } else {
        (*stationary).saturating_add(1)
    };
    if *stationary > model.b.len() || !rows::feasible(model, x, tolerance, owner, c).await? {
        return Ok(Some(Outcome::Numerical {
            point: std::mem::take(x),
        }));
    }
    active[block] = true;
    if let Some(reason) = objective.advanced(x, c).await? {
        objective.retain_active(std::mem::take(active));
        return Ok(Some(Outcome::Stopped {
            point: std::mem::take(x),
            reason,
        }));
    }
    Ok(None)
}
