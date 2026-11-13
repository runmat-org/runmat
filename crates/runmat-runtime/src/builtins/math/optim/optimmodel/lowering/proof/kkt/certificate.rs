//! Exact certificate preparation retires before stationarity, gap and complementarity.
use super::*;

pub(super) struct Certificate {
    data: phases::Data,
    gradient: Vec<Rational>,
    x_norm: Rational,
    objective_scale: Rational,
    differential_cost: Option<Rational>,
    evaluated: Option<exact::Evaluated>,
}

pub(super) async fn prepare<'a, 'source>(
    authority: &'a mut Authority<'source>,
    data: phases::Data,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<(
    Certificate,
    &'a mut Authority<'source>,
    &'a mut ScalarCooperation,
)> {
    let phases::Data {
        candidate,
        weights,
        weighted,
    } = data;
    let policy = authority.policy;
    if authority.objective_scale.is_none() {
        if let Some(objective) = authority.program.objective.squares() {
            authority.objective_scale = Some(
                objective
                    .scale(
                        authority.program,
                        authority
                            .origin
                            .as_ref()
                            .expect("original affine intercepts"),
                        policy,
                        cooperation,
                    )
                    .await?,
            );
        }
    }
    let data_scale = authority
        .program
        .objective
        .quadratic()
        .map(|evidence| &evidence.scale)
        .or(authority.objective_scale.as_ref());
    let mut x_norm = Rational::zero();
    if data_scale.is_some() {
        for value in &candidate {
            let magnitude = value.abs();
            if magnitude
                .compare(&x_norm, policy, cooperation)
                .await?
                .is_gt()
            {
                x_norm = magnitude;
            }
            cooperation.completed(1).await?;
        }
    }
    let origin = authority
        .origin
        .as_ref()
        .expect("original affine intercepts");
    let evaluated = if authority.program.objective.squares().is_some() {
        Some(
            exact::evaluate(
                authority.program,
                exact::Coordinates::Point(&candidate),
                exact::Retain::Roots,
                policy,
                cooperation,
            )
            .await?,
        )
    } else {
        None
    };
    let mut differential_cost = None;
    let mut seeds = Vec::new();
    if let Some(objective) = authority.program.objective.squares() {
        let (weights, cost) = objective
            .seeds(
                evaluated.as_ref().expect("original residual point"),
                origin,
                authority.program,
                policy,
                cooperation,
            )
            .await?;
        seeds = weights;
        differential_cost = Some(cost);
    } else if let Some(objective) = authority
        .formulation
        .objective
        .as_ref()
        .filter(|_| authority.program.objective.quadratic().is_none())
    {
        seeds.try_reserve(1).map_err(|e| policy.reserve(e))?;
        seeds.push(exact::Seed {
            root: objective.root,
            row: 0,
            value: rational(
                if objective.sense == Sense::Minimize {
                    1.
                } else {
                    -1.
                },
                policy,
            )?,
        });
    }
    let objective_gradient = if let Some(evidence) = authority.program.objective.quadratic() {
        let sense = authority
            .formulation
            .objective
            .as_ref()
            .map_or(Sense::Minimize, |objective| objective.sense);
        let gradient = evidence
            .gradient(
                authority.program,
                exact::Coordinates::Point(&candidate),
                sense,
                policy,
                cooperation,
            )
            .await?;
        differential_cost =
            Some(directional_cost(&gradient, &candidate, policy, cooperation).await?);
        gradient
    } else {
        exact::accumulate(authority.program, origin, seeds, policy, cooperation).await?
    };
    let objective_scale = if let Some(scale) = data_scale {
        scale
            .magnitude(&candidate, &x_norm, policy, cooperation)
            .await?
    } else {
        let mut objective_scale = Rational::zero();
        for value in &objective_gradient {
            if value
                .abs()
                .compare(&objective_scale, policy, cooperation)
                .await?
                .is_gt()
            {
                objective_scale = value.abs();
            }
            cooperation.completed(1).await?;
        }
        if objective_scale.is_zero() {
            objective_scale = rational(1., policy)?;
        }
        objective_scale
    };

    let certificate = Certificate {
        data: phases::Data {
            candidate,
            weights,
            weighted,
        },
        gradient: objective_gradient,
        x_norm,
        objective_scale,
        differential_cost,
        evaluated,
    };
    Ok((certificate, authority, cooperation))
}

pub(super) async fn check(
    authority: &Authority<'_>,
    certificate: Certificate,
    roles: &[RowRole],
    tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let Certificate {
        data:
            phases::Data {
                candidate,
                weights,
                weighted,
            },
        gradient: objective_gradient,
        x_norm,
        objective_scale,
        differential_cost,
        mut evaluated,
    } = certificate;
    let policy = authority.policy;
    let origin = authority
        .origin
        .as_ref()
        .expect("original affine intercepts");
    let data_scale = authority
        .program
        .objective
        .quadratic()
        .map(|evidence| &evidence.scale)
        .or(authority.objective_scale.as_ref());
    let tolerance = rational(tolerance, policy)?;
    for (column, (objective, row)) in objective_gradient
        .into_iter()
        .zip(&weighted.gradient)
        .enumerate()
    {
        let column_scale = if let Some(scale) = data_scale {
            let value = scale.at(column, &x_norm, policy, cooperation).await?;
            if value.is_zero() {
                Rational::from_usize(1)
            } else {
                value
            }
        } else {
            objective_scale.clone()
        };
        let magnitude = column_scale
            .add(&objective.abs(), policy, cooperation)
            .await?
            .add(&row.abs(), policy, cooperation)
            .await?;
        if objective
            .add(row, policy, cooperation)
            .await?
            .abs()
            .compare(
                &tolerance.mul(&magnitude, policy, cooperation).await?,
                policy,
                cooperation,
            )
            .await?
            .is_gt()
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    let evaluated = match evaluated.take() {
        Some(value) => value,
        None => {
            exact::evaluate(
                authority.program,
                exact::Coordinates::Point(&candidate),
                exact::Retain::Roots,
                policy,
                cooperation,
            )
            .await?
        }
    };
    let mut cost = differential_cost.unwrap_or_else(Rational::zero);
    if authority.program.objective.squares().is_none()
        && authority.program.objective.quadratic().is_none()
    {
        if let Some(objective) = &authority.formulation.objective {
            let Some(value) =
                difference(&evaluated, origin, objective.root, 0, policy, cooperation).await?
            else {
                return Ok(false);
            };
            cost = if objective.sense == Sense::Minimize {
                value
            } else {
                -value
            };
        }
    }
    let scale = objective_scale
        .add(&cost.abs(), policy, cooperation)
        .await?
        .add(&weighted.intercept.abs(), policy, cooperation)
        .await?;
    let threshold = tolerance.mul(&scale, policy, cooperation).await?;
    if cost
        .sub(&weighted.intercept, policy, cooperation)
        .await?
        .abs()
        .compare(&threshold, policy, cooperation)
        .await?
        .is_gt()
    {
        return Ok(false);
    }
    for (role, weight) in roles.iter().zip(&weights) {
        if matches!(role, RowRole::Equality(_)) {
            continue;
        }
        if weight.is_negative()
            && weight
                .abs()
                .compare(
                    &tolerance
                        .mul(
                            &objective_scale
                                .add(&weight.abs(), policy, cooperation)
                                .await?,
                            policy,
                            cooperation,
                        )
                        .await?,
                    policy,
                    cooperation,
                )
                .await?
                .is_gt()
        {
            return Ok(false);
        }
        let Some(residual) = authority
            .residual(role, &candidate, &evaluated, cooperation)
            .await?
        else {
            return Ok(false);
        };
        if weight
            .mul(&residual, policy, cooperation)
            .await?
            .abs()
            .compare(&threshold, policy, cooperation)
            .await?
            .is_gt()
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
