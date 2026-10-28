//! One shared Nelder-Mead controller; the original oracle owns all effects.
use super::geometry::*;
use super::*;
mod initial;
mod publication;

struct Invocation<'a, O, E> {
    policy: Policy,
    oracle: &'a mut O,
    options: &'a Controls,
    events: &'a mut E,
}

struct Initialized {
    n: usize,
    count: usize,
    vertices: Vec<Point>,
    reason: Option<Reason>,
}

struct Completed {
    vertices: Vec<Point>,
    order: Vec<usize>,
    iterations: usize,
    procedure: Procedure,
    reason: Reason,
}

pub(in crate::builtins::math::optim) fn minimize<'a, O: Oracle + 'a, E: Events + 'a>(
    policy: Policy,
    initial: Initial,
    oracle: &'a mut O,
    options: &'a Controls,
    events: &'a mut E,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    let invocation = Invocation {
        policy,
        oracle,
        options,
        events,
    };
    crate::execution::after_result(
        initial::prepare(invocation, initial),
        |(state, invocation)| {
            crate::execution::after_result(iterate(state, invocation), publication::finish)
        },
    )
}

async fn iterate<O: Oracle, E: Events>(
    state: Initialized,
    invocation: Invocation<'_, O, E>,
) -> BuiltinResult<(Completed, Invocation<'_, O, E>)> {
    let Initialized {
        n,
        count,
        mut vertices,
        mut reason,
    } = state;
    let Invocation {
        policy,
        oracle,
        options,
        events,
    } = invocation;
    let mut iterations = 0;
    let mut procedure = Procedure::Initial;
    if reason.is_none() {
        for coordinate in 0..n {
            if oracle.remaining() == 0 {
                reason = Some(Reason::Evaluations);
                break;
            }
            let mut x = copy(policy, &vertices[0].x, oracle.cooperation()).await?;
            x[coordinate] = if x[coordinate] == 0.0 {
                0.00025
            } else {
                x[coordinate] * 1.05
            };
            if !x[coordinate].is_finite() {
                return Err(policy.invalid("initial simplex coordinate is not finite"));
            }
            match evaluate(x, oracle).await? {
                Ok(point) => vertices.push(point),
                Err(stop) => {
                    reason = Some(reason_for(stop));
                    break;
                }
            }
        }
    }
    let mut order = policy.buffer(count)?;
    order.extend(0..vertices.len());
    sort(policy, &mut order, &vertices, oracle.cooperation()).await?;
    if reason.is_none() {
        iterations = 1;
        procedure = Procedure::Simplex;
        reason = notify(
            events,
            &vertices[order[0]],
            iterations,
            oracle.calls(),
            procedure,
            oracle.cooperation(),
        )
        .await?;
    }
    let mut centroid = policy.buffer(if reason.is_none() { n } else { 0 })?;
    if reason.is_none() {
        centroid.resize(n, 0.0);
    }
    while reason.is_none() {
        oracle.cooperation().completed(0).await?;
        if converged(&vertices, &order, options, oracle.cooperation()).await? {
            reason = Some(Reason::Converged);
            break;
        }
        if iterations >= options.max_iter {
            reason = Some(Reason::Iterations);
            break;
        }
        if oracle.remaining() == 0 {
            reason = Some(Reason::Evaluations);
            break;
        }
        center(
            policy,
            &vertices,
            &order,
            &mut centroid,
            oracle.cooperation(),
        )
        .await?;
        let worst = order[n];
        let reflected = affine(
            policy,
            &centroid,
            &vertices[worst].x,
            -1.0,
            oracle.cooperation(),
        )
        .await?;
        let reflected = match evaluate(reflected, oracle).await? {
            Ok(point) => point,
            Err(stop) => {
                reason = Some(reason_for(stop));
                break;
            }
        };
        let mut completed = true;
        if less(reflected.f, vertices[order[0]].f) {
            let expanded = if oracle.remaining() == 0 {
                Err(OracleStop::Evaluations)
            } else {
                let expanded = affine(
                    policy,
                    &centroid,
                    &vertices[worst].x,
                    -2.0,
                    oracle.cooperation(),
                )
                .await?;
                evaluate(expanded, oracle).await?
            };
            match expanded {
                Ok(expanded) if less(expanded.f, reflected.f) => {
                    vertices[worst] = expanded;
                    procedure = Procedure::Expand;
                }
                Ok(_) => {
                    vertices[worst] = reflected;
                    procedure = Procedure::Reflect;
                }
                Err(stop) => {
                    vertices[worst] = reflected;
                    reason = Some(reason_for(stop));
                    completed = false;
                }
            }
        } else if less(reflected.f, vertices[order[n - 1]].f) {
            vertices[worst] = reflected;
            procedure = Procedure::Reflect;
        } else {
            let outside = less(reflected.f, vertices[worst].f);
            let contraction = if oracle.remaining() == 0 {
                Err(OracleStop::Evaluations)
            } else {
                let contracted = affine(
                    policy,
                    &centroid,
                    if outside {
                        &reflected.x
                    } else {
                        &vertices[worst].x
                    },
                    0.5,
                    oracle.cooperation(),
                )
                .await?;
                evaluate(contracted, oracle).await?
            };
            match contraction {
                Ok(contracted)
                    if if outside {
                        !contracted.f.is_nan() && contracted.f <= reflected.f
                    } else {
                        less(contracted.f, vertices[worst].f)
                    } =>
                {
                    vertices[worst] = contracted;
                    procedure = if outside {
                        Procedure::ContractOutside
                    } else {
                        Procedure::ContractInside
                    };
                }
                Err(stop) => {
                    reason = Some(reason_for(stop));
                    completed = false;
                }
                Ok(_) => {
                    // Shrink the original simplex, not the rejected reflection.
                    // Partial budget stops keep only actually evaluated points.
                    for &slot in order.iter().skip(1) {
                        if oracle.remaining() == 0 {
                            reason = Some(Reason::Evaluations);
                            completed = false;
                            break;
                        }
                        let x = affine(
                            policy,
                            &vertices[order[0]].x,
                            &vertices[slot].x,
                            0.5,
                            oracle.cooperation(),
                        )
                        .await?;
                        match evaluate(x, oracle).await? {
                            Ok(point) => vertices[slot] = point,
                            Err(stop) => {
                                reason = Some(reason_for(stop));
                                completed = false;
                                break;
                            }
                        }
                    }
                    procedure = Procedure::Shrink;
                }
            }
        }
        sort(policy, &mut order, &vertices, oracle.cooperation()).await?;
        if completed {
            iterations = iterations
                .checked_add(1)
                .ok_or_else(|| policy.invalid("iteration counter overflow"))?;
            reason = notify(
                events,
                &vertices[order[0]],
                iterations,
                oracle.calls(),
                procedure,
                oracle.cooperation(),
            )
            .await?;
        }
    }
    let reason = reason.expect("terminal reason");
    Ok((
        Completed {
            vertices,
            order,
            iterations,
            procedure,
            reason,
        },
        Invocation {
            policy,
            oracle,
            options,
            events,
        },
    ))
}

fn reason_for(stop: OracleStop) -> Reason {
    match stop {
        OracleStop::Evaluations => Reason::Evaluations,
        OracleStop::Time => Reason::Time,
    }
}
async fn evaluate(
    x: Vec<f64>,
    oracle: &mut impl Oracle,
) -> BuiltinResult<Result<Point, OracleStop>> {
    Ok(match oracle.measure(&x).await? {
        Measurement::Defined(f) => Ok(Point { x, f }),
        Measurement::Undefined => Ok(Point { x, f: f64::NAN }),
        Measurement::Stopped(stop) => Err(stop),
    })
}
async fn notify(
    events: &mut impl Events,
    point: &Point,
    iteration: usize,
    calls: usize,
    procedure: Procedure,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Reason>> {
    Ok(events
        .notify(
            OutputState::Iter,
            point,
            iteration,
            calls,
            procedure,
            cooperation,
        )
        .await?
        .then_some(Reason::Observer))
}
