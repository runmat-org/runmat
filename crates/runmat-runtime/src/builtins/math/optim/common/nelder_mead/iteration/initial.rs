//! Admit and observe the initial point before simplex iteration exists.
use super::*;

pub(super) async fn prepare<O: Oracle, E: Events>(
    invocation: Invocation<'_, O, E>,
    initial: Initial,
) -> BuiltinResult<(Initialized, Invocation<'_, O, E>)> {
    let Invocation {
        policy,
        oracle,
        options,
        events,
    } = invocation;
    let n = match &initial {
        Initial::Evaluate(x) => x.len(),
        Initial::Known(point) => point.x.len(),
    };
    let count = if options.max_iter == 0 {
        1
    } else {
        geometry(policy, n)?
    };
    let mut vertices = policy.buffer(count)?;
    let initial = match initial {
        Initial::Known(point) => point,
        Initial::Evaluate(x) => evaluate(x, oracle)
            .await?
            .map_err(|_| policy.invalid("initial evaluation budget is empty"))?,
    };
    vertices.push(initial);
    let mut reason = if events
        .notify(
            OutputState::Init,
            &vertices[0],
            0,
            oracle.calls(),
            Procedure::Initial,
            oracle.cooperation(),
        )
        .await?
    {
        Some(Reason::Observer)
    } else {
        None
    };
    if reason.is_none() && options.max_iter == 0 {
        reason = Some(Reason::Iterations);
    }
    Ok((
        Initialized {
            n,
            count,
            vertices,
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
