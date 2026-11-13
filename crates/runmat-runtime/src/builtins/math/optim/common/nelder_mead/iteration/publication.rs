//! Publish one terminal event after the numerical iteration frame retires.
use super::*;

pub(super) async fn finish<O: Oracle, E: Events>(
    (completed, invocation): (Completed, Invocation<'_, O, E>),
) -> BuiltinResult<Outcome> {
    let Completed {
        mut vertices,
        order,
        iterations,
        procedure,
        reason,
    } = completed;
    let Invocation { oracle, events, .. } = invocation;
    let best = order[0];
    events
        .notify(
            OutputState::Done,
            &vertices[best],
            iterations,
            oracle.calls(),
            procedure,
            oracle.cooperation(),
        )
        .await?;
    let point = vertices.swap_remove(best);
    Ok(Outcome {
        point,
        reason,
        iterations,
        func_count: oracle.calls(),
    })
}
