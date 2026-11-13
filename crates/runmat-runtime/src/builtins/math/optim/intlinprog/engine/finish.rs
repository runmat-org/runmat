//! Same existing integer-tree finish owner, with original source validation retained.
use super::*;
pub(super) async fn finish(
    tree: &Tree,
    best: Option<Incumbent>,
    terminal: Terminal,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let lower = if matches!(terminal, Terminal::Unbounded) {
        None
    } else if tree.open.is_empty() && best.is_some() {
        best.as_ref().map(|v| v.objective.clone())
    } else {
        tree.lower_bound(cooperation).await?.map(|v| {
            let value = v.value();
            // A feasible incumbent is itself a remaining feasible alternative
            // when every unprocessed node bound is larger than its objective.
            best.as_ref()
                .map_or_else(|| value.clone(), |b| value.min(&b.objective).clone())
        })
    };
    Ok(Outcome {
        incumbent: best,
        lower,
        terminal,
    })
}
