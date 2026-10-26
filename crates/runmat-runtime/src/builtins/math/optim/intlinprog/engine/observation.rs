//! Optional MILP observations borrow proofs and never authorize numerical work.
use super::*;

pub(super) struct Event<'a> {
    pub(super) state: common::output_callback::OutputState,
    pub(super) new_point: bool,
    pub(super) best: Option<&'a Incumbent>,
    pub(super) lower: Option<&'a BigRational>,
}
pub(super) async fn observe<V: compiled::Validation>(
    event: Event<'_>,
    workspace: &Workspace,
    controls: &mut controls::Controls,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Option<Stop>> {
    let Event {
        state,
        new_point,
        best,
        lower,
    } = event;
    if !controls.observers.enabled() {
        return Ok(None);
    }
    let runtime = cooperation.runtime().cloned();
    let x = if new_point {
        let witness = best.ok_or_else(|| numerical("new incumbent event lacks its witness"))?;
        let Some(binary) = publication::BinaryPoint::prepare(
            &witness.point,
            &workspace.integers,
            &workspace.semis,
            cooperation,
        )
        .await?
        else {
            return Ok(Some(Stop::Numerical));
        };
        // Coordinate scratch is allocated only for an enabled observation.
        // It is the actual public binary point, not a copied exact witness or model.
        let mut scratch =
            runmat_value::exact_capacity_vec(binary.values.len()).map_err(allocation)?;
        scratch.resize_with(binary.values.len(), <BigRational as num_traits::Zero>::zero);
        let evaluated = binary
            .evaluate(
                &mut scratch,
                &workspace.node(),
                controls,
                validation,
                cooperation,
            )
            .await?;
        drop(scratch);
        if !evaluated.admitted {
            return Ok(Some(Stop::Numerical));
        }
        controls.observers.publish_fval(evaluated.objective);
        let n = binary.values.len();
        Value::Tensor(Tensor::new(binary.values, vec![n, 1]).map_err(numerical)?)
    } else {
        common::output_callback::empty(runtime.as_ref())?
    };
    let lower = lower.map(|lower| best.map_or(lower, |best| lower.min(&best.objective)));
    let relativegap = if !workspace.is_mixed() {
        None
    } else {
        best.map(|best| {
            lower
                .and_then(|lower| gap::relative(&best.objective, lower))
                .as_ref()
                .map(publication::nonnegative)
                .unwrap_or(f64::INFINITY)
        })
    };
    let snapshot = observers::Snapshot {
        fval: controls.observers.published_fval(),
        dualbound: lower.map(publication::number).unwrap_or(f64::NEG_INFINITY),
        relativegap,
        numnodes: workspace.is_mixed().then_some(controls.nodes),
        numfeaspoints: controls.improving_points,
        elapsed: controls.elapsed(),
    };
    Ok(controls
        .observers
        .notify(state, x, snapshot, cooperation)
        .await?
        .then_some(Stop::Observer))
}

pub(super) struct Frontier<'a> {
    pub tree: &'a Tree,
    pub current: Option<usize>,
}
pub(super) async fn accept_observed<V: compiled::Validation>(
    mut candidate: Incumbent,
    best: &mut Option<Incumbent>,
    frontier: Frontier<'_>,
    workspace: &Workspace,
    controls: &mut controls::Controls,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Option<Stop>> {
    if !validation
        .incumbent(
            &workspace.node(),
            &candidate.point,
            &mut candidate.objective,
            cooperation,
        )
        .await?
    {
        return Ok(None);
    }
    if !accept(candidate, best, controls)? || !controls.observers.enabled() {
        return Ok(None);
    }
    let lower = frontier
        .tree
        .lower_bound_including(frontier.current, cooperation)
        .await?;
    observe(
        Event {
            state: common::output_callback::OutputState::Iter,
            new_point: true,
            best: best.as_ref(),
            lower: lower.as_ref().map(|v| v.value()),
        },
        workspace,
        controls,
        cooperation,
        validation,
    )
    .await
}
