//! Flat phase driver: one observer generation per completed inner/major poll,
//! one canonical Done phase, and no callback replay for final publication.
use super::*;

pub(super) async fn drive(
    mut engine: iteration::Engine,
    mut options: Options,
    linear: &super::super::linear::Linear,
    domain: &observation::Domain,
    input: &Input<'_>,
    evaluator: &mut evaluation::Evaluator<'_>,
    deadline: &mut Deadline,
    observers: &mut observation::Observers,
) -> BuiltinResult<(iteration::Engine, Stop, Options)> {
    use common::output_callback::OutputState;
    use observation::Phase;
    enum Lifecycle {
        Running(Phase),
        Finishing(Stop),
    }
    let mut lifecycle = Lifecycle::Running(Phase::Regular(OutputState::Init));
    loop {
        let (phase, reason) = match lifecycle {
            Lifecycle::Running(phase) => (phase, None),
            Lifecycle::Finishing(reason) => (Phase::Regular(OutputState::Done), Some(reason)),
        };
        let notification = observers
            .notify(phase, &engine, reason, &options, domain, evaluator)
            .await?;
        // Done outputs have no following iteration; do not gather dormant updates.
        if let Some(reason) = reason {
            return Ok((engine, reason, options));
        }
        if let Some(replacement) = notification.replacement {
            options = replacement;
            evaluator.state.execution = evaluation::batch::Execution::select(
                &options.controls,
                evaluator.context.objective,
                evaluator.context.constraints,
                evaluator.cooperation.runtime(),
            )?;
            evaluator.state.limit = options.controls.evaluations;
            match (
                options.controls.search == Some(super::super::search::Search::Rbf),
                evaluator.state.history.is_some(),
            ) {
                (true, false) => {
                    evaluator.state.history = Some(super::super::search::rbf::History::new(
                        engine.point.x.len(),
                        evaluator.state.objective_calls,
                    )?)
                }
                (false, true) => evaluator.state.history = None,
                _ => {}
            }
            super::super::cache::Cache::configure(
                &mut evaluator.state.cache,
                options.controls.cache,
                evaluator.cooperation,
            )
            .await?;
            engine.reconfigure(&options.controls, evaluator).await?;
            deadline.reconfigure(options.controls.maximum_time)?;
        }
        if notification.stop {
            lifecycle = Lifecycle::Finishing(Stop::Observer);
            continue;
        }
        match engine
            .advance(linear, &options, input, evaluator, deadline)
            .await?
        {
            iteration::Event::Inner => lifecycle = Lifecycle::Running(Phase::Interrupt),
            iteration::Event::Iteration => {
                if matches!(
                    engine.method,
                    iteration::Method::Multipliers | iteration::Method::Penalty
                ) {
                    if let Some(cache) = &mut evaluator.state.cache {
                        cache.clear();
                    }
                }
                lifecycle = Lifecycle::Running(Phase::Regular(OutputState::Iter));
            }
            iteration::Event::Stopped(reason) => lifecycle = Lifecycle::Finishing(reason),
        }
    }
}
