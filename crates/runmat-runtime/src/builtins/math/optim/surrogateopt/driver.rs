//! Observer/display driver over the original numerical continuation. No user
//! callback is invoked by a modeled probe or replayed to prepare publication.
use super::{
    checkpoint::Session,
    controller::{Controller, Event, Stop},
    invalid,
    notice::Notice,
    observation::Objective,
    observers::Observers,
    publication,
};
use crate::call::descriptor::RepeatableCallable;
use crate::{
    builtins::math::optim::common::{output_callback::OutputState, ScalarCooperation},
    BuiltinResult,
};
use runmat_builtins::OptimDisplay;
mod mode;
pub(super) fn run<'a, 'source: 'a>(
    controller: &'a mut Controller<'source>,
    observers: &'a mut Observers,
    display: OptimDisplay,
    c: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Stop>> + 'a + use<'a, 'source> {
    selected_drive(controller, observers, display, mode::Ordinary, c)
}
pub(super) fn run_checkpoint<'a, 'source: 'a>(
    controller: &'a mut Controller<'source>,
    observers: &'a mut Observers,
    display: OptimDisplay,
    session: &'a mut Session,
    objective: &'a RepeatableCallable,
    resume: bool,
    c: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Stop>> + 'a + use<'a, 'source> {
    selected_drive(
        controller,
        observers,
        display,
        mode::Checkpoint {
            session,
            objective,
            resume,
        },
        c,
    )
}

fn selected_drive<'a, 'source: 'a, M: mode::Mode + 'a>(
    controller: &'a mut Controller<'source>,
    observers: &'a mut Observers,
    display: OptimDisplay,
    mode: M,
    c: &'a mut ScalarCooperation,
) -> crate::execution::RuntimeContinuationFuture<'a, Stop> {
    let runtime = c.runtime().cloned();
    // One numerical controller per solve/checkpoint invocation. Its large
    // construction frame retires before polling callbacks; trials and probes
    // do not allocate another controller or acquire another runtime authority.
    crate::execution::rooted_continuation::admit_optional_hostcall_with_contextual_error(
        runtime.as_ref(),
        || drive(controller, observers, display, mode, c),
    )
    .unwrap_or_else(crate::execution::RuntimeContinuationFuture::rejected)
}

async fn drive<M: mode::Mode>(
    controller: &mut Controller<'_>,
    observers: &mut Observers,
    display: OptimDisplay,
    mut mode: M,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Stop> {
    if !runmat_builtins::OptimSolver::Surrogateopt.accepts_display(display) {
        return Err(invalid("unsupported Display mode"));
    }
    let mut observer_stopped = observers.notify(OutputState::Init, controller, c).await?;
    if observer_stopped {
        controller.request_stop();
    }
    controller.acknowledge_notice();
    mode.resume(controller, observers, &mut observer_stopped, c)
        .await?;
    let mut header = false;
    loop {
        match controller.advance(c).await? {
            Event::Iteration => {
                if display == OptimDisplay::Iter {
                    report(controller, None, &mut header, c)?;
                }
                let observed = observers
                    .notify_notice(OutputState::Iter, Notice::live(controller)?, c)
                    .await?;
                mode.record(observed.frame)?;
                if observed.stop {
                    observer_stopped = true;
                    controller.request_stop();
                }
                controller.acknowledge_notice();
                mode.iteration(
                    controller,
                    observers,
                    display,
                    &mut header,
                    &mut observer_stopped,
                    c,
                )
                .await?;
            }
            Event::Done(stop) => {
                observers.notify(OutputState::Done, controller, c).await?;
                controller.acknowledge_notice();
                mode.done(controller, observers, c).await?;
                if matches!(display, OptimDisplay::Iter | OptimDisplay::Final) {
                    report(controller, Some(stop), &mut header, c)?;
                }
                return Ok(stop);
            }
        }
    }
}
fn report(
    controller: &Controller<'_>,
    stop: Option<Stop>,
    header: &mut bool,
    c: &ScalarCooperation,
) -> BuiltinResult<()> {
    let history = controller.campaign.history();
    let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
    let line = |text: String| {
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, text)
    };
    if !*header {
        if let Some(schema) = history.schema() {
            line(format!(
                "surrogateopt: {} variables, objective={}, {} nonlinear inequalities",
                history.variables(),
                if schema.objective == super::observation::ObjectiveKind::Present {
                    "scalar"
                } else {
                    "none"
                },
                schema.inequalities
            ))?;
            line(
                "Function evaluations  Best objective  Maximum infeasibility  Surrogate resets"
                    .to_owned(),
            )?;
            *header = true;
        }
    }
    let (objective, infeasibility) = controller
        .campaign
        .tracker()
        .best()
        .and_then(|i| history.trial(i).observation.measurement())
        .map_or_else(
            || ("-".to_owned(), "-".to_owned()),
            |value| {
                if value.nonlinear.violated == 0 {
                    (
                        match value.objective {
                            Objective::Absent => "-".to_owned(),
                            Objective::Measured(v) => v.to_string(),
                        },
                        "-".to_owned(),
                    )
                } else {
                    (
                        "-".to_owned(),
                        value
                            .nonlinear
                            .maximum
                            .map_or_else(|| "-".to_owned(), |v| v.to_string()),
                    )
                }
            },
        );
    line(format!(
        "{}  {}  {}  {}",
        controller.counts().1,
        objective,
        infeasibility,
        controller.campaign.phase().0
    ))?;
    if let Some(stop) = stop {
        line(publication::message(stop, publication::flag(controller, stop)).to_owned())?;
    }
    Ok(())
}
