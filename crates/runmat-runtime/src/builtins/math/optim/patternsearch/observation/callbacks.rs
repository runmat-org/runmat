//! Result roots outlive replacement-option admission. The original callback
//! group remains rooted even if an earlier callback returns new options.
use super::*;
use crate::builtins::math::optim::{common, patternsearch::evaluation};

fn callbacks(value: Option<&Value>) -> &[Value] {
    match value {
        None | Some(Value::Tensor(_)) => &[],
        Some(Value::Cell(value)) => value.data(),
        Some(value) => std::slice::from_ref(value),
    }
}
fn logical(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(value) if value.data.len() == 1 => Ok(value.data[0] != 0),
        _ => Err(invalid(format!("{label} must be a logical scalar"))),
    }
}
impl Observers {
    pub(super) async fn callbacks(
        &mut self,
        phase: Phase,
        metadata: Value,
        engine: &Engine,
        options: &Options,
        domain: &Domain,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Notification> {
        let mut notification = Notification {
            stop: false,
            replacement: None,
        };
        let flag = Value::from(phase.name());
        for callback in callbacks(options.callback(Field::OutputFcn)) {
            evaluator.cooperation.completed(0).await?;
            let current = notification.replacement.as_ref().unwrap_or(options);
            let returned = common::call_function_sequence_in_context(
                evaluator.cooperation.runtime(),
                callback,
                [metadata.clone(), current.value().clone(), flag.clone()],
                3,
            )
            .await?;
            let returned = evaluation::select(returned, 3, evaluator.cooperation.runtime())?;
            let mut values = {
                let _scope = evaluator
                    .cooperation
                    .runtime()
                    .map(crate::context::RuntimeContext::enter);
                returned.try_into_values().map_err(|mut error| {
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?
            };
            notification.stop |= logical(
                &values.next().expect("selected stop result"),
                "OutputFcn stop",
            )?;
            let replacement = values.next().expect("selected options result");
            let changed = logical(
                &values.next().expect("selected changed result"),
                "OutputFcn optchanged",
            )?;
            if changed && matches!(phase, Phase::Regular(OutputState::Init | OutputState::Iter)) {
                // The iterator retains the full returned root forest through
                // ALL provider awaits and successor publication in prepare.
                notification.replacement = Some(
                    Options::prepare(
                        Some(&replacement),
                        engine.point.x.len(),
                        domain.equalities,
                        domain.linear,
                        domain.nonlinear,
                        domain.policy,
                        evaluator.cooperation,
                    )
                    .await?,
                );
            }
            evaluator.cooperation.completed(0).await?;
        }
        if phase == Phase::Interrupt {
            return Ok(notification);
        }
        let current = notification.replacement.as_ref().unwrap_or(options);
        let run_plots = phase != Phase::Regular(OutputState::Iter)
            || engine
                .iterations
                .is_multiple_of(current.controls.plot_interval);
        if !run_plots {
            return Ok(notification);
        }
        let plot = current.callback(Field::PlotFcn);
        let count = output_callback::callback_count(plot, CallbackForm::HandlesOrNames)?;
        if count != self.plot_count {
            #[cfg(not(feature = "plot-core"))]
            if count > 0 {
                return Err(invalid("PlotFcn requires plot-core"));
            }
            #[cfg(feature = "plot-core")]
            {
                self.layout = if count > 0 {
                    Some(
                        crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                            count,
                            evaluator.cooperation.runtime(),
                        )?,
                    )
                } else {
                    None
                };
            }
            self.plot_count = count;
        }
        for (index, callback) in callbacks(plot).iter().enumerate() {
            evaluator.cooperation.completed(0).await?;
            self.select_plot(index, evaluator.cooperation.runtime())?;
            let returned = common::call_function_sequence_in_context(
                evaluator.cooperation.runtime(),
                callback,
                [metadata.clone(), flag.clone()],
                1,
            )
            .await?;
            let returned = evaluation::select(returned, 1, evaluator.cooperation.runtime())?;
            let returned = {
                let _scope = evaluator
                    .cooperation
                    .runtime()
                    .map(crate::context::RuntimeContext::enter);
                returned.try_into_single().map_err(|mut error| {
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?
            };
            notification.stop |= logical(returned.value(), "PlotFcn stop")?;
            evaluator.cooperation.completed(0).await?;
        }
        Ok(notification)
    }

    fn select_plot(
        &self,
        index: usize,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        #[cfg(feature = "plot-core")]
        {
            self.layout
                .as_ref()
                .ok_or_else(|| invalid("plot layout is unavailable"))?
                .select(index, runtime)
        }
        #[cfg(not(feature = "plot-core"))]
        {
            let _ = (index, runtime);
            Err(invalid("PlotFcn requires plot-core"))
        }
    }
}
