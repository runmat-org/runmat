//! Actual two-argument swarm observers share current Tensor backing and the
//! original rooted metadata generation. Plot layout retains only UI receivers.
use super::{
    invalid,
    options::Options,
    state::{State, Termination},
    ScalarCooperation,
};
use crate::builtins::math::optim::common::output_callback::{
    self, CallbackForm, ObserverGroup, OutputState,
};
use crate::BuiltinResult;
use runmat_value::Value;
pub(super) struct Observers {
    output: Option<Value>,
    plot: Option<Value>,
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
}
impl Observers {
    pub fn admit(options: &Options) -> BuiltinResult<usize> {
        let count = output_callback::callback_count(
            options.plot.as_ref(),
            super::common::swarm_options::plot_form(),
        )?;
        #[cfg(not(feature = "plot-core"))]
        if count > 0 {
            return Err(invalid("PlotFcn requires the actual plot-core UI owner"));
        }
        Ok(count)
    }
    pub fn new(options: &Options, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        let count = Self::admit(options)?;
        #[cfg(not(feature = "plot-core"))]
        let _ = (cooperation, count);
        #[cfg(feature = "plot-core")]
        let layout = if count > 0 {
            Some(
                crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                    count,
                    cooperation.runtime(),
                )?,
            )
        } else {
            None
        };
        Ok(Self {
            output: options.output.clone(),
            plot: options.plot.clone(),
            #[cfg(feature = "plot-core")]
            layout,
        })
    }
    pub async fn notify(
        &self,
        phase: OutputState,
        state: &State,
        counts: super::evaluation::Counts,
        options: &Options,
        reason: Option<Termination>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let show = match options.display {
            runmat_builtins::OptimDisplay::Off | runmat_builtins::OptimDisplay::None => false,
            runmat_builtins::OptimDisplay::Final => phase == OutputState::Done,
            runmat_builtins::OptimDisplay::Iter => {
                phase == OutputState::Done
                    || state.iteration.is_multiple_of(options.display_interval)
            }
            runmat_builtins::OptimDisplay::Notify => false,
            runmat_builtins::OptimDisplay::Diagnose => {
                let mut error = invalid("diagnostic display is not supported by particleswarm");
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(error);
            }
        };
        if show {
            let mean = state.fitness.mean(cooperation).await?;
            let _scope = cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                format!(
                    "particleswarm {}: iteration={} funccount={} bestfval={:?} meanfval={:?} stalliterations={}",
                    phase.name(),
                    state.iteration,
                    counts.points,
                    state.best_fitness,
                    mean,
                    state.stall_iterations
                ),
            )?;
            if let Some(reason) = reason {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    reason.message(),
                )?;
            }
        }
        if output_callback::callback_count(self.output.as_ref(), CallbackForm::Handles)? == 0
            && output_callback::callback_count(
                self.plot.as_ref(),
                super::common::swarm_options::plot_form(),
            )? == 0
        {
            return Ok(false);
        }
        let runtime = cooperation.runtime().cloned();
        let best = state.best_point(cooperation).await?;
        let mean = state.fitness.mean(cooperation).await?;
        let fields = [
            ("funccount", Value::Num(counts.points as f64)),
            ("bestx", best),
            ("bestfval", Value::Num(state.best_fitness)),
            ("iteration", Value::Num(state.iteration as f64)),
            ("meanfval", mean),
            ("stalliterations", Value::Num(state.stall_iterations as f64)),
            ("swarm", state.population.value()),
            ("swarmfvals", state.fitness.value()),
        ];
        let metadata = crate::execution::construct_struct_from_fields_in_optional_context(
            runtime.as_ref(),
            fields.iter().map(|(name, value)| (*name, value)),
        )?;
        let phase = crate::call::argument_text::copy_str(phase.name(), runtime.as_ref())
            .map(|text| Value::String(text.into()))?;
        let select = |index| -> BuiltinResult<()> {
            #[cfg(feature = "plot-core")]
            {
                self.layout
                    .as_ref()
                    .ok_or_else(|| invalid("plot layout is unavailable"))?
                    .select(index, runtime.as_ref())
            }
            #[cfg(not(feature = "plot-core"))]
            {
                let _ = index;
                Err(invalid("plot-core unavailable"))
            }
        };
        let groups = [
            ObserverGroup {
                value: self.output.as_ref(),
                form: CallbackForm::Handles,
                select: None,
            },
            ObserverGroup {
                value: self.plot.as_ref(),
                form: super::common::swarm_options::plot_form(),
                select: Some(&select),
            },
        ];
        output_callback::invoke_population_groups(
            "particleswarm",
            &groups,
            metadata,
            phase,
            cooperation,
        )
        .await
    }
}
