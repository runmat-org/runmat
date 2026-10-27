//! Pattern output functions return three outputs and can replace options.
//! They are not the unrelated boolean three-argument optimization observer.
use super::{
    evaluation::Evaluator, invalid, iteration::Engine, option_metadata::Policy, options::Options,
    state::Stop,
};
use crate::builtins::math::optim::common::output_callback::{self, CallbackForm, OutputState};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{OptimDisplay, OptimOptionField as Field};
use runmat_value::Value;
mod callbacks;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Regular(OutputState),
    /// Only nonlinear subproblem output functions receive this phase. They
    /// may request stop but cannot change the enclosing options generation.
    Interrupt,
}
impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Regular(state) => state.name(),
            Self::Interrupt => "interrupt",
        }
    }
}

pub(super) struct Domain {
    pub equalities: bool,
    pub linear: bool,
    pub nonlinear: bool,
    pub policy: Policy,
}
pub(super) struct Notification {
    pub stop: bool,
    pub replacement: Option<Options>,
}
pub(super) struct Observers {
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
    plot_count: usize,
}
impl Observers {
    pub fn new(options: &Options, evaluator: &Evaluator<'_>) -> BuiltinResult<Self> {
        let count = output_callback::callback_count(
            options.callback(Field::PlotFcn),
            CallbackForm::HandlesOrNames,
        )?;
        #[cfg(not(feature = "plot-core"))]
        {
            let _ = evaluator;
            if count > 0 {
                return Err(invalid("PlotFcn requires plot-core"));
            }
        }
        Ok(Self {
            #[cfg(feature = "plot-core")]
            layout: if count > 0 {
                Some(
                    crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                        count,
                        evaluator.cooperation.runtime(),
                    )?,
                )
            } else {
                None
            },
            plot_count: count,
        })
    }

    pub async fn notify(
        &mut self,
        phase: Phase,
        engine: &Engine,
        reason: Option<Stop>,
        options: &Options,
        domain: &Domain,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Notification> {
        self.report(phase, engine, reason, options, evaluator)?;
        let outputs = output_callback::callback_count(
            options.callback(Field::OutputFcn),
            CallbackForm::Handles,
        )?;
        let plots = output_callback::callback_count(
            options.callback(Field::PlotFcn),
            CallbackForm::HandlesOrNames,
        )?;
        if outputs == 0 && (plots == 0 || phase == Phase::Interrupt) {
            return Ok(Notification {
                stop: false,
                replacement: None,
            });
        }
        let metadata = self
            .metadata(phase, engine, domain.nonlinear, evaluator)
            .await?;
        metadata
            .with_owned_async(|metadata| async move {
                self.callbacks(phase, metadata, engine, options, domain, evaluator)
                    .await
            })
            .await
    }

    fn report(
        &self,
        phase: Phase,
        engine: &Engine,
        reason: Option<Stop>,
        options: &Options,
        evaluator: &Evaluator<'_>,
    ) -> BuiltinResult<()> {
        if phase == Phase::Interrupt {
            return Ok(());
        }
        let display = options.controls.display;
        let show = match display {
            OptimDisplay::Off | OptimDisplay::None => false,
            OptimDisplay::Final => phase == Phase::Regular(OutputState::Done),
            OptimDisplay::Iter => true,
            OptimDisplay::Diagnose => true,
            OptimDisplay::Notify => {
                return Err(invalid("Notify display is not supported by patternsearch"))
            }
        };
        if !show {
            return Ok(());
        }
        let _scope = evaluator
            .cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        if phase == Phase::Regular(OutputState::Init) && display == OptimDisplay::Diagnose {
            crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                format!("patternsearch classic: variables={} PollMethod={} objectiveBudget={} completedIterationBudget={} MeshTolerance={} ConstraintTolerance={} nonlinear={}",
                    engine.point.x.len(), options.controls.polling.name(), options.controls.evaluations,
                    options.controls.iterations, options.controls.mesh.tolerance,
                    options.controls.constraint_tolerance, evaluator.context.constraints.is_some()))?;
        }
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
            format!("patternsearch {}: iteration={} funccount={} constraintcalls={} fval={} meshsize={} method={}",
                phase.name(), engine.iterations, evaluator.state.objective_calls, evaluator.state.constraint_calls,
                engine.point.f, engine.mesh.size(), engine.method.name()))?;
        if let Some(reason) = reason {
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                reason.message(),
            )?;
        }
        Ok(())
    }

    async fn metadata(
        &self,
        phase: Phase,
        engine: &Engine,
        nonlinear: bool,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<RootedValue> {
        let runtime = evaluator.cooperation.runtime().cloned();
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "x",
            output_callback::vector(
                &engine.point.x,
                evaluator.context.shape,
                evaluator.context.scalar,
                evaluator.cooperation,
            )
            .await?,
        );
        let interrupted = phase == Phase::Interrupt;
        fields.insert(
            "iteration",
            Value::Num(if interrupted {
                engine.inner_iterations
            } else {
                engine.iterations
            } as f64),
        );
        let fval = if interrupted {
            engine.inner_objective(evaluator).await?
        } else {
            Some(engine.point.f)
        };
        fields.insert(
            "fval",
            match fval {
                Some(value) => Value::Num(value),
                None => output_callback::empty(runtime.as_ref())?,
            },
        );
        fields.insert("meshsize", Value::Num(engine.mesh.size()));
        fields.insert(
            "funccount",
            Value::Num(evaluator.state.objective_calls as f64),
        );
        fields.insert("method", Value::from(engine.method.name()));
        let optional = |value: Option<f64>| match value {
            Some(value) => Ok(Value::Num(value)),
            None => output_callback::empty(runtime.as_ref()),
        };
        let change = if interrupted {
            engine.inner_change
        } else {
            engine.change
        };
        fields.insert("TolFun", optional(change.map(|change| change.f))?);
        fields.insert("TolX", optional(change.map(|change| change.x))?);
        if nonlinear {
            fields.insert(
                "nonlinineq",
                output_callback::vector(
                    &engine.point.inequality,
                    &[engine.point.inequality.len(), 1],
                    false,
                    evaluator.cooperation,
                )
                .await?,
            );
            fields.insert(
                "nonlineq",
                output_callback::vector(
                    &engine.point.equality,
                    &[engine.point.equality.len(), 1],
                    false,
                    evaluator.cooperation,
                )
                .await?,
            );
        }
        RootedValue::in_optional_context(Value::Struct(fields.finish()), runtime.as_ref())
    }
}
