//! Immutable original-selected observers run at measured update boundaries.
//! Their retained result roots span stop decoding; they do not spend the
//! objective allowance or resolve a callback name again after selection.
use super::{
    controller::Controller,
    invalid,
    notice::{Notice, Observed},
    public_values as values,
};
use crate::{
    builtins::math::optim::common::{
        output_callback::{self, OutputState},
        ScalarCooperation,
    },
    call::{descriptor::RepeatableCallable, BuiltinArguments},
    execution::RootedValue,
    BuiltinResult,
};
use runmat_value::Value;

pub(super) fn select(value: &Value, c: &ScalarCooperation) -> BuiltinResult<RepeatableCallable> {
    let unavailable = || {
        crate::build_runtime_error(
            "surrogateopt: callbacks require original selected-program source admission",
        )
        .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    };
    let runtime = c.runtime().ok_or_else(unavailable)?;
    let program = runtime.selected_graph_program().ok_or_else(unavailable)?;
    let callsite = {
        let _entry = runtime.enter();
        crate::callsite::current_function_input_callsite()
            .or_else(crate::callsite::current_callsite)
    };
    program
        .select_repeatable_callable(runtime, RootedValue::in_context(value.clone(), runtime)?)
        .map(|selected| selected.with_callsite(callsite))
}
fn targets(value: Option<&Value>) -> &[Value] {
    match value {
        None => &[],
        Some(Value::Tensor(v)) if v.is_empty() => &[],
        Some(Value::Cell(v)) => v.data(),
        Some(v) => std::slice::from_ref(v),
    }
}
fn selected(
    value: Option<&Value>,
    c: &ScalarCooperation,
) -> BuiltinResult<Vec<RepeatableCallable>> {
    if let Some(value) = value {
        output_callback::validate_for_builtin(
            super::NAME,
            value,
            output_callback::CallbackForm::HandlesOrNames,
        )?;
    }
    let mut result = values::capacity(targets(value).len())?;
    for target in targets(value) {
        result.push(select(target, c)?);
    }
    Ok(result)
}
pub(super) struct SelectedObservers {
    outputs: Vec<RepeatableCallable>,
    plots: Vec<RepeatableCallable>,
}
impl SelectedObservers {
    pub(super) fn with_output(
        mut self,
        value: Option<&Value>,
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        self.outputs = selected(value, c)?;
        Ok(self)
    }
    pub(super) fn with_plot(
        mut self,
        value: Option<&Value>,
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        self.plots = selected(value, c)?;
        Ok(self)
    }
    pub(super) fn prepare(
        output: Option<&Value>,
        plot: Option<&Value>,
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            outputs: selected(output, c)?,
            plots: selected(plot, c)?,
        })
    }
    pub(super) fn from_restored(
        outputs: Vec<RepeatableCallable>,
        plots: Vec<RepeatableCallable>,
    ) -> Self {
        Self { outputs, plots }
    }
    pub(super) fn outputs(&self) -> &[RepeatableCallable] {
        &self.outputs
    }
    pub(super) fn plots(&self) -> &[RepeatableCallable] {
        &self.plots
    }
    pub(super) fn admit_layout(self, c: &ScalarCooperation) -> BuiltinResult<Observers> {
        Observers::from_selected(self.outputs, self.plots, c)
    }
}

pub(super) struct Observers {
    outputs: Vec<RepeatableCallable>,
    plots: Vec<RepeatableCallable>,
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
}
impl Observers {
    pub(super) fn outputs(&self) -> &[RepeatableCallable] {
        &self.outputs
    }
    pub(super) fn plots(&self) -> &[RepeatableCallable] {
        &self.plots
    }
    fn from_selected(
        outputs: Vec<RepeatableCallable>,
        plots: Vec<RepeatableCallable>,
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        #[cfg(not(feature = "plot-core"))]
        let _ = c;
        #[cfg(not(feature = "plot-core"))]
        if !plots.is_empty() {
            return Err(invalid("PlotFcn requires the actual plot-core UI owner"));
        }
        #[cfg(feature = "plot-core")]
        let layout = if plots.is_empty() {
            None
        } else {
            Some(
                crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                    plots.len(),
                    c.runtime(),
                )?,
            )
        };
        Ok(Self {
            outputs,
            plots,
            #[cfg(feature = "plot-core")]
            layout,
        })
    }
    pub(super) async fn notify(
        &mut self,
        phase: OutputState,
        controller: &Controller<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.notify_notice(phase, Notice::live(controller)?, c)
            .await
            .map(|observed| observed.stop)
    }
    pub(super) async fn notify_notice(
        &mut self,
        phase: OutputState,
        notice: Notice<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Observed> {
        if self.outputs.is_empty() && self.plots.is_empty() {
            return Ok(Observed {
                stop: false,
                frame: notice.frame,
            });
        }
        let (point, metadata) = metadata(phase, &notice, c).await?;
        let state = Value::string(crate::call::argument_text::copy_str(
            phase.name(),
            c.runtime(),
        )?);
        let args = [point.value().clone(), metadata.value().clone(), state];
        let mut stop = false;
        for callback in &self.outputs {
            stop |= invoke(callback, &args, c).await?;
        }
        for (index, callback) in self.plots.iter().enumerate() {
            #[cfg(feature = "plot-core")]
            self.layout
                .as_ref()
                .expect("selected plot layout")
                .select(index, c.runtime())?;
            #[cfg(not(feature = "plot-core"))]
            let _ = index;
            stop |= invoke(callback, &args, c).await?;
        }
        // Original rooted metadata and point remain live across every output
        // and plot await, including callbacks that retain their arguments.
        Ok(Observed {
            stop,
            frame: notice.frame,
        })
    }
}
async fn invoke(
    callback: &RepeatableCallable,
    args: &[Value],
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    c.completed(0).await?;
    let returned = callback
        .invoke_retained(
            c.runtime(),
            BuiltinArguments::from_borrowed(args, c.runtime())?,
            1,
        )
        .await?;
    c.completed(0).await?;
    let returned = {
        let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
        returned
            .resolve_sequence(
                runmat_types::SequenceUse::from_requested_outputs(
                    runmat_types::RequestedOutputCount::Exactly(1),
                ),
                crate::sequence::SequenceResolutionContext::default(),
            )?
            .try_into_single()?
    };
    match returned.value() {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(value) if value.data.len() == 1 => Ok(value.data[0] != 0),
        _ => Err(invalid(
            "OutputFcn and PlotFcn must return a logical scalar stop flag",
        )),
    }
}
async fn metadata(
    phase: OutputState,
    notice: &Notice<'_>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<(RootedValue, RootedValue)> {
    let history = notice.history;
    let tracker = notice.frame.tracker;
    let init = phase == OutputState::Init;
    let best = (!init)
        .then_some(tracker.best)
        .flatten()
        .map(|i| history.trial(i));
    let current = (!init)
        .then_some(tracker.current)
        .flatten()
        .map(|i| history.trial(i));
    let incumbent = (!init)
        .then_some(tracker.incumbent)
        .flatten()
        .map(|i| history.trial(i));
    let point = RootedValue::in_optional_context(values::point(best, c).await?, c.runtime())?;
    let count = notice.frame.evaluations.0;
    let iteration = if phase == OutputState::Iter {
        count.saturating_sub(1)
    } else {
        count
    };
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("constrviolation", values::violation(best, c)?);
    fields.insert("currentConstrviolation", values::violation(current, c)?);
    fields.insert("currentFlag", values::origin(current, c)?);
    fields.insert("currentFval", values::objective(current, c)?);
    fields.insert("currentIneq", values::inequalities(current, c).await?);
    fields.insert("currentX", values::point(current, c).await?);
    fields.insert(
        "elapsedtime",
        Value::Num(notice.frame.elapsed.as_secs_f64()),
    );
    fields.insert("flag", values::origin(best, c)?);
    fields.insert("funccount", Value::Num(count as f64));
    fields.insert("fval", values::objective(best, c)?);
    fields.insert("incumbentConstrviolation", values::violation(incumbent, c)?);
    fields.insert("incumbentIneq", values::inequalities(incumbent, c).await?);
    fields.insert("incumbentFlag", values::origin(incumbent, c)?);
    fields.insert("incumbentFval", values::objective(incumbent, c)?);
    fields.insert("incumbentX", values::point(incumbent, c).await?);
    fields.insert("ineq", values::inequalities(best, c).await?);
    fields.insert("iteration", Value::Num(iteration as f64));
    fields.insert("surrogateReset", Value::Bool(notice.frame.reset));
    fields.insert(
        "surrogateResetCount",
        Value::Num(notice.frame.resets as f64),
    );
    // The selected tolerance is carried explicitly for shared constrained
    // plotting; no plot helper infers solver identity from field spelling.
    fields.insert("constraintTolerance", Value::Num(notice.tolerance));
    fields.insert("feasibleCount", Value::Num(tracker.feasible as f64));
    fields.insert("checkpointReplay", Value::Bool(notice.replay));
    fields.insert("checkpointResume", Value::Bool(notice.resume));
    let metadata = RootedValue::in_optional_context(Value::Struct(fields.finish()), c.runtime())?;
    Ok((point, metadata))
}
