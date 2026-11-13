//! Mutable GA observers return a state generation, not a stop Boolean. Roots
//! survive every semantic call and option edits are admitted by the controller.
use super::{common, invalid, ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{OptimGeneticObserverState, OptimGeneticStateField, OptimOptionField};
use runmat_value::Value;
#[cfg(test)]
mod tests;

pub(super) struct Observers {
    output: Vec<crate::call::descriptor::RepeatableCallable>,
    plot: Vec<crate::call::descriptor::RepeatableCallable>,
}
pub(super) struct Returned {
    pub state: RootedValue,
    pub controls: RootedValue,
    pub options_changed: bool,
    pub stopped: bool,
}
impl Observers {
    pub fn select(controls: &Value, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        let Value::Struct(controls) = controls else {
            return Err(invalid("observer options must be a scalar structure"));
        };
        let fields = common::ScalarOptionFields::from_struct_for_solver(
            Some(controls),
            runmat_builtins::OptimSolver::Ga,
        );
        let select =
            |field, form| -> BuiltinResult<Vec<crate::call::descriptor::RepeatableCallable>> {
                let Some(value) = fields.exact(field) else {
                    return Ok(Vec::new());
                };
                common::output_callback::validate_for_builtin("ga", value, form)?;
                let targets = targets(value);
                let mut selected = super::POLICY.capacity(targets.len(), cooperation.runtime())?;
                for target in targets {
                    selected.push(super::callbacks::select(target, cooperation)?);
                }
                Ok(selected)
            };
        Ok(Self {
            output: select(
                OptimOptionField::OutputFcn,
                common::output_callback::CallbackForm::Handles,
            )?,
            plot: select(
                OptimOptionField::PlotFcn,
                common::output_callback::CallbackForm::HandlesOrNames,
            )?,
        })
    }
    pub fn plot_count(&self) -> usize {
        self.plot.len()
    }
    pub fn active(&self) -> bool {
        !self.output.is_empty() || self.plot_count() != 0
    }
    #[cfg(test)]
    pub async fn notify(
        &self,
        state: RootedValue,
        controls: RootedValue,
        phase: OptimGeneticObserverState,
        plots_due: bool,
        select_plot: Option<&dyn Fn(usize) -> BuiltinResult<()>>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Returned> {
        match self
            .notify_with(
                state,
                controls,
                phase,
                plots_due,
                select_plot,
                &mut super::callbacks::Direct,
                cooperation,
            )
            .await?
        {
            std::ops::ControlFlow::Continue(returned) => Ok(returned),
            std::ops::ControlFlow::Break((never, _)) => match never {},
        }
    }
    pub async fn notify_with<I: super::callbacks::Invocation>(
        &self,
        mut state: RootedValue,
        mut controls: RootedValue,
        phase: OptimGeneticObserverState,
        plots_due: bool,
        select_plot: Option<&dyn Fn(usize) -> BuiltinResult<()>>,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<(I::Stop, Returned), Returned>> {
        let mut stopped = stop_requested(state.value())?;
        let mut options_changed = false;
        for (plot, callbacks) in [
            (false, self.output.as_slice()),
            (true, self.plot.as_slice()),
        ] {
            if plot && !plots_due {
                continue;
            }
            for (index, callback) in callbacks.iter().enumerate() {
                cooperation.completed(0).await?;
                if let Some(stop) = invocation.stop(super::callbacks::Event::Observer(phase)) {
                    return Ok(std::ops::ControlFlow::Break((
                        stop,
                        Returned {
                            state,
                            controls,
                            options_changed,
                            stopped,
                        },
                    )));
                }
                if plot {
                    if let Some(select) = select_plot {
                        select(index)?;
                    }
                }
                let arguments = [
                    controls.value().clone(),
                    state.value().clone(),
                    Value::from(phase.name()),
                ];
                let prepared = crate::call::BuiltinArguments::from_borrowed(
                    &arguments,
                    cooperation.runtime(),
                )?;
                let returned = match invocation
                    .call(
                        callback,
                        prepared,
                        if plot { 1 } else { 3 },
                        super::callbacks::Event::Observer(phase),
                        cooperation,
                    )
                    .await?
                {
                    std::ops::ControlFlow::Continue(returned) => returned,
                    std::ops::ControlFlow::Break(stop) => {
                        return Ok(std::ops::ControlFlow::Break((
                            stop,
                            Returned {
                                state,
                                controls,
                                options_changed,
                                stopped,
                            },
                        )))
                    }
                };
                if returned.len() != if plot { 1 } else { 3 } {
                    return Err(invalid(
                        "GA observers must return state, or state/options/optchanged for OutputFcn",
                    ));
                }
                let mut returned = {
                    let _scope = cooperation
                        .runtime()
                        .map(crate::context::RuntimeContext::enter);
                    returned.try_into_values()?
                };
                let next_state = returned.next().expect("requested state");
                stopped |= stop_requested(&next_state)?;
                let next_state =
                    RootedValue::in_optional_context(next_state, cooperation.runtime())?;
                if !plot {
                    let next_controls = returned.next().expect("requested controls");
                    let changed = returned.next().expect("requested optchanged");
                    if phase != OptimGeneticObserverState::Interrupt && changed_flag(&changed)? {
                        if !matches!(next_controls, Value::Struct(_)) {
                            return Err(invalid(
                                "changed observer options must be a scalar structure",
                            ));
                        }
                        controls =
                            RootedValue::in_optional_context(next_controls, cooperation.runtime())?;
                        options_changed = true;
                    }
                }
                state = next_state;
                cooperation.completed(1).await?;
                // Do not short-circuit on StopFlag: every selected observer
                // owns its call. A later cleared flag cannot revoke a stop.
            }
        }
        Ok(std::ops::ControlFlow::Continue(Returned {
            state,
            controls,
            options_changed,
            stopped,
        }))
    }
}
fn targets(value: &Value) -> &[Value] {
    match value {
        Value::Tensor(value) if value.is_empty() => &[],
        Value::Cell(value) => value.data(),
        value => std::slice::from_ref(value),
    }
}
fn stop_requested(state: &Value) -> BuiltinResult<bool> {
    let Value::Struct(state) = state else {
        return Err(invalid("GA observer state must be a scalar structure"));
    };
    let flag = state
        .fields
        .get(OptimGeneticStateField::StopFlag.name())
        .ok_or_else(|| invalid("GA observer state requires StopFlag"))?;
    crate::builtins::common::validation::value_text_scalar_string(flag)
        .map(|flag| !flag.is_empty())
        .ok_or_else(|| invalid("GA StopFlag must be scalar text or a character row vector"))
}
fn changed_flag(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(value) if value.data.len() == 1 => Ok(value.data[0] != 0),
        Value::Num(value) if *value == 0. || *value == 1. => Ok(*value == 1.),
        Value::Tensor(value) if value.len() == 1 => {
            let value = crate::builtins::common::tensor::tensor_value_f64(value, 0);
            if value == 0. || value == 1. {
                Ok(value == 1.)
            } else {
                Err(invalid("optchanged must be a logical scalar"))
            }
        }
        _ => Err(invalid("optchanged must be a logical scalar")),
    }
}
