//! Options-first mutable observers; one rooted original metadata generation.
use super::{
    common, evaluation, inputs, invalid,
    options::Options,
    state::{State, Termination},
    ScalarCooperation,
};
use crate::{execution::RootedValue, BuiltinResult};
use common::output_callback::OutputState;
use runmat_value::Value;
pub(super) fn logical(value: &Value, role: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(v) => Ok(*v),
        Value::LogicalArray(v) if v.data.len() == 1 => Ok(v.data[0] != 0),
        _ => Err(invalid(format!("{role} must return a logical scalar"))),
    }
}
pub(super) async fn metadata(
    state: &State,
    count: usize,
    started: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let temperature = common::output_callback::vector(
        &state.temperature,
        &[state.temperature.len(), 1],
        false,
        c,
    )
    .await?;
    let k = common::output_callback::vector(&state.k, &[state.k.len(), 1], false, c).await?;
    let fields = [
        ("x", state.current.value().clone()),
        ("fval", Value::Num(state.current_f)),
        ("bestx", state.best.value().clone()),
        ("bestfval", Value::Num(state.best_f)),
        ("temperature", temperature),
        ("iteration", Value::Num(state.iteration as f64)),
        ("funccount", Value::Num(count as f64)),
        ("t0", Value::Num(started)),
        ("k", k),
        ("averagechange", Value::Num(state.stall.average())),
        ("stallwindowfull", Value::Bool(state.stall.full())),
    ];
    crate::execution::construct_struct_from_fields_in_optional_context(
        c.runtime(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}
pub(super) struct Observers {
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
    plots: usize,
}
pub(super) struct Notice {
    pub phase: OutputState,
    pub count: usize,
    pub started: f64,
    pub reason: Option<Termination>,
}
fn callbacks(value: Option<&Value>) -> &[Value] {
    match value {
        None => &[],
        Some(Value::Cell(v)) => v.data(),
        Some(v) if inputs::is_empty(v) => &[],
        Some(v) => std::slice::from_ref(v),
    }
}
fn arguments(callback: &Value, base: &[Value]) -> BuiltinResult<(Value, Vec<Value>)> {
    let mut args = base.to_vec();
    if let Value::Cell(v) = callback {
        let (target, extra) = v
            .data()
            .split_first()
            .ok_or_else(|| invalid("empty callback parameter cell"))?;
        args.extend_from_slice(extra);
        Ok((target.clone(), args))
    } else {
        Ok((callback.clone(), args))
    }
}
impl Observers {
    pub fn new(options: &Options, c: &ScalarCooperation) -> BuiltinResult<Self> {
        let count = callbacks(options.plot.as_ref()).len();
        #[cfg(not(feature = "plot-core"))]
        if count > 0 {
            return Err(invalid("PlotFcn requires the actual plot-core UI owner"));
        }
        #[cfg(feature = "plot-core")]
        let layout = if count > 0 {
            Some(
                crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                    count,
                    c.runtime(),
                )?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "plot-core"))]
        let _ = c;
        Ok(Self {
            #[cfg(feature = "plot-core")]
            layout,
            plots: count,
        })
    }
    pub async fn notify(
        &mut self,
        notice: Notice,
        state: &State,
        options: &mut Options,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Notice {
            phase,
            count,
            started,
            reason,
        } = notice;
        let show = match options.display {
            runmat_builtins::OptimDisplay::None | runmat_builtins::OptimDisplay::Notify => {
                return Err(invalid("unsupported SA Display mode"))
            }
            runmat_builtins::OptimDisplay::Off => false,
            runmat_builtins::OptimDisplay::Final => phase == OutputState::Done,
            runmat_builtins::OptimDisplay::Iter | runmat_builtins::OptimDisplay::Diagnose => {
                phase == OutputState::Done
                    || state.iteration.is_multiple_of(options.display_interval)
            }
        };
        if show {
            let mean = state
                .temperature
                .iter()
                .map(|v| v / state.temperature.len() as f64)
                .sum::<f64>();
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                format!("simulannealbnd {}: iteration={} funccount={} bestfval={} fval={} temperature={}",
                    phase.name(),state.iteration,count,state.best_f,state.current_f,mean))?;
            if phase == OutputState::Init
                && options.display == runmat_builtins::OptimDisplay::Diagnose
            {
                crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                    format!("DataType={} dimensions={} MaxFunctionEvaluations={} MaxStallIterations={} ReannealInterval={}",
                        options.datatype.name(),state.k.len(),options.evaluations,options.stall,options.reanneal))?;
            }
            if let Some(reason) = reason {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    reason.message(),
                )?;
            }
        }
        let output = options.output.clone();
        if callbacks(output.as_ref()).is_empty() && callbacks(options.plot.as_ref()).is_empty() {
            return Ok(false);
        }
        let metadata = metadata(state, count, started, c).await?;
        let flag = Value::string(crate::call::argument_text::copy_str(
            phase.name(),
            c.runtime(),
        )?);
        let mut stop = false;
        for callback in callbacks(output.as_ref()) {
            let (target, args) = arguments(
                callback,
                &[
                    options.public.value().clone(),
                    metadata.value().clone(),
                    flag.clone(),
                ],
            )?;
            let returned = evaluation::invoke(&target, &args, 3, c).await?;
            if returned.len() != 3 {
                return Err(invalid("OutputFcn must return stop,options,optchanged"));
            }
            let mut values = {
                let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
                returned.try_into_values()?
            };
            stop |= logical(&values.next().expect("three"), "OutputFcn stop")?;
            let replacement = values.next().expect("three");
            let changed = logical(&values.next().expect("three"), "OutputFcn optchanged")?;
            if changed {
                // The original rooted output sequence/iterator survives every
                // readback and the complete replacement's root publication.
                let mut replacement = Options::parse(replacement, c).await?;
                if replacement.datatype != options.datatype {
                    return Err(invalid(
                        "OutputFcn cannot replace the decision domain's DataType",
                    ));
                }
                replacement.resolve(state.k.len(), c).await?;
                // A changed hybrid target is selected with original authority
                // before its next invocation by the iteration controller.
                *options = replacement;
            }
        }
        let plot = options.plot.clone();
        let count_plots = callbacks(plot.as_ref()).len();
        if self.plots != count_plots {
            #[cfg(feature = "plot-core")]
            {
                self.layout = if count_plots > 0 {
                    Some(
                        crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                            count_plots,
                            c.runtime(),
                        )?,
                    )
                } else {
                    None
                };
            }
            self.plots = count_plots;
        }
        if phase != OutputState::Iter || state.iteration.is_multiple_of(options.plot_interval) {
            for (index, callback) in callbacks(plot.as_ref()).iter().enumerate() {
                #[cfg(feature = "plot-core")]
                {
                    self.layout
                        .as_ref()
                        .ok_or_else(|| invalid("plot layout missing"))?
                        .select(index, c.runtime())?;
                    let (target, args) = arguments(
                        callback,
                        &[
                            options.public.value().clone(),
                            metadata.value().clone(),
                            flag.clone(),
                        ],
                    )?;
                    let returned = evaluation::invoke(&target, &args, 1, c).await?;
                    let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
                    stop |= logical(returned.try_into_single()?.value(), "PlotFcn")?;
                }
                #[cfg(not(feature = "plot-core"))]
                {
                    let _ = (index, callback);
                    return Err(invalid("plot-core unavailable"));
                }
            }
        }
        Ok(stop)
    }
}
