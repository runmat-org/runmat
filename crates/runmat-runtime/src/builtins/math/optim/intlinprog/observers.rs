//! MILP event payloads, not continuous-solver iteration/gradient assumptions.
use super::*;
use common::output_callback::{self, ObserverGroup, OutputState};
use runmat_builtins::OptimOptionField as Field;

pub(super) struct Observers {
    options: Option<common::PreparedSolverOptions>,
    output_count: usize,
    plot_count: usize,
    started: f64,
    published_fval: Option<f64>,
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
}
pub(super) struct Snapshot {
    pub fval: f64,
    pub dualbound: f64,
    pub relativegap: Option<f64>,
    pub numnodes: Option<usize>,
    pub numfeaspoints: usize,
    pub elapsed: f64,
}
impl Observers {
    pub fn new(options: Option<common::PreparedSolverOptions>) -> BuiltinResult<Self> {
        let value = |field: Field| {
            options
                .as_ref()
                .and_then(|options| options.structure().fields.get(field.name()))
        };
        let output_count = output_callback::callback_count(
            value(Field::OutputFcn),
            output_callback::milp_form(Field::OutputFcn),
        )?;
        let plot_count = output_callback::callback_count(
            value(Field::PlotFcn),
            output_callback::milp_form(Field::PlotFcn),
        )?;
        #[cfg(not(feature = "plot-core"))]
        if plot_count > 0 {
            let mut error = crate::context::RuntimeCapabilityError::new(
                crate::context::RuntimeCapability::GraphicsPresentation,
                "intlinprog PlotFcn requires plot-core graphics support",
            )
            .into_runtime_error();
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
        Ok(Self {
            options,
            output_count,
            plot_count,
            started: 0.0,
            published_fval: None,
            #[cfg(feature = "plot-core")]
            layout: None,
        })
    }
    pub fn enabled(&self) -> bool {
        self.output_count > 0 || self.plot_count > 0
    }
    pub fn published_fval(&self) -> f64 {
        self.published_fval.unwrap_or(f64::INFINITY)
    }
    pub fn publish_fval(&mut self, value: f64) {
        self.published_fval = Some(value);
    }
    pub async fn notify(
        &mut self,
        state: OutputState,
        x: Value,
        snapshot: Snapshot,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.enabled() {
            return Ok(false);
        }
        if state == OutputState::Init {
            self.started = snapshot.elapsed;
        }
        cooperation.completed(0).await?;
        let runtime = cooperation.runtime().cloned();
        #[cfg(feature = "plot-core")]
        if self.plot_count > 0 && self.layout.is_none() {
            self.layout = Some(
                crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                    self.plot_count,
                    runtime.as_ref(),
                )?,
            );
        }
        let select = |index| -> BuiltinResult<()> {
            #[cfg(feature = "plot-core")]
            {
                self.layout
                    .as_ref()
                    .ok_or_else(|| invalid("MILP plot layout unavailable"))?
                    .select(index, runtime.as_ref())
            }
            #[cfg(not(feature = "plot-core"))]
            {
                let _ = index;
                Err(invalid("plot-core is unavailable"))
            }
        };
        let fields = [
            ("dualbound", Value::Num(snapshot.dualbound)),
            ("fval", Value::Num(snapshot.fval)),
            ("numfeaspoints", Value::Num(snapshot.numfeaspoints as f64)),
            (
                "numnodes",
                match snapshot.numnodes {
                    Some(n) => Value::Num(n as f64),
                    None => output_callback::empty(runtime.as_ref())?,
                },
            ),
            (
                "relativegap",
                match snapshot.relativegap {
                    Some(gap) => Value::Num(gap),
                    None => output_callback::empty(runtime.as_ref())?,
                },
            ),
            (
                "time",
                Value::Num((snapshot.elapsed - self.started).max(0.0)),
            ),
        ];
        let metadata = crate::execution::construct_struct_from_fields_in_optional_context(
            runtime.as_ref(),
            fields.iter().map(|(name, value)| (*name, value)),
        )?;
        let state = crate::call::argument_text::copy_str(state.name(), runtime.as_ref())
            .map(|text| Value::String(text.into()))?;
        let value = |field: Field| {
            self.options
                .as_ref()
                .and_then(|options| options.structure().fields.get(field.name()))
        };
        let groups = [
            ObserverGroup {
                value: value(Field::OutputFcn),
                form: output_callback::milp_form(Field::OutputFcn),
                select: None,
            },
            ObserverGroup {
                value: value(Field::PlotFcn),
                form: output_callback::milp_form(Field::PlotFcn),
                select: Some(&select),
            },
        ];
        output_callback::invoke_groups(NAME, &groups, x, metadata, state, cooperation).await
    }
}
