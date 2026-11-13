use super::*;
use common::output_callback::{self, CallbackForm, ObserverGroup, OutputState};
pub(in crate::builtins::math::optim) struct Observers<'a> {
    output: Option<&'a Value>,
    plots: Option<&'a Value>,
    shape: &'a [usize],
    scalar: bool,
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
}
impl<'a> Observers<'a> {
    pub fn new(options: &options::Options<'a>, shape: &'a [usize], scalar: bool) -> Self {
        Self {
            output: options.output,
            plots: options.plots,
            shape,
            scalar,
            #[cfg(feature = "plot-core")]
            layout: None,
        }
    }
    pub async fn notify(
        &mut self,
        state: OutputState,
        point: &evaluation::Point,
        iteration: usize,
        calls: usize,
        procedure: reporting::Procedure,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let outputs = output_callback::callback_count(self.output, CallbackForm::Handles)?;
        let plots = output_callback::callback_count(self.plots, CallbackForm::HandlesOrNames)?;
        if outputs == 0 && plots == 0 {
            return Ok(false);
        }
        cooperation.completed(0).await?;
        let runtime = cooperation.runtime().cloned();
        #[cfg(feature = "plot-core")]
        if plots > 0 && self.layout.is_none() {
            self.layout = Some(
                crate::builtins::plotting::state::OptimizationPlotLayout::prepare(
                    plots,
                    runtime.as_ref(),
                )?,
            );
        }
        #[cfg(not(feature = "plot-core"))]
        if plots > 0 {
            return Err(invalid("PlotFcns layout requires plot-core"));
        }
        let select = |index| -> BuiltinResult<()> {
            #[cfg(feature = "plot-core")]
            {
                self.layout
                    .as_ref()
                    .ok_or_else(|| invalid("plot callback layout is unavailable"))?
                    .select(index, runtime.as_ref())
            }
            #[cfg(not(feature = "plot-core"))]
            {
                let _ = index;
                Err(invalid("plot-core is unavailable"))
            }
        };
        let x = output_callback::vector(&point.x, self.shape, self.scalar, cooperation).await?;
        let procedure = crate::call::argument_text::copy_str(procedure.name(), runtime.as_ref())
            .map(|text| Value::String(text.into()))?;
        let fields = [
            ("iteration", Value::Num(iteration as f64)),
            ("funccount", Value::Num(calls as f64)),
            ("fval", Value::Num(point.f)),
            ("procedure", procedure),
        ];
        let metadata = crate::execution::construct_struct_from_fields_in_optional_context(
            runtime.as_ref(),
            fields.iter().map(|(name, value)| (*name, value)),
        )?;
        let state = crate::call::argument_text::copy_str(state.name(), runtime.as_ref())
            .map(|text| Value::String(text.into()))?;
        let groups = [
            ObserverGroup {
                value: self.output,
                form: CallbackForm::Handles,
                select: None,
            },
            ObserverGroup {
                value: self.plots,
                form: CallbackForm::HandlesOrNames,
                select: Some(&select),
            },
        ];
        output_callback::invoke_groups(NAME, &groups, x, metadata, state, cooperation).await
    }
}
