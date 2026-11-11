//! Original invocation output/plot groups and actual diagnostic publication.
use super::super::{
    evaluation::Evaluator,
    outcome::{Outcome, Parameters, Termination},
};
use super::{bernstein::invalid, Domain};
use crate::builtins::math::optim::common::output_callback::{
    self, CallbackForm, ObserverGroup, OutputState,
};
use crate::BuiltinResult;
use runmat_builtins::OptimDisplay;
use runmat_value::Value;
pub(in crate::builtins::math::optim::fmincon) fn exitflag(
    termination: Termination,
) -> BuiltinResult<i32> {
    Ok(match termination {
        Termination::Optimality => 1,
        Termination::Step => 4,
        Termination::Function => 5,
        Termination::Limit | Termination::Time => 0,
        Termination::Observer => -1,
        Termination::NoFeasiblePoint => -2,
        Termination::Numerical => {
            return Err(invalid(
                "a reliable semi-infinite constrained step or mesh could not be obtained",
            ))
        }
    })
}
pub(in crate::builtins::math::optim::fmincon) fn description(
    termination: Termination,
) -> &'static str {
    match termination {
        Termination::Optimality=>"The current interpolated semi-infinite constraints and independent nonlinear KKT checks meet the requested tolerances; this is a local numerical result, not a global minimum certificate.",
        Termination::Step=>"The represented search direction or permitted step is below StepTolerance and the current constraints meet ConstraintTolerance.",
        Termination::Function=>"The magnitude of the measured objective directional derivative is below FunctionTolerance and the current constraints meet ConstraintTolerance.",
        Termination::Limit=>"An iteration, evaluated-point or mesh-resolution resource limit was reached.",
        Termination::Time=>"The selected internal feasibility deadline was reached.",
        Termination::Observer=>"The output or plot function requested termination.",
        Termination::NoFeasiblePoint=>"Local maximum-violation restoration found no improving feasible point; this is not a global nonlinear infeasibility certificate.",
        Termination::Numerical=>"A reliable semi-infinite constrained step or mesh could not be obtained.",
    }
}
impl Domain {
    pub fn configure_plots(
        &mut self,
        plots: Option<crate::execution::RootedValue>,
    ) -> BuiltinResult<()> {
        if let Some(value) = &plots {
            output_callback::validate_for_builtin(
                "fseminf",
                value.value(),
                CallbackForm::HandlesOrNames,
            )?;
            #[cfg(not(feature = "plot-core"))]
            if output_callback::callback_count(Some(value.value()), CallbackForm::HandlesOrNames)?
                > 0
            {
                return Err(invalid("PlotFcn requires plot-core"));
            }
        }
        self.plots = plots;
        Ok(())
    }
}
pub(in crate::builtins::math::optim::fmincon) async fn observe(
    out: &Outcome,
    evaluator: &mut Evaluator<'_>,
    parameters: &Parameters<'_>,
    phase: OutputState,
    direction: &[f64],
    alpha: Option<f64>,
) -> BuiltinResult<bool> {
    let domain = evaluator
        .semi_infinite
        .take()
        .expect("selected semi-infinite observer domain");
    #[cfg(feature = "plot-core")]
    let mut domain = domain;
    let result=async {
        let terminal=phase==OutputState::Done;
        let show=match parameters.display {
            OptimDisplay::Off|OptimDisplay::None=>false,
            OptimDisplay::Iter=>true,OptimDisplay::Final=>terminal,
            OptimDisplay::Notify=>terminal&&!matches!(out.termination,Termination::Optimality|Termination::Step|Termination::Function),
            OptimDisplay::Diagnose=>return Err(invalid("diagnostic Display is not supported by fseminf")),
        };
        if show||(phase==OutputState::Init&&domain.controls.diagnostics) {
            let _entry=evaluator.cooperation.runtime().map(crate::context::RuntimeContext::enter);
            crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                format!("fseminf active-set {}: iteration={} funcCount={} objectiveCalls={} constraintCalls={} fval={:?} constrviolation={:?} firstorderopt={:?}",
                    phase.name(),out.iterations,evaluator.budget.counts().points,evaluator.budget.counts().objective_calls,
                    evaluator.budget.counts().constraint_calls,out.point.as_ref().map(|p|p.f),out.violation,out.optimality))?;
            if terminal&&show {
                crate::console::record_console_line(crate::console::ConsoleStream::Stdout,description(out.termination))?;
                if domain.controls.detailed_display {
                    crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                        format!("ConstraintTolerance={} OptimalityTolerance={} StepTolerance={} FunctionTolerance={:?} meshTolerance={} TolConSQP={}",
                            parameters.feasibility,parameters.optimality,parameters.step,parameters.function,
                            domain.controls.mesh_tolerance,domain.controls.inner_tolerance))?;
                }
            }
            if phase==OutputState::Init&&domain.controls.diagnostics {
                let rows=if domain.state.has_current(){domain.state.frozen()?.generation().rows.len()}else{0};
                crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                    format!("ntheta={} variables={} admittedInterpolatedRows={}",domain.state.family_count(),out.x.len(),rows))?;
            }
        }
        let plots=domain.plots.as_ref().map(|value|value.value());
        let plot_count=output_callback::callback_count(plots,CallbackForm::HandlesOrNames)?;
        let output_count=output_callback::callback_count(parameters.callback,CallbackForm::Handles)?;
        if plot_count==0&&output_count==0 {return Ok(false);}
        let runtime=evaluator.cooperation.runtime().cloned();
        #[cfg(feature="plot-core")]
        if plot_count>0&&domain.layout.is_none() {
            domain.layout=Some(crate::builtins::plotting::state::OptimizationPlotLayout::prepare(plot_count,runtime.as_ref())?);
        }
        let select=|index|->BuiltinResult<()> {
            #[cfg(feature="plot-core")]
            {domain.layout.as_ref().ok_or_else(||invalid("plot callback layout is unavailable"))?.select(index,runtime.as_ref())}
            #[cfg(not(feature="plot-core"))]
            {let _=index;Err(invalid("plot callback layout requires plot-core"))}
        };
        let x=output_callback::vector(&out.x,&evaluator.shape,evaluator.scalar,evaluator.cooperation).await?;
        let gradient=match out.point.as_ref().filter(|p|!p.projected_objective).and_then(|p|p.gradient.as_deref()) {
            Some(gradient)=>output_callback::vector(gradient,&evaluator.shape,evaluator.scalar,evaluator.cooperation).await?,
            None=>output_callback::empty(runtime.as_ref())?,
        };
        let direction=if direction.is_empty(){output_callback::empty(runtime.as_ref())?}
            else{output_callback::vector(direction,&[direction.len(),1],false,evaluator.cooperation).await?};
        let optional=|value:Option<f64>|->BuiltinResult<Value> {
            match value {Some(value)=>Ok(Value::Num(value)),None=>output_callback::empty(runtime.as_ref())}
        };
        let mut fields=runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("iteration",Value::Num(out.iterations as f64));
        fields.insert("funccount",Value::Num(evaluator.budget.counts().points as f64));
        fields.insert("fval",optional(out.point.as_ref().map(|p|p.f))?);
        fields.insert("gradient",gradient);fields.insert("searchdirection",direction);
        fields.insert("constrviolation",optional(out.violation)?);fields.insert("firstorderopt",optional(out.optimality)?);
        fields.insert("stepsize",Value::Num(out.step));fields.insert("lssteplength",optional(alpha.or(out.alpha))?);
        fields.insert("procedure",Value::from(if phase==OutputState::Init{"initial"}else if terminal{"done"}else{"working-set SQP"}));
        let metadata=crate::execution::RootedValue::in_optional_context(Value::Struct(fields.finish()),runtime.as_ref())?;
        let groups=[
            ObserverGroup{value:parameters.callback,form:CallbackForm::Handles,select:None},
            ObserverGroup{value:plots,form:CallbackForm::HandlesOrNames,select:Some(&select)},
        ];
        output_callback::invoke_groups("fseminf",&groups,x,metadata,Value::from(phase.name()),evaluator.cooperation).await
    }.await;
    evaluator.semi_infinite = Some(domain);
    result
}
