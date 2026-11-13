//! Trust-region observer and result schema. QN's ten-field observer remains
//! unchanged; TR never invents a line-search length or a final Hessian replay.
use super::super::evaluation::ObjectiveState;
use super::{
    iteration::{Outcome, Termination},
    *,
};
use common::output_callback::{CallbackForm, ObserverGroup, OutputState};

pub(super) struct Snapshot<'a> {
    pub state: &'a ObjectiveState,
    pub shape: &'a [usize],
    pub scalar: bool,
    pub iterations: usize,
    pub evaluations: usize,
    pub cg_iterations: usize,
    pub step_size: f64,
    pub radius: f64,
}

pub(super) async fn observe(
    callback: Option<&Value>,
    phase: OutputState,
    snapshot: Snapshot<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let Some(callback) = callback else {
        return Ok(false);
    };
    let x = common::output_callback::vector(
        &snapshot.state.x,
        snapshot.shape,
        snapshot.scalar,
        cooperation,
    )
    .await?;
    let gradient = common::output_callback::vector(
        &snapshot.state.grad,
        snapshot.shape,
        snapshot.scalar,
        cooperation,
    )
    .await?;
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iteration", Value::Num(snapshot.iterations as f64));
    fields.insert("funccount", Value::Num(snapshot.evaluations as f64));
    fields.insert("fval", Value::Num(snapshot.state.f));
    fields.insert("gradient", gradient);
    fields.insert(
        "firstorderopt",
        Value::Num(optimality(&snapshot.state.grad)),
    );
    fields.insert("stepsize", Value::Num(snapshot.step_size));
    fields.insert("cgiterations", Value::Num(snapshot.cg_iterations as f64));
    fields.insert("trustregionradius", Value::Num(snapshot.radius));
    let metadata = crate::execution::RootedValue::in_optional_context(
        Value::Struct(fields.finish()),
        cooperation.runtime(),
    )?;
    common::output_callback::invoke_groups(
        NAME,
        &[ObserverGroup {
            value: Some(callback),
            form: CallbackForm::Handles,
            select: None,
        }],
        x,
        metadata,
        Value::from(phase.name()),
        cooperation,
    )
    .await
}

pub(super) fn optimality(gradient: &[f64]) -> f64 {
    gradient.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

pub(super) fn progress(
    display: DisplayMode,
    initial: bool,
    snapshot: Snapshot<'_>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if display != DisplayMode::Iter {
        return Ok(());
    }
    let _entry = runtime.map(RuntimeContext::enter);
    if initial {
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout," Iteration  Func-count             f(x)   CG-iterations   Trust-radius  First-order optimality")?;
    }
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!(
            "{:>10}  {:>10}  {:>15.6e}  {:>14}  {:>13.6e}  {:>22.6e}",
            snapshot.iterations,
            snapshot.evaluations,
            snapshot.state.f,
            snapshot.cg_iterations,
            snapshot.radius,
            optimality(&snapshot.state.grad)
        ),
    )
}

pub(in crate::builtins::math::optim::fminunc) async fn finalize(
    outcome: Outcome,
    shape: &[usize],
    scalar: bool,
    options: &FminuncOptions,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    if outcome.termination == Termination::Numerical {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            cooperation.runtime(),
            &runmat_builtins::FMINUNC_NUMERICAL_FAILURE_EXTENSION,
            NAME,
        )?;
    }
    let message = format!(
        "{} Iterations: {}, FuncCount: {}.",
        outcome.termination.description(),
        outcome.iterations,
        outcome.evaluations
    );
    if matches!(options.display, DisplayMode::Final | DisplayMode::Iter)
        || options.display == DisplayMode::Notify && outcome.termination.flag() <= 0
    {
        let _entry = cooperation.runtime().map(RuntimeContext::enter);
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!(
                "fminunc: fval = {:.6e}, firstorderopt = {:.6e}, exitflag = {}. {message}",
                outcome.state.f,
                optimality(&outcome.state.grad),
                outcome.termination.flag()
            ),
        )?;
    }
    let Some(requested) = crate::output_count::current_output_count() else {
        return publish_result(
            vector_to_value(NAME, outcome.state.x, shape, scalar)?,
            cooperation.runtime(),
        );
    };
    if requested == 0 {
        return Ok(crate::execution::RootedValueSequence::empty());
    }
    let mut outputs = buffer(requested)?;
    outputs.push(vector_to_value(NAME, outcome.state.x, shape, scalar)?);
    if requested >= 2 {
        outputs.push(Value::Num(outcome.state.f));
    }
    if requested >= 3 {
        outputs.push(Value::Num(outcome.termination.flag() as f64));
    }
    if requested >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("iterations", Value::Num(outcome.iterations as f64));
        fields.insert("funcCount", Value::Num(outcome.evaluations as f64));
        fields.insert("algorithm", Value::from("trust-region"));
        fields.insert("firstorderopt", Value::Num(optimality(&outcome.state.grad)));
        fields.insert("stepsize", Value::Num(outcome.step_size));
        fields.insert("cgiterations", Value::Num(outcome.cg_iterations as f64));
        fields.insert("message", Value::from(message));
        outputs.push(Value::Struct(fields.finish()));
    }
    if requested >= 5 {
        outputs.push(vector_to_value(NAME, outcome.state.grad, shape, scalar)?);
    }
    if requested >= 6 {
        if let Some(model) = outcome.model {
            return model
                .into_value()
                .with_owned_async(|value| async move {
                    outputs.push(value);
                    // Successor registration occurs under the original
                    // model lease, before it or any Hinfo descendant retires.
                    publish_result(Value::output_list(outputs), cooperation.runtime())
                })
                .await;
        }
        outputs.push(common::output_callback::empty(cooperation.runtime())?);
    }
    publish_result(Value::output_list(outputs), cooperation.runtime())
}
