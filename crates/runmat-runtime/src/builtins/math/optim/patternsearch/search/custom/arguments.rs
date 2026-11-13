//! Public search arguments are normal language callback inputs, not a second oracle.
use super::*;
use crate::builtins::math::optim::common::output_callback;
pub(super) async fn prepare(
    base: &Point,
    input: &Input<'_>,
    options: &Options,
    iteration: usize,
    mesh: f64,
    method: &'static str,
    scaling: &Scaling,
    score: f64,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Vec<Value>> {
    let cooperation = &mut *evaluator.cooperation;
    let runtime = cooperation.runtime().cloned();
    let x = output_callback::vector(
        &base.x,
        evaluator.context.shape,
        evaluator.context.scalar,
        cooperation,
    )
    .await?;
    let mut scales = super::super::super::buffer(scaling.dimension())?;
    for axis in 0..scaling.dimension() {
        scales.push(scaling.factor(axis));
        cooperation.completed(1).await?;
    }
    let mut metadata = runmat_value::UnboundedStructValueBuilder::new();
    metadata.insert("x", x.clone());
    metadata.insert("fval", Value::Num(score));
    metadata.insert("iteration", Value::Num(iteration as f64));
    metadata.insert(
        "funccount",
        Value::Num(evaluator.state.objective_calls as f64),
    );
    metadata.insert(
        "scale",
        common::vector_to_value(
            "patternsearch",
            scales,
            evaluator.context.shape,
            evaluator.context.scalar,
        )?,
    );
    metadata.insert("meshsize", Value::Num(mesh));
    metadata.insert("method", Value::from(method));
    let nonempty = |value: Option<&Value>| {
        value.is_some_and(|value| !super::super::super::input::empty(value))
    };
    let linear = input.inequality.is_some_and(|(a, _)| nonempty(Some(a)))
        || input.equality.is_some_and(|(a, _)| nonempty(Some(a)));
    metadata.insert(
        "problemtype",
        Value::from(if linear {
            "linearconstraints"
        } else if nonempty(input.lower) || nonempty(input.upper) {
            "boundconstraints"
        } else {
            "unconstrained"
        }),
    );
    // Capacity includes the scoped merit handle, inserted only by the admitted
    // child's factory. Numeric Tensor/Sparse payloads keep immutable backings.
    let mut values = super::super::super::buffer(10)?;
    values.push(x);
    for value in [
        input.inequality.map(|v| v.0),
        input.inequality.map(|v| v.1),
        input.equality.map(|v| v.0),
        input.equality.map(|v| v.1),
        input.lower,
        input.upper,
    ] {
        values.push(match value {
            Some(value) => value.clone(),
            None => output_callback::empty(runtime.as_ref())?,
        });
    }
    values.push(Value::Struct(metadata.finish()));
    // This is the actual public options argument, not a tracking snapshot.
    values.push(options.value().clone());
    Ok(values)
}
