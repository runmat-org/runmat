//! One original-context numerical evaluation of an immutable expression DAG.
//! Canonical numerical builtins do the arithmetic; this owner schedules graph
//! dependencies, shape checks, assignment slots and original variable values.
use super::*;
use runmat_gc::GcHandle;
use std::collections::HashMap;
mod plan;
pub(super) mod selection;
mod state;
mod variable;
pub(super) use plan::Plan;
pub(super) use state::count as element_count;
use state::State;
pub(super) use variable::prepare as prepare_point;
pub(super) use variable::validate_names as validate_variable_names;

pub(super) async fn evaluate(
    root: GcHandle,
    point: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let inputs = policy.roots(std::iter::once(point), cooperation.runtime())?;
    let plan = plan::Plan::prepare(root, policy, cooperation).await?;
    let mut evaluated = run(plan, point, inputs, policy, cooperation).await?;
    evaluated.take(root, policy, cooperation).await
}
pub(super) enum NumericUndefined {
    Nonreal,
}
fn required(
    value: Result<Evaluated, NumericUndefined>,
    policy: Policy,
) -> BuiltinResult<Evaluated> {
    value.map_err(|NumericUndefined::Nonreal| {
        policy.invalid("expression numerical evaluation requires real double storage")
    })
}
pub(super) struct Evaluated {
    values: HashMap<GcHandle, State>,
    uses: HashMap<GcHandle, usize>,
}
impl Evaluated {
    pub(super) async fn take(
        &mut self,
        root: GcHandle,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<RootedValue> {
        take(root, &mut self.values, &mut self.uses, policy, cooperation)
            .await?
            .into_ready(policy, cooperation)
            .await
    }
}
pub(super) async fn evaluate_roots(
    roots: &[GcHandle],
    point: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Evaluated> {
    let inputs = policy.roots(std::iter::once(point), cooperation.runtime())?;
    let plan = plan::Plan::prepare_roots(roots, policy, cooperation).await?;
    run(plan, point, inputs, policy, cooperation).await
}
async fn run(
    mut plan: plan::Plan,
    point: &Value,
    inputs: crate::execution::RootedValueLease,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Evaluated> {
    let uses = std::mem::take(&mut plan.uses);
    required(
        run_plan(&plan, None, uses, point, inputs, policy, cooperation).await?,
        policy,
    )
}
/// The same real model evaluator exposes numerical nonreal output as a typed
/// trial stop. Operational errors are not intercepted or reinterpreted.
pub(super) async fn evaluate_trial(
    plan: &Plan,
    selected: Option<&selection::Selection>,
    roots: &[GcHandle],
    point: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Result<Evaluated, NumericUndefined>> {
    let inputs = policy.roots(std::iter::once(point), cooperation.runtime())?;
    let mut uses = match selected {
        Some(selection) => selection.uses(plan, policy, cooperation).await?,
        None => {
            let mut uses = HashMap::new();
            uses.try_reserve(plan.uses.len())
                .map_err(|e| policy.reserve(e))?;
            for (&handle, &count) in &plan.uses {
                uses.insert(handle, count);
                cooperation.completed(1).await?;
            }
            uses
        }
    };
    // Dependency counts belong to the immutable plan; output retention belongs
    // to this request. A solver can request an intermediate alongside an
    // ancestor objective without evaluating the graph twice or retiring that
    // intermediate after its last internal edge.
    let planned_roots = selected.map_or(plan.roots.as_slice(), |selection| {
        selection.roots.as_slice()
    });
    if roots != planned_roots {
        for root in planned_roots {
            let count = uses
                .get_mut(root)
                .ok_or_else(|| policy.invalid("planned output root has no use count"))?;
            *count = count
                .checked_sub(1)
                .ok_or_else(|| policy.invalid("planned output root-use count underflow"))?;
            cooperation.completed(1).await?;
        }
        for root in roots {
            if !plan.nodes.contains_key(root)
                || selected.is_some_and(|selection| !selection.nodes.contains(root))
            {
                return Err(policy
                    .invalid("requested output root is outside the selected expression graph"));
            }
            let count = uses.get_mut(root).ok_or_else(|| {
                policy.invalid("requested output root has no planned dependency count")
            })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| policy.invalid("requested output root-use count overflow"))?;
            cooperation.completed(1).await?;
        }
    }
    run_plan(
        plan,
        selected.map(|selection| &selection.nodes),
        uses,
        point,
        inputs,
        policy,
        cooperation,
    )
    .await
}
async fn run_plan(
    plan: &Plan,
    selected: Option<&std::collections::HashSet<GcHandle>>,
    mut uses: HashMap<GcHandle, usize>,
    point: &Value,
    _inputs: crate::execution::RootedValueLease,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Result<Evaluated, NumericUndefined>> {
    variable::validate_names(plan, policy, cooperation).await?;
    let point = variable::prepare(plan, point, policy, cooperation).await?;
    let mut values = HashMap::new();
    for &handle in &plan.order {
        if selected.is_some_and(|nodes| !nodes.contains(&handle)) {
            cooperation.completed(1).await?;
            continue;
        }
        let node = plan.nodes.get(&handle).expect("planned node").node();
        let value = match &node.kind {
            Kind::EmptyInitializer | Kind::Zero => {
                state::count(&node.shape, policy)?;
                State::uniform(node.shape.clone(), Some(0.0))
            }
            Kind::Unassigned => {
                state::count(&node.shape, policy)?;
                State::uniform(node.shape.clone(), None)
            }
            Kind::Labels { source } => {
                take(*source, &mut values, &mut uses, policy, cooperation).await?
            }
            Kind::Constant(value) => State::ready(RootedValue::in_optional_context(
                value.clone(),
                cooperation.runtime(),
            )?),
            Kind::Variable(value) => {
                State::ready(variable::read(value, &point, policy, cooperation).await?)
            }
            Kind::Unary { operator, operand } => {
                let input = take(*operand, &mut values, &mut uses, policy, cooperation)
                    .await?
                    .into_ready(policy, cooperation)
                    .await?;
                State::ready(
                    invoke(
                        operator
                            .overload_name()
                            .ok_or_else(|| policy.invalid("nonarithmetic node operator"))?
                            .display_name(),
                        &[input.value().clone()],
                        policy,
                        cooperation,
                    )
                    .await?,
                )
            }
            Kind::Binary {
                operator,
                left,
                right,
            } => {
                let left = take(*left, &mut values, &mut uses, policy, cooperation)
                    .await?
                    .into_ready(policy, cooperation)
                    .await?;
                let right = take(*right, &mut values, &mut uses, policy, cooperation)
                    .await?
                    .into_ready(policy, cooperation)
                    .await?;
                State::ready(
                    invoke(
                        operator
                            .overload_name()
                            .ok_or_else(|| policy.invalid("nonarithmetic node operator"))?
                            .display_name(),
                        &[left.value().clone(), right.value().clone()],
                        policy,
                        cooperation,
                    )
                    .await?,
                )
            }
            Kind::Index { source, indices } => {
                let source = take(*source, &mut values, &mut uses, policy, cooperation).await?;
                let mut slots = runmat_value::exact_capacity_vec(indices.len())
                    .map_err(|source| policy.storage(source))?;
                for &index in indices.iter() {
                    slots.push(source.at(index, policy)?);
                    cooperation.completed(1).await?;
                }
                State::partial(node.shape.clone(), slots)
            }
            Kind::Assign {
                source,
                indices,
                replacement,
                selection_shape,
            } => {
                let mut source = take(*source, &mut values, &mut uses, policy, cooperation).await?;
                let replacement =
                    take(*replacement, &mut values, &mut uses, policy, cooperation).await?;
                let width = replacement.count(policy)?;
                let replacement_shape = replacement.shape()?;
                let layout = crate::indexing::write_slice::AssignmentRhsLayout::multidimensional(
                    &replacement_shape,
                    width,
                    selection_shape,
                )?;
                if !source.is_partial() {
                    let count = source.count(policy)?;
                    let mut slots = runmat_value::exact_capacity_vec(count)
                        .map_err(|source| policy.storage(source))?;
                    for index in 0..count {
                        slots.push(source.at(index, policy)?);
                        cooperation.completed(1).await?;
                    }
                    source = State::partial(node.shape.clone(), slots);
                }
                let slots = &mut source
                    .partial_mut()
                    .expect("converted assignment slots")
                    .slots;
                for (position, &index) in indices.iter().enumerate() {
                    let slot = slots.get_mut(index).ok_or_else(|| {
                        policy.invalid("expression assignment exceeds original geometry")
                    })?;
                    *slot = replacement.at(layout.source_index(position), policy)?;
                    cooperation.completed(1).await?;
                }
                source
            }
            Kind::Concatenate {
                dimension,
                operands,
            } => {
                let mut roots = runmat_value::exact_capacity_vec(operands.len())
                    .map_err(|source| policy.storage(source))?;
                for &operand in operands {
                    roots.push(
                        take(operand, &mut values, &mut uses, policy, cooperation)
                            .await?
                            .into_ready(policy, cooperation)
                            .await?,
                    );
                }
                let mut arguments = runmat_value::exact_capacity_vec(
                    roots
                        .len()
                        .checked_add(1)
                        .ok_or_else(|| policy.invalid("concatenation arity overflow"))?,
                )
                .map_err(|source| policy.storage(source))?;
                arguments.push(Value::Int(runmat_value::IntValue::U64(
                    u64::try_from(
                        dimension
                            .checked_add(1)
                            .ok_or_else(|| policy.invalid("concatenation dimension overflow"))?,
                    )
                    .map_err(|_| policy.invalid("concatenation dimension exceeds uint64"))?,
                )));
                arguments.extend(roots.iter().map(|root| root.value().clone()));
                State::ready(invoke("cat", &arguments, policy, cooperation).await?)
            }
            Kind::Builtin {
                function,
                operands,
                arguments,
            } => {
                let mut roots = runmat_value::exact_capacity_vec(operands.len())
                    .map_err(|source| policy.storage(source))?;
                for &operand in operands {
                    roots.push(
                        take(operand, &mut values, &mut uses, policy, cooperation)
                            .await?
                            .into_ready(policy, cooperation)
                            .await?,
                    );
                }
                let mut inputs = runmat_value::exact_capacity_vec(
                    roots
                        .len()
                        .checked_add(arguments.len())
                        .ok_or_else(|| policy.invalid("builtin arity overflow"))?,
                )
                .map_err(|source| policy.storage(source))?;
                inputs.extend(roots.iter().map(|root| root.value().clone()));
                inputs.extend(arguments.iter().cloned());
                State::ready(invoke(&function.0, &inputs, policy, cooperation).await?)
            }
        };
        if let Some(output) = value.ready_ref() {
            if matches!(output.value(), Value::Complex(..) | Value::ComplexTensor(_))
                || matches!(output.value(), Value::SparseTensor(array) if array.is_complex())
            {
                let shape =
                    crate::builtins::common::shape::value_dimensions_borrowed(output.value())?;
                if !crate::builtins::common::shape::array_shapes_equal(&shape, &node.shape) {
                    return Err(
                        policy.invalid("numeric evaluation changed the compiled expression shape")
                    );
                }
                return Ok(Err(NumericUndefined::Nonreal));
            }
            let shape = construct::numeric_shape(output.value(), policy, cooperation).await?;
            if !crate::builtins::common::shape::array_shapes_equal(&shape, &node.shape) {
                return Err(
                    policy.invalid("numeric evaluation changed the compiled expression shape")
                );
            }
        }
        values
            .try_reserve(1)
            .map_err(|source| policy.reserve(source))?;
        values.insert(handle, value);
        cooperation.completed(1).await?;
    }
    Ok(Ok(Evaluated { values, uses }))
}

async fn take(
    handle: GcHandle,
    values: &mut HashMap<GcHandle, State>,
    uses: &mut HashMap<GcHandle, usize>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<State> {
    let count = uses
        .get_mut(&handle)
        .ok_or_else(|| policy.invalid("expression edge has no planned use"))?;
    if *count == 0 {
        return Err(policy.invalid("expression edge used after retirement"));
    }
    *count -= 1;
    if *count == 0 {
        values
            .remove(&handle)
            .ok_or_else(|| policy.invalid("expression dependency was not evaluated"))
    } else {
        values
            .get(&handle)
            .ok_or_else(|| policy.invalid("expression dependency was not evaluated"))?
            .duplicate(policy, cooperation)
            .await
    }
}
pub(super) async fn invoke(
    name: &str,
    arguments: &[Value],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    cooperation.completed(0).await?;
    let output = match cooperation.runtime() {
        Some(runtime) => crate::call_builtin_async_in_context(runtime, name, arguments).await?,
        None => {
            crate::context::PreparedUnselectedScope::prepare(None)?
                .scope(crate::call_builtin_async(name, arguments))
                .await?
        }
    };
    let _ = policy;
    cooperation.completed(0).await?;
    Ok(output)
}
