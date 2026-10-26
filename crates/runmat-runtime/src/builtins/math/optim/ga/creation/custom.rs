//! Custom creation keeps callback results rooted through normal row assignment.
use super::super::callbacks::Invocation;
use super::super::{common, invalid, options, population, ScalarCooperation, POLICY};
use crate::{execution::RootedValue, BuiltinResult};
use common::population::projection::Arguments;
use runmat_builtins::OptimGeneticCreation;
use runmat_value::Value;
use std::ops::ControlFlow;

/// A numeric custom creator returns its whole generation. Supplied rows take
/// precedence without copying the callback's full population backing.
pub(super) async fn numeric<I: Invocation>(
    original: super::Draft,
    creator: &options::Operator<OptimGeneticCreation>,
    problem: &Value,
    controls: &Value,
    bitstring: bool,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, super::Draft>> {
    let variables = original.variables();
    let nvars = u64::try_from(variables)
        .ok()
        .and_then(|count| runmat_value::IntValue::U64(count).to_exact_f64())
        .ok_or_else(|| invalid("nvars is not exactly representable for the creation callback"))?;
    let returned = creator
        .call_with(
            [Value::Num(nvars), problem.clone(), controls.clone()],
            1,
            invoker,
            cooperation,
        )
        .await?;
    let returned = match returned {
        ControlFlow::Continue(returned) => returned,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    let returned = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_single()?
    };
    // Keep the result forest rooted while its storage is inspected/materialized.
    let value = returned
        .with_owned_async(|value| async {
            super::super::population::generation::numeric_tensor(
                value,
                original.members(),
                variables,
                cooperation,
            )
            .await
            .map(Value::Tensor)
        })
        .await?;
    let mut created = super::Draft::new(
        Some(value),
        original.members(),
        variables,
        bitstring,
        cooperation,
    )
    .await?;
    for row in 0..original.missing().start {
        let point = original.row(row, cooperation).await?;
        created.set(row, &point, bitstring, cooperation).await?;
    }
    // Preserve the original supplied-prefix boundary for subsequent creation
    // and admission, rather than mistaking callback-produced rows for seeds.
    created.set_supplied(original.missing().start)?;
    Ok(ControlFlow::Continue(created))
}

pub(in super::super) async fn create_with<I: Invocation>(
    initial: Option<Value>,
    members: usize,
    variables: usize,
    creator: &mut options::Operator<OptimGeneticCreation>,
    problem: &Value,
    controls: &Value,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, (population::Generation, usize)>> {
    let mut initial = if let Some(value) = initial {
        if super::super::inputs::raw::unspecified(&value) {
            None
        } else {
            let root = RootedValue::in_optional_context(value, cooperation.runtime())?;
            Some(
                match population::custom::Custom::from_initial(root, members, invoker, cooperation)
                    .await?
                {
                    ControlFlow::Continue(value) => value,
                    ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
                },
            )
        }
    } else {
        None
    };
    if initial
        .as_ref()
        .is_some_and(|initial| initial.members() == members)
    {
        return Ok(ControlFlow::Continue((
            population::Generation::Custom {
                value: initial.take().expect("complete original generation"),
                variables,
            },
            members,
        )));
    }
    if creator.custom().is_none() {
        return Err(invalid(
            "a partial custom population requires a custom CreationFcn",
        ));
    }
    creator.bind(cooperation)?;
    let nvars = u64::try_from(variables)
        .ok()
        .and_then(|count| runmat_value::IntValue::U64(count).to_exact_f64())
        .ok_or_else(|| invalid("nvars is not exactly representable for the creation callback"))?;
    let returned = creator
        .call_with(
            [Value::Num(nvars), problem.clone(), controls.clone()],
            1,
            invoker,
            cooperation,
        )
        .await?;
    let returned = match returned {
        ControlFlow::Continue(returned) => returned,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    let returned = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_single()?
    };
    let mut created =
        match population::custom::Custom::from_result(returned, members, invoker, cooperation)
            .await?
        {
            ControlFlow::Continue(created) => created,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
    let supplied = initial.as_ref().map_or(0, |initial| initial.members());
    if let Some(initial) = initial {
        if initial.members() != 0 {
            let mut rows = POLICY.capacity(initial.members(), cooperation.runtime())?;
            for row in 0..initial.members() {
                rows.push(row);
                cooperation.completed(1).await?;
            }
            if let ControlFlow::Break(stop) = created
                .replace_rows(&rows, &initial, invoker, cooperation)
                .await?
            {
                return Ok(ControlFlow::Break(stop));
            }
        }
    }
    Ok(ControlFlow::Continue((
        population::Generation::Custom {
            value: created,
            variables,
        },
        supplied,
    )))
}
