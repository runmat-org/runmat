//! A custom selector returns ordered, possibly repeated parent indices. This
//! is not the sorted-unique structural integer-column decoder.
use super::super::{common, invalid, numeric, options::Operator, ScalarCooperation, POLICY};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticSelection;
use runmat_value::Value;
use std::ops::ControlFlow;

pub(in super::super) async fn custom<I: super::super::callbacks::Invocation>(
    operator: &Operator<OptimGeneticSelection>,
    expectation: Value,
    parents: usize,
    members: usize,
    controls: &Value,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Vec<usize>>> {
    let count = u64::try_from(parents)
        .ok()
        .and_then(|count| runmat_value::IntValue::U64(count).to_exact_f64())
        .ok_or_else(|| invalid("nParents cannot be represented exactly for selection callback"))?;
    let returned = operator
        .call_with(
            [expectation, Value::Num(count), controls.clone()],
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
    returned
        .with_owned_async(|value| async {
            let values = common::constrained_numeric::read(
                value,
                "selected parents",
                common::constrained_numeric::Geometry::Vector,
                &numeric::INPUT_POLICY,
                cooperation,
            )
            .await?;
            let common::constrained_numeric::Numeric::Real { values, .. } = values else {
                return Err(invalid("selection must return real parent indices"));
            };
            if values.len() != parents {
                return Err(invalid("selection must return exactly nParents indices"));
            }
            let mut selected = POLICY.capacity(parents, cooperation.runtime())?;
            for value in values {
                let index = common::option_usize_value(
                    "ga",
                    Some(&Value::Num(value)),
                    "selected parent",
                    0,
                )?;
                if index == 0 || index > members {
                    return Err(invalid(
                        "selected parents must be valid 1-based population indices",
                    ));
                }
                selected.push(index - 1);
                cooperation.completed(1).await?;
            }
            Ok(selected)
        })
        .await
        .map(ControlFlow::Continue)
}
