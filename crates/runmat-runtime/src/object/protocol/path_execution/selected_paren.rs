//! Parentheses on selected non-array receivers with existing typed owners.

use runmat_value::Value;

use crate::execution::RootedValue;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexKind, ObjectIndexSelector, ObjectSubscript,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

#[derive(Clone, Copy)]
enum SelectedParenKind {
    SymbolicFunction,
    ForeignResource,
}

fn classify(base: &Value, step: &ObjectSubscript) -> Option<SelectedParenKind> {
    if step.kind() != ObjectIndexKind::Paren {
        return None;
    }
    match base {
        Value::Symbolic(expr) if expr.function_reference_signature().is_some() => {
            Some(SelectedParenKind::SymbolicFunction)
        }
        Value::Foreign(_) => Some(SelectedParenKind::ForeignResource),
        _ => None,
    }
}

pub(super) fn supports(base: &Value, step: Option<&ObjectSubscript>) -> bool {
    step.and_then(|step| classify(base, step)).is_some()
}

pub(super) async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    step: ObjectSubscript,
) -> Result<RootedValue, RuntimeError> {
    let kind = classify(&base, &step).ok_or_else(|| {
        semantic_error(
            "InvalidObjectSubscriptBase",
            "value does not support this subscript",
        )
    })?;
    let (ObjectIndexSelector::IndexValues { components }, _) = step.into_parts() else {
        return Err(semantic_error(
            "InvalidObjectSubscriptPath",
            "parentheses require value arguments",
        ));
    };
    if components.iter().any(|component| {
        matches!(
            component,
            ObjectIndexComponent::Colon | ObjectIndexComponent::End
        )
    }) {
        return Err(match kind {
            SelectedParenKind::SymbolicFunction => semantic_error(
                "UnsupportedSymbolicFunctionSelector",
                "symbolic function calls do not support colon or contextual end selector syntax",
            ),
            SelectedParenKind::ForeignResource => semantic_error(
                "InvalidObjectSubscriptPath",
                "foreign item indices require values, not colon or contextual end syntax",
            ),
        });
    }
    let arguments = super::callable::component_arguments(runtime, components)?;
    match (kind, base) {
        (SelectedParenKind::SymbolicFunction, Value::Symbolic(expr)) => {
            let (name, parameters) = expr
                .function_reference_signature()
                .expect("classified symbolic function reference");
            RootedValue::new(crate::indexing::symbolic::apply_function_reference(
                name,
                parameters,
                arguments.values(),
            )?)
            .map_err(|error| error.into_runtime_error())
        }
        (SelectedParenKind::ForeignResource, Value::Foreign(reference)) => {
            crate::foreign::index_foreign_resource_retained(runtime, reference, arguments.into())
                .await
        }
        _ => unreachable!("selected parentheses kind matches its receiver"),
    }
}
