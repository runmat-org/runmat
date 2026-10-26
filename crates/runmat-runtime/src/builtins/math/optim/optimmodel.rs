//! Problem-based expression ownership and compilation over original variable
//! definitions. Numerical arithmetic/solves remain in canonical domain owners.
use super::common::ScalarCooperation;
use crate::{context::RuntimeContext, execution::RootedValue, BuiltinResult};
use runmat_types::standard;
use runmat_value::optimization_model::{
    OptimizationExpressionKind as Kind, OptimizationExpressionNode as Node,
    OptimizationExpressionRef as ExpressionRef, OptimizationModelPayload as Payload,
};
use runmat_value::{ObjectInstance, SharedShape, Utf8Text, Value};
pub(crate) mod bindings;
mod constraints;
mod construct;
mod equation;
mod evaluation;
mod exitflag;
mod functions;
mod indexing;
mod lowering;
pub(crate) mod native_fields;
mod operators;
mod owner;
mod problem;
mod properties;
mod protocol;
mod registration;
mod solve_arguments;
#[cfg(test)]
mod tests;

use super::common::model_policy::Policy;

fn is_expression(object: &ObjectInstance) -> bool {
    object.class_name().is(standard::OPTIMIZATION_EXPRESSION)
        && object.selected_class().is_none()
        && matches!(
            object.optimization_model_payload(),
            Some(Payload::Expression(_))
        )
}
