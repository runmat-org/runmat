use crate::builtins::common::uniform_scalar_output::UniformScalarCollector;
use crate::{gather_if_needed_async, make_cell_with_shape, BuiltinResult};
use runmat_value::Value;

use super::error;
use super::error_context;
use super::options::Invocation;
use super::plan::Plan;

pub(super) async fn execute(function: Value, arguments: Vec<Value>) -> BuiltinResult<Value> {
    let plan = Plan::build(Invocation::parse(function, arguments)?)?;
    let execution = Execution::prepare(plan).await?;
    if execution.plan.uniform_output {
        execution.uniform().await
    } else {
        execution.nonuniform().await
    }
}

struct Execution {
    plan: Plan,
    host_extra_arguments: Vec<Value>,
}

impl Execution {
    async fn prepare(plan: Plan) -> BuiltinResult<Self> {
        let mut host_extra_arguments = Vec::with_capacity(plan.extra_arguments.len());
        for argument in &plan.extra_arguments {
            host_extra_arguments.push(gather_if_needed_async(argument).await?);
        }
        Ok(Self {
            plan,
            host_extra_arguments,
        })
    }

    async fn uniform(&self) -> BuiltinResult<Value> {
        let mut collector = UniformScalarCollector::new();
        for index in 0..self.plan.element_count {
            let value = self.evaluate(index).await?;
            collector
                .push(&gather_if_needed_async(&value).await?)
                .map_err(super::output::map_error)?;
        }
        collector
            .finish(&self.plan.shape)
            .map_err(super::output::map_error)
    }

    async fn nonuniform(&self) -> BuiltinResult<Value> {
        let mut output = Vec::with_capacity(self.plan.element_count);
        for index in 0..self.plan.element_count {
            output.push(self.evaluate(index).await?);
        }
        make_cell_with_shape(output, self.plan.shape.clone())
            .map_err(|reason| error::internal(format!("cellfun: {reason}")))
    }

    async fn evaluate(&self, index: usize) -> BuiltinResult<Value> {
        let mut cell_values = Vec::with_capacity(self.plan.cells.len());
        for cell in &self.plan.cells {
            let value = cell
                .data
                .get(index)
                .cloned()
                .unwrap_or(Value::Num(f64::NAN));
            cell_values.push(gather_if_needed_async(&value).await?);
        }
        let mut arguments = Vec::with_capacity(cell_values.len() + self.host_extra_arguments.len());
        arguments.extend(cell_values.iter().cloned());
        arguments.extend(self.host_extra_arguments.iter().cloned());
        match self.plan.callable.call(&arguments).await {
            Ok(value) => Ok(value),
            Err(callback_error) => {
                let Some(handler) = &self.plan.error_handler else {
                    return Err(callback_error);
                };
                let mut handler_arguments = Vec::with_capacity(arguments.len() + 1);
                handler_arguments.push(error_context::value(
                    &callback_error,
                    index,
                    &self.plan.shape,
                )?);
                handler_arguments.extend(arguments);
                handler.call(&handler_arguments).await
            }
        }
    }
}
