use crate::{gather_if_needed_async, BuiltinResult};
use runmat_value::Value;

use super::error_context;
use super::options::Invocation;
use super::plan::Plan;

pub(super) async fn execute(function: Value, arguments: Vec<Value>) -> BuiltinResult<Value> {
    let plan = Plan::build(Invocation::parse(function, arguments)?)?;
    plan.callable
        .validate_outputs(crate::current_requested_outputs())?;
    if let Some(handler) = &plan.error_handler {
        handler.validate_outputs(crate::current_requested_outputs())?;
    }
    let execution = Execution::prepare(plan).await?;
    execution.run().await
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

    async fn run(&self) -> BuiltinResult<Value> {
        let requested = crate::current_requested_outputs();
        let mut collectors = (0..requested)
            .map(|_| super::output::Collector::new(self.plan.uniform_output))
            .collect::<Vec<_>>();
        for index in 0..self.plan.element_count {
            let result = self.evaluate(index, requested).await?;
            if requested == 0 {
                continue;
            }
            for (collector, value) in collectors
                .iter_mut()
                .zip(super::output::normalize(result, requested)?)
            {
                collector.push(value).await?;
            }
        }
        let values = collectors
            .into_iter()
            .map(|collector| collector.finish(&self.plan.shape))
            .collect::<BuiltinResult<Vec<_>>>()?;
        if requested == 1 {
            Ok(values.into_iter().next().expect("one output"))
        } else {
            Ok(Value::OutputList(values))
        }
    }

    async fn evaluate(&self, index: usize, requested: usize) -> BuiltinResult<Value> {
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
        match self.plan.callable.call(&arguments, requested).await {
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
                handler.call(&handler_arguments, requested).await
            }
        }
    }
}
