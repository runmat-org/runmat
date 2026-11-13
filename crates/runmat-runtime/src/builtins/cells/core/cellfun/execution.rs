use crate::execution::{RootedValue, RootedValuesBuilder};
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_value::Value;

use super::error_context;
use super::options::Invocation;
use super::plan::Plan;

pub(super) async fn execute(
    context: Option<&crate::context::RuntimeContext>,
    function: Value,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _function_root =
        crate::builtins::common::mapped_callable::root_callable_value(&function, context)?;
    let plan = Plan::build(Invocation::parse(function, arguments)?)?;
    plan.callable
        .validate_outputs(crate::current_requested_outputs())?;
    if let Some(handler) = &plan.error_handler {
        handler.validate_outputs(crate::current_requested_outputs())?;
    }
    let execution = Execution::prepare(context, plan).await?;
    execution.run().await
}

struct Execution<'a> {
    context: Option<&'a crate::context::RuntimeContext>,
    plan: Plan,
    host_extra_arguments: crate::call::BuiltinArguments,
}

impl<'a> Execution<'a> {
    async fn prepare(
        context: Option<&'a crate::context::RuntimeContext>,
        plan: Plan,
    ) -> BuiltinResult<Self> {
        let gather = plan.extra_arguments.try_map_preserving_roots_async(
            plan.extra_arguments.len(),
            context,
            async |index| gather_if_needed_async(&plan.extra_arguments[index]).await,
        );
        let host_extra_arguments = match context {
            Some(context) => context.scope(gather).await?,
            None => gather.await?,
        };
        Ok(Self {
            context,
            plan,
            host_extra_arguments,
        })
    }

    async fn run(&self) -> BuiltinResult<RootedValue> {
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
        let mut outputs = RootedValuesBuilder::with_capacity(collectors.len());
        for collector in collectors {
            collector
                .finish(&self.plan.shape)?
                .with_owned(|value| outputs.push(value))
                .map_err(|error| error.into_runtime_error())?;
        }
        outputs
            .finish()
            .map_err(|error| error.into_runtime_error())?
            .with_owned_output_storage(|values| {
                let value = if requested == 1 {
                    values
                        .try_into_first()
                        .map_err(|_| {
                            crate::sequence::sequence_error_to_runtime(
                                runmat_value::ValueSequenceError::SharedOutputStorage,
                            )
                        })?
                        .expect("one output")
                } else {
                    Value::OutputList(values)
                };
                RootedValue::new(value).map_err(|error| error.into_runtime_error())
            })
    }

    async fn evaluate(&self, index: usize, requested: usize) -> BuiltinResult<RootedValue> {
        let count = self
            .plan
            .cells
            .len()
            .checked_add(self.host_extra_arguments.len())
            .ok_or_else(|| super::error::internal("cellfun: callback argument count overflow"))?;
        let gather =
            self.plan
                .cells
                .try_map_preserving_roots_async(count, self.context, async |slot| {
                    if slot < self.plan.cells.len() {
                        let Value::Cell(cell) = &self.plan.cells[slot] else {
                            unreachable!("partitioned cell inputs");
                        };
                        // The canonical Range-backed iterator maps nth in O(1),
                        // including page offsets for N-D cells; physical slots
                        // inside each page are row-major, not logical order.
                        let value = cell.iter_column_major().nth(index).ok_or_else(|| {
                            super::error::internal("cellfun: cell storage does not match its shape")
                        })?;
                        gather_if_needed_async(value).await
                    } else {
                        Ok(self.host_extra_arguments[slot - self.plan.cells.len()].clone())
                    }
                });
        let arguments = match self.context {
            Some(context) => context.scope(gather).await?,
            None => gather.await?,
        };
        match self
            .plan
            .callable
            .call_retained(self.context, arguments.fork(), requested)
            .await
        {
            Ok(value) => Ok(value),
            Err(callback_error) => {
                let Some(handler) = &self.plan.error_handler else {
                    return Err(callback_error);
                };
                let prefix =
                    error_context::value(self.context, &callback_error, index, &self.plan.shape)?;
                let handler_arguments =
                    crate::builtins::common::mapped_callable::prepend_rooted_argument(
                        &arguments,
                        prefix,
                        self.context,
                    )
                    .await?;
                handler
                    .call_retained(self.context, handler_arguments, requested)
                    .await
            }
        }
    }
}
