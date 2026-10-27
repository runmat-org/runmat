use super::*;
impl Evaluator<'_> {
    /// One real original selected call. Width-one worker/vectorized work never
    /// becomes an unadvertised host callback; the direct path allocates no wave.
    pub(in crate::builtins::math::optim::patternsearch) async fn invoke_one(
        &mut self,
        role: CallbackRole,
        argument: Value,
    ) -> BuiltinResult<RootedValueSequence> {
        let requested = role.requested_outputs();
        self.state.callbacks.charge(role, 1)?;
        let execution = self.context.execution.unwrap_or(&self.state.execution);
        let batch = match role {
            CallbackRole::Objective => execution.objective.as_ref(),
            CallbackRole::NonlinearConstraints => execution.constraints.as_ref(),
        };
        if let Some(batch) = batch {
            let mut arguments = buffer(1)?;
            arguments.push(vec![argument]);
            let (_, budget) = self.cooperation.runtime_budget();
            let mut returned = batch
                .execute(
                    arguments,
                    requested,
                    budget
                        .ok_or_else(|| invalid("selected callback has no original CPU budget"))?,
                )?
                .await?;
            if returned.len() != 1 {
                return Err(invalid("selected single callback cardinality changed"));
            }
            Ok(returned.pop().expect("one selected response"))
        } else {
            let target = match role {
                CallbackRole::Objective => self.context.objective,
                CallbackRole::NonlinearConstraints => {
                    self.context.constraints.expect("selected nonlinear target")
                }
            };
            common::call_function_sequence_in_context(
                self.cooperation.runtime(),
                target,
                [argument],
                requested,
            )
            .await
        }
    }
}
