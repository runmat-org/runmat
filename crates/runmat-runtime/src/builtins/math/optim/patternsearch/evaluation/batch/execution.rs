use super::*;
pub(in crate::builtins::math::optim::patternsearch) struct Execution {
    pub vectorized: bool,
    pub(super) objective: Option<CallbackBatch>,
    pub(super) constraints: Option<CallbackBatch>,
}
impl Execution {
    pub fn direct(&self) -> bool {
        !self.vectorized && self.objective.is_none()
    }
    pub fn select(
        controls: &super::super::super::controls::Controls,
        objective: &Value,
        constraints: Option<&Value>,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Self> {
        Self::select_with(
            controls.vectorized,
            controls.parallel,
            objective,
            constraints,
            runtime,
        )
    }
    pub fn select_with(
        vectorized: bool,
        parallel: CallbackParallelMode,
        objective: &Value,
        constraints: Option<&Value>,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let mode = if vectorized {
            CallbackParallelMode::Off
        } else {
            parallel
        };
        let mut selected_objective = CallbackBatch::select(runtime, objective, mode)?;
        let mut selected_constraints = constraints
            .map(|value| CallbackBatch::select(runtime, value, mode))
            .transpose()?
            .flatten();
        // A sweep has one effective placement. Auto can decline before any
        // submission; never replay an accepted worker result on the host.
        if constraints.is_some() && (selected_objective.is_none() || selected_constraints.is_none())
        {
            selected_objective = None;
            selected_constraints = None;
        }
        Ok(Self {
            vectorized,
            objective: selected_objective,
            constraints: selected_constraints,
        })
    }
    pub fn width(&self, complete: bool) -> usize {
        if !complete {
            return 1;
        }
        if self.vectorized {
            return usize::MAX;
        }
        self.objective.as_ref().map_or(1, |objective| {
            self.constraints
                .as_ref()
                .map_or(objective.width(), |constraints| {
                    objective.width().min(constraints.width())
                })
        })
    }
}
