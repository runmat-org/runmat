use super::*;

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(super) fn exception_stack(
        &mut self,
        value: &'a ExceptionValue,
        index: usize,
        depth: u16,
    ) -> Result<(), ContractError> {
        if let Some(frame) = value.stack.get(index) {
            self.budget.add_text(frame)?;
            self.tasks
                .push(Task::ExceptionStack(value, index + 1, depth));
        } else if let Some(runtime) = &value.runtime {
            if let Some(builtin) = &runtime.builtin {
                self.budget.add_text(builtin)?;
            }
            if let Some(task_id) = &runtime.task_id {
                self.budget.add_text(task_id)?;
            }
            self.budget.add_elements(runtime.call_frames.len() as u64)?;
            self.tasks.push(Task::ExceptionFrames(value, 0, depth));
        } else {
            self.budget.add_elements(value.causes.len() as u64)?;
            self.tasks
                .push(Task::ExceptionCauses(&value.causes, 0, depth));
        }
        Ok(())
    }

    pub(super) fn exception_frames(
        &mut self,
        value: &'a ExceptionValue,
        index: usize,
        depth: u16,
    ) -> Result<(), ContractError> {
        let runtime = value.runtime.as_ref().expect("metadata task");
        if let Some(frame) = runtime.call_frames.get(index) {
            self.budget.add_text(&frame.function)?;
            match &frame.source {
                ExceptionFrameSource::Admitted(id) => {
                    let absent = match self.budget.sources {
                        Sources::Local => false,
                        Sources::Missing => true,
                        Sources::Map(map) => !map.contains_source(*id),
                    };
                    if absent {
                        return Err(invalid(
                            "portable exception source",
                            "source ordinal is absent from the selected artifact map",
                        ));
                    }
                }
                ExceptionFrameSource::Detached(presentation) => {
                    if !self.budget.detached_frames {
                        return Err(invalid(
                            "detached exception source",
                            "Direct program transport does not admit detached diagnostic frames",
                        ));
                    }
                    self.budget.add_text(&presentation.display_name)?;
                    if presentation.display_name.is_empty()
                        || presentation.line.is_some() != presentation.column.is_some()
                        || presentation.line == Some(0)
                        || presentation.column == Some(0)
                        || presentation.line.is_some() != frame.span.is_some()
                    {
                        return Err(invalid(
                            "detached exception source",
                            "diagnostic frame has invalid presentation coordinates",
                        ));
                    }
                }
                ExceptionFrameSource::Unknown => {}
            }
            self.tasks
                .push(Task::ExceptionFrames(value, index + 1, depth));
        } else {
            self.budget.add_elements(runtime.call_stack.len() as u64)?;
            self.tasks.push(Task::ExceptionCallStack(value, 0, depth));
        }
        Ok(())
    }

    pub(super) fn exception_call_stack(
        &mut self,
        value: &'a ExceptionValue,
        index: usize,
        depth: u16,
    ) -> Result<(), ContractError> {
        let runtime = value.runtime.as_ref().expect("metadata task");
        if let Some(frame) = runtime.call_stack.get(index) {
            self.budget.add_text(frame)?;
            self.tasks
                .push(Task::ExceptionCallStack(value, index + 1, depth));
        } else {
            if let Some(phase) = &runtime.phase {
                self.budget.add_text(phase)?;
            }
            self.budget.add_elements(value.causes.len() as u64)?;
            self.tasks
                .push(Task::ExceptionCauses(&value.causes, 0, depth));
        }
        Ok(())
    }

    pub(super) fn exception_cause(
        &mut self,
        values: &'a [ExceptionValue],
        index: usize,
        depth: u16,
    ) -> Result<(), ContractError> {
        if let Some(value) = values.get(index) {
            let nested = depth.checked_add(1).ok_or(ContractError::Limit {
                field: "value depth",
                limit: self.budget.limits.max_depth.into(),
            })?;
            self.budget.node(nested)?;
            self.tasks
                .push(Task::ExceptionCauses(values, index + 1, depth));
            self.budget.add_text(&value.identifier)?;
            self.budget.add_text(&value.message)?;
            self.budget.add_elements(value.stack.len() as u64)?;
            self.tasks.push(Task::ExceptionStack(value, 0, nested));
        }
        Ok(())
    }
}
