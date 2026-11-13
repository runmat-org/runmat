//! Exception causes, presentation text and frame arrays use the same bounded
//! traversal as numeric captures; admitted source ordinals remain untouched.
use super::*;
impl<'a> SelectedGraphJsonCursor<'a> {
    pub(super) fn exception(
        &mut self,
        value: &'a ExceptionValue,
    ) -> Result<(), SelectedGraphJsonError> {
        if let Some(runtime) = &value.runtime {
            self.object([
                ("identifier", Task::Text(&value.identifier)),
                ("message", Task::Text(&value.message)),
                ("stack", Task::Array(Array::Texts(&value.stack))),
                ("causes", Task::Array(Array::Exceptions(&value.causes))),
                ("runtime", Task::Runtime(runtime)),
            ])
        } else {
            self.object([
                ("identifier", Task::Text(&value.identifier)),
                ("message", Task::Text(&value.message)),
                ("stack", Task::Array(Array::Texts(&value.stack))),
                ("causes", Task::Array(Array::Exceptions(&value.causes))),
            ])
        }
    }
    pub(super) fn runtime(
        &mut self,
        value: &'a ExceptionRuntimeMetadata,
    ) -> Result<(), SelectedGraphJsonError> {
        self.object([
            ("span", Task::Atom(&value.span)),
            ("builtin", Task::OptionalText(value.builtin.as_deref())),
            ("task_id", Task::OptionalText(value.task_id.as_deref())),
            (
                "call_frames",
                Task::Array(Array::Frames(&value.call_frames)),
            ),
            ("call_frames_elided", Task::Atom(&value.call_frames_elided)),
            ("call_stack", Task::Array(Array::Texts(&value.call_stack))),
            ("phase", Task::OptionalText(value.phase.as_deref())),
            ("gpu_gather_retry", Task::Atom(&value.gpu_gather_retry)),
            ("failure_origin", Task::Atom(&value.failure_origin)),
        ])
    }
    pub(super) fn call_frame(
        &mut self,
        value: &'a ExceptionCallFrame,
    ) -> Result<(), SelectedGraphJsonError> {
        let source = match &value.source {
            ExceptionFrameSource::Admitted(source) => Task::Atom(source),
            _ => Task::Raw(b"null"),
        };
        if let ExceptionFrameSource::Detached(presentation) = &value.source {
            self.push(Task::Raw(b"}"))?;
            self.object([
                ("display_name", Task::Text(&presentation.display_name)),
                ("line", Task::Atom(&presentation.line)),
                ("column", Task::Atom(&presentation.column)),
            ])?;
            self.push(Task::Raw(b",\"detached_presentation\":"))?;
            self.push(Task::Atom(&value.span))?;
            self.push(Task::Key("span"))?;
            self.push(Task::Raw(b","))?;
            self.push(source)?;
            self.push(Task::Key("source_id"))?;
            self.push(Task::Raw(b","))?;
            self.push(Task::Text(&value.function))?;
            self.push(Task::Key("function"))?;
            self.push(Task::Raw(b"{"))
        } else {
            self.object([
                ("function", Task::Text(&value.function)),
                ("source_id", source),
                ("span", Task::Atom(&value.span)),
            ])
        }
    }
}
