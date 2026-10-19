use runmat_types::InferenceDiagnostic;

pub(super) struct DiagnosticCollector {
    render: bool,
    rejected: bool,
    pub(super) messages: Vec<InferenceDiagnostic>,
}

impl DiagnosticCollector {
    pub(super) fn new(render: bool) -> Self {
        Self {
            render,
            rejected: false,
            messages: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, make: impl FnOnce() -> InferenceDiagnostic) {
        self.rejected = true;
        if self.render {
            self.messages.push(make());
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        !self.rejected
    }
}
