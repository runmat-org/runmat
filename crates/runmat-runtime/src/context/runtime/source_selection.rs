use super::RuntimeContext;
use std::cell::RefCell;
use std::rc::Rc;

/// A selected invocation cannot execute a source ordinal absent from its
/// admitted program binding, or one whose issued lease is no longer live.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramSourceSelectionError {
    Missing(runmat_types::ProgramSourceId),
    Retired(runmat_types::ProgramSourceId),
}

impl std::fmt::Display for ProgramSourceSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(source) => write!(formatter, "program source {} is absent", source.0),
            Self::Retired(source) => write!(formatter, "program source {} is retired", source.0),
        }
    }
}

impl std::error::Error for ProgramSourceSelectionError {}

#[derive(Clone, Copy)]
pub(super) enum InvocationSourceSelection {
    Inherited,
    Selected,
    Unbound,
}

impl RuntimeContext {
    /// Select one admitted program source for this invocation. A native
    /// function must not inherit its caller's current file merely because it
    /// executes inside that caller's session. An unbound embedding has no
    /// current file rather than borrowing an unrelated source identity.
    pub fn with_current_program_source(
        mut self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Result<Self, ProgramSourceSelectionError> {
        let current = if let Some(binding) = &self.program_sources {
            let source = binding
                .source_ref(ordinal)
                .ok_or(ProgramSourceSelectionError::Missing(ordinal))?
                .resolve_in(&self)
                .ok_or(ProgramSourceSelectionError::Retired(ordinal))?;
            Some(
                crate::source_context::source_info_in_context(&self, source)
                    .ok_or(ProgramSourceSelectionError::Retired(ordinal))?,
            )
        } else {
            None
        };
        self.source_current = Some(Rc::new(RefCell::new(
            crate::source_context::SourceCurrentProjection::new(current),
        )));
        self.source_selection = if self.program_sources.is_some() {
            InvocationSourceSelection::Selected
        } else {
            InvocationSourceSelection::Unbound
        };
        Ok(self)
    }

    pub(super) fn with_poll_source_catalog(mut self) -> Self {
        let active = super::super::scope::active_runtime_context()
            .filter(|active| active.same_source_projection_owner(&self));
        let current = match self.source_selection {
            InvocationSourceSelection::Selected | InvocationSourceSelection::Unbound => self
                .source_current
                .as_ref()
                .and_then(|current| current.borrow().snapshot()),
            InvocationSourceSelection::Inherited => active
                .as_ref()
                .and_then(|active| {
                    active
                        .source_current()
                        .map(|current| current.borrow().snapshot())
                })
                .or_else(|| {
                    self.source_current
                        .as_ref()
                        .map(|current| current.borrow().snapshot())
                })
                .unwrap_or_else(|| self.state.source.borrow().current.clone()),
        };
        self.source_current = Some(Rc::new(RefCell::new(
            crate::source_context::SourceCurrentProjection::new(current),
        )));
        let snapshot = active
            .as_ref()
            .and_then(|active| {
                active
                    .source_catalog_view()
                    .map(|view| view.borrow().snapshot_catalog())
            })
            .or_else(|| {
                self.source_catalog_view
                    .as_ref()
                    .map(|view| view.borrow().snapshot_catalog())
            })
            .unwrap_or_else(|| self.state.source.borrow().catalog.clone());
        self.source_catalog_view = Some(Rc::new(RefCell::new(
            crate::source_context::SourceCatalogProjection::new(snapshot),
        )));
        self
    }
}
