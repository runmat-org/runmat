use super::RuntimeContext;
use crate::context::host_allocation::HostOutputPermit;
use crate::context::host_effect::ManagedHostEffectStageReservation;
use crate::context::poll_projection::PollProjection;
use runmat_value::ExactHostNumericAllocationError;
use std::rc::Rc;
use std::sync::Arc;

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

#[derive(Clone)]
pub(super) enum InvocationSourceSelection {
    Inherited,
    Selected(Rc<()>),
    Unbound(Rc<()>),
}

impl InvocationSourceSelection {
    pub(super) fn same_lineage(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Inherited, Self::Inherited) => true,
            (Self::Selected(left), Self::Selected(right))
            | (Self::Unbound(left), Self::Unbound(right)) => Rc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl RuntimeContext {
    /// Resolve an ordinal only through the selected issuer and its still-live
    /// immutable source record. A context-free embedding has no program
    /// binding and returns `None`; an admitted binding missing this ordinal is
    /// a typed failure, never permission to inherit the caller's file.
    pub fn admitted_program_source_id(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Result<Option<runmat_types::SourceId>, ProgramSourceSelectionError> {
        let Some(binding) = &self.program_sources else {
            return Ok(None);
        };
        let source = binding
            .source_ref(ordinal)
            .ok_or(ProgramSourceSelectionError::Missing(ordinal))?
            .resolve_in(self)
            .ok_or(ProgramSourceSelectionError::Retired(ordinal))?;
        crate::source_context::source_info_in_context(self, source)
            .ok_or(ProgramSourceSelectionError::Retired(ordinal))?;
        Ok(Some(source))
    }

    /// Select one admitted program source for this invocation. A native
    /// function must not inherit its caller's current file merely because it
    /// executes inside that caller's session. An unbound embedding has no
    /// current file rather than borrowing an unrelated source identity.
    pub fn with_current_program_source(
        mut self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Result<Self, ProgramSourceSelectionError> {
        let current = self
            .admitted_program_source_id(ordinal)?
            .and_then(|source| crate::source_context::source_info_in_context(&self, source));
        self.source_current = Some(PollProjection::legacy(
            crate::source_context::SourceCurrentProjection::new(current),
        ));
        self.source_selection = if self.program_sources.is_some() {
            InvocationSourceSelection::Selected(Rc::new(()))
        } else {
            InvocationSourceSelection::Unbound(Rc::new(()))
        };
        Ok(self)
    }

    fn poll_source_catalog_snapshots(
        &self,
    ) -> (
        Option<crate::source_context::SourceInfo>,
        Rc<std::collections::HashMap<runmat_types::SourceId, crate::source_context::SourceInfo>>,
    ) {
        super::super::scope::with_active_runtime_context(|active| {
            let active = active.filter(|active| active.same_source_projection_owner(self));
            let current = match &self.source_selection {
                InvocationSourceSelection::Selected(_) | InvocationSourceSelection::Unbound(_) => {
                    active
                        .as_ref()
                        .and_then(|active| active.source_current())
                        .or(self.source_current.as_ref())
                        .and_then(|current| current.borrow().snapshot())
                }
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
            (current, snapshot)
        })
    }

    pub(in crate::context) fn capture_poll_source_catalog(&mut self) {
        let (current, snapshot) = self.poll_source_catalog_snapshots();
        self.source_current = Some(PollProjection::legacy(
            crate::source_context::SourceCurrentProjection::new(current),
        ));
        self.source_catalog_view = Some(PollProjection::legacy(
            crate::source_context::SourceCatalogProjection::new(snapshot),
        ));
    }

    pub(in crate::context) fn with_admitted_poll_source_catalog(
        mut self,
        permit: &HostOutputPermit,
        scratch: Arc<ManagedHostEffectStageReservation>,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let (current, snapshot) = self.poll_source_catalog_snapshots();
        let current = PollProjection::prepare(
            crate::source_context::SourceCurrentProjection::new(current),
            permit,
            Arc::clone(&scratch),
        )?;
        let catalog = PollProjection::prepare(
            crate::source_context::SourceCatalogProjection::new(snapshot),
            permit,
            scratch,
        )?;
        self.source_current = Some(current);
        self.source_catalog_view = Some(catalog);
        Ok(self)
    }
}
