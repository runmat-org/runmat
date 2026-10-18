//! Immutable mutation-target capture for dynamic workspace execution.
//!
//! The capture retains the exact service which issued its completion identity,
//! preventing pointer-identity reuse while a child is suspended. Values,
//! names, session-global targets and GC roots remain owned by the retained
//! logical generation; no detached workspace mirror is constructed here.

use std::rc::Rc;

use crate::context::{RuntimeContext, RuntimeContextLocalId, RuntimeWorkspaceService};
use crate::RuntimeError;

use super::{
    GlobalBindingTarget, LogicalWorkspaceGeneration, WorkspaceCompletionIdentity,
    WorkspaceSlotReadLease,
};
use runmat_value::Value;

mod overlay;
mod issuer;
mod read_view;
use issuer::DynamicCaptureIssuer;
pub use overlay::{DynamicWorkspaceOverlayEntry, DynamicWorkspaceTargetOverlay};
pub use read_view::CapturedWorkspaceReadView;

/// One immutable entry from an exact dynamic-workspace mutation target.
/// The value lease and optional global destination belong to the same
/// captured generation; neither is resolved again from live mutable state.
#[derive(Clone, Debug)]
pub struct DynamicWorkspaceTargetEntry<'a> {
    name: &'a str,
    value: WorkspaceSlotReadLease,
    global_target: Option<GlobalBindingTarget>,
}

impl<'a> DynamicWorkspaceTargetEntry<'a> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn value(&self) -> &Value {
        self.value.value()
    }

    pub fn value_lease(&self) -> &WorkspaceSlotReadLease {
        &self.value
    }

    pub fn into_value_lease(self) -> WorkspaceSlotReadLease {
        self.value
    }

    pub fn global_target(&self) -> Option<&GlobalBindingTarget> {
        self.global_target.as_ref()
    }
}

#[derive(Clone)]
pub struct DynamicWorkspaceTargetCapture {
    context: RuntimeContextLocalId,
    identity: WorkspaceCompletionIdentity,
    view: CapturedWorkspaceReadView,
    issuer: DynamicCaptureIssuer,
}

impl std::fmt::Debug for DynamicWorkspaceTargetCapture {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicWorkspaceTargetCapture")
            .field("context", &self.context)
            .field("identity", &self.identity)
            .field("view", &self.view)
            .finish_non_exhaustive()
    }
}

impl DynamicWorkspaceTargetCapture {
    pub(crate) fn issued(
        context: RuntimeContextLocalId,
        identity: WorkspaceCompletionIdentity,
        generation: LogicalWorkspaceGeneration,
        issuer: Rc<dyn RuntimeWorkspaceService>,
    ) -> Self {
        Self {
            context,
            identity,
            view: CapturedWorkspaceReadView::new(
                generation,
                DynamicWorkspaceTargetOverlay::default(),
            ),
            issuer: DynamicCaptureIssuer::existing(issuer),
        }
    }

    /// Adapt an exact lower workspace capture to a retained composite
    /// service. The new issuer is a distinct authority: it must itself retain
    /// and validate the lower identity and every lexical activation revision
    /// before completing the child. This constructor never changes or forges
    /// the lower Base identity.
    pub fn with_admitted_issuer_and_overlay<T>(
        self,
        context: &RuntimeContext,
        issuer: T,
        overlay: DynamicWorkspaceTargetOverlay,
    ) -> Result<Self, RuntimeError>
    where
        T: RuntimeWorkspaceService + 'static,
    {
        if self.context != context.local_identity() {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:MutationContextMismatch",
                "dynamic target belongs to a different runtime context",
            ));
        }
        let issuer = DynamicCaptureIssuer::prepare(context, issuer)?;
        let identity = self.identity.reissued_by(issuer.service());
        Ok(Self {
            context: self.context,
            identity,
            view: self.view.with_overlay(overlay),
            issuer,
        })
    }

    pub fn context(&self) -> RuntimeContextLocalId {
        self.context
    }

    pub fn identity(&self) -> WorkspaceCompletionIdentity {
        self.identity
    }

    /// Retain only the immutable value and namespace view after execution.
    /// A terminal result cannot issue another completion, so its context,
    /// identity, and mutation service are released here.
    pub fn into_read_view(self) -> CapturedWorkspaceReadView {
        self.view
    }

    /// Retain a context-checked lexical overlay without creating a mutation
    /// issuer. Terminal consumers need only this immutable view.
    pub fn into_read_view_with_overlay(
        self,
        context: &RuntimeContext,
        overlay: DynamicWorkspaceTargetOverlay,
    ) -> Result<CapturedWorkspaceReadView, RuntimeError> {
        if self.context != context.local_identity() {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:MutationContextMismatch",
                "dynamic target belongs to a different runtime context",
            ));
        }
        overlay.validate_with_context(context)?;
        Ok(self.view.with_overlay(overlay))
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.view.get(name)
    }

    pub fn read_lease(&self, name: &str) -> Option<WorkspaceSlotReadLease> {
        self.view.read_lease(name)
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        self.view.global_target(name)
    }

    /// Exact captured namespace targets, including globals whose session
    /// value is absent. A lexical overlay shadows the Base target without
    /// manufacturing a second target map.
    pub fn global_bindings(&self) -> impl Iterator<Item = (&str, &GlobalBindingTarget)> {
        self.view.global_bindings()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.view.iter()
    }

    /// Iterate retained value owners and exact namespace destinations from
    /// the one captured generation. This is the producer boundary for an
    /// admitted child frame: consumers must retain these leases rather than
    /// rebuilding a detached `Vec<Value>` snapshot.
    pub fn iter_leases(&self) -> impl Iterator<Item = DynamicWorkspaceTargetEntry<'_>> {
        self.iter().map(|(name, _)| {
            let value = self
                .read_lease(name)
                .expect("captured entry has a retained value owner");
            DynamicWorkspaceTargetEntry {
                name,
                value,
                global_target: self.global_target(name).cloned(),
            }
        })
    }

    pub fn issuer(&self) -> &dyn RuntimeWorkspaceService {
        self.issuer.service()
    }
}
