//! One retained Native dynamic target, including lexical shadow bindings.
//!
//! The live frame never moves out of its service. A separate per-capture
//! issuer retains the lower Base identity and the immutable lexical view, so
//! completion cannot mistake a Native target for a Base-only destination.

use std::alloc::Layout;
use std::rc::Rc;

use runmat_runtime::call::lexical::{
    LexicalBindingRevision, LexicalCompletionChange, LexicalCompletionScope,
    PreparedLexicalCompletionBatch,
};
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::{RuntimeContext, RuntimeWorkspaceService};
use runmat_runtime::workspace::{
    CapturedWorkspaceReadView, DynamicWorkspaceOverlayEntry, DynamicWorkspaceTargetCapture,
    DynamicWorkspaceTargetOverlay, WorkspaceCompletionAction, WorkspaceCompletionIdentity,
    WorkspaceCompletionSource, WorkspaceCompletionValue, WorkspaceSlotReadLease,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim};

use super::{native_workspace_closed, workspace_read_borrow_error, NativeWorkspaceService};

struct NativeCaptureIssuer {
    service: Rc<NativeWorkspaceService>,
    lower: DynamicWorkspaceTargetCapture,
    overlay: DynamicWorkspaceTargetOverlay,
}

// Native owns boxed Runtime errors at its internal boundary. The public
// workspace-service trait still receives the original typed error unchanged.
type CaptureResult<T> = Result<T, Box<RuntimeError>>;

pub(super) fn capture_target(
    service: Rc<NativeWorkspaceService>,
    context: &RuntimeContext,
) -> CaptureResult<Option<DynamicWorkspaceTargetCapture>> {
    let Some((lower, overlay)) = prepare_capture_parts(service.as_ref(), context)? else {
        return Ok(None);
    };
    let issuer = NativeCaptureIssuer {
        service,
        lower: lower.clone(),
        overlay: overlay.clone(),
    };
    let capture = lower.with_admitted_issuer_and_overlay(context, issuer, overlay)?;
    if capture
        .issuer()
        .capture_workspace_completion_identity_with_context(context)?
        != Some(capture.identity())
    {
        return Err(capture_error("dynamic target changed during issuer admission").into());
    }
    Ok(Some(capture))
}

pub(super) fn capture_read_view(
    service: &NativeWorkspaceService,
    context: &RuntimeContext,
) -> CaptureResult<Option<CapturedWorkspaceReadView>> {
    prepare_capture_parts(service, context)?
        .map(|(lower, overlay)| lower.into_read_view_with_overlay(context, overlay))
        .transpose()
        .map_err(Box::new)
}

/// Both consumers retain the same authentic lower generation and lexical
/// revision checks. Only mutation-target capture issues a Native completion
/// service; a terminal read result never needs one.
fn prepare_capture_parts(
    service: &NativeWorkspaceService,
    context: &RuntimeContext,
) -> CaptureResult<Option<(DynamicWorkspaceTargetCapture, DynamicWorkspaceTargetOverlay)>> {
    service
        .ensure_effect_in_operation(Some(context))
        .map_err(|error| *error)?;
    let lower_service = service
        .canonical
        .as_ref()
        .and_then(std::rc::Weak::upgrade)
        .ok_or_else(native_workspace_closed)?;
    let Some(lower) = lower_service.capture_dynamic_workspace_target_with_context(context)? else {
        return Ok(None);
    };
    let frame = service.frame().map_err(RuntimeError::from)?;
    let bindings = frame
        .lexical
        .try_borrow()
        .map_err(workspace_read_borrow_error)?;
    let overlay = if let Some(lexical) = bindings.as_ref() {
        let visible = lexical
            .by_name
            .values()
            .filter(|binding| {
                binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit
            })
            .count();
        let source = lexical
            .by_name
            .iter()
            .filter(|(_, binding)| {
                binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit
            })
            .map(|(name, binding)| {
                let owner = service
                    .local_names
                    .slot(name)
                    .and_then(|slot| service.local_names.name_owner(slot))
                    .ok_or_else(|| capture_error("lexical binding lost its workspace name"))?
                    .clone();
                let revision = LexicalBindingRevision::capture_with_context(
                    &lexical.environment,
                    binding.binding,
                    lexical_scope(binding.kind).map_err(|error| *error)?,
                    context,
                )?;
                let value = runmat_runtime::call::lexical::read_binding_lease_with_context(
                    &lexical.environment,
                    binding.binding,
                    context,
                )?
                .map(WorkspaceSlotReadLease::from);
                Ok(DynamicWorkspaceOverlayEntry::new(owner, value, revision))
            });
        DynamicWorkspaceTargetOverlay::prepare(context, visible, source)?
    } else {
        DynamicWorkspaceTargetOverlay::default()
    };
    drop(bindings);
    overlay.validate_with_context(context)?;
    if lower
        .issuer()
        .capture_workspace_completion_identity_with_context(context)?
        != Some(lower.identity())
    {
        return Err(
            capture_error("lower workspace changed while capturing lexical bindings").into(),
        );
    }
    overlay.validate_with_context(context)?;
    Ok(Some((lower, overlay)))
}

impl NativeCaptureIssuer {
    fn validate(&self, context: &RuntimeContext) -> CaptureResult<WorkspaceCompletionIdentity> {
        self.service
            .ensure_effect_in_operation(Some(context))
            .map_err(|error| *error)?;
        if context.local_identity() != self.lower.context() {
            return Err(capture_error("dynamic target belongs to another runtime context").into());
        }
        let current = self
            .lower
            .issuer()
            .capture_workspace_completion_identity_with_context(context)?;
        if current != Some(self.lower.identity()) {
            return Err(capture_error("lower workspace changed during dynamic execution").into());
        }
        self.overlay.validate_with_context(context)?;
        Ok(self.lower.identity().reissued_by(self))
    }

    fn prepare_completion(
        &self,
        context: &RuntimeContext,
        source: &dyn WorkspaceCompletionSource,
    ) -> CaptureResult<()> {
        let lexical_count = (0..source.len())
            .filter(|index| {
                source
                    .action(*index)
                    .is_some_and(|action| self.overlay.contains_name(action.name()))
            })
            .count();
        let base_count = source
            .len()
            .checked_sub(lexical_count)
            .ok_or_else(|| capture_error("dynamic source classification overflow"))?;
        let mut base_indices = ClaimedVec::<usize>::prepare(context, base_count)?;
        let mut lexical_changes =
            ClaimedVec::<LexicalCompletionChange>::prepare(context, lexical_count)?;
        let frame = self.service.frame().map_err(RuntimeError::from)?;
        let bindings = frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?;
        for index in 0..source.len() {
            let action = source
                .action(index)
                .ok_or_else(|| capture_error("dynamic source changed during preparation"))?;
            let name = action.name();
            if !self.overlay.contains_name(name) {
                base_indices.push(index)?;
                continue;
            }
            let lexical = bindings
                .as_ref()
                .and_then(|owner| owner.by_name.get(name).map(|binding| (owner, binding)))
                .ok_or_else(|| capture_error("lexical source target changed"))?;
            let scope = lexical_scope(lexical.1.kind)?;
            let change = match action {
                WorkspaceCompletionAction::AssignLocal {
                    value: WorkspaceCompletionValue::Retained(source),
                    ..
                } => LexicalCompletionChange::Assign {
                    environment: lexical.0.environment.clone(),
                    binding: lexical.1.binding,
                    scope,
                    source,
                },
                WorkspaceCompletionAction::Remove { .. } => LexicalCompletionChange::Remove {
                    environment: lexical.0.environment.clone(),
                    binding: lexical.1.binding,
                    scope,
                },
                WorkspaceCompletionAction::AssignLocal {
                    value: WorkspaceCompletionValue::Borrowed(_),
                    ..
                } => {
                    return Err(capture_error(
                        "lexical completion needs a retained source value generation",
                    )
                    .into());
                }
                WorkspaceCompletionAction::BindGlobal { .. }
                | WorkspaceCompletionAction::UnbindGlobal { .. } => {
                    return Err(capture_error(
                        "lexical binding cannot change to a session-global namespace",
                    )
                    .into());
                }
            };
            lexical_changes.push(change)?;
        }
        drop(bindings);
        let base_source = IndexedSource {
            source,
            indices: &base_indices.values,
        };
        let base = self
            .lower
            .issuer()
            .prepare_workspace_completion_with_context(
                context,
                self.lower.identity(),
                &base_source,
            )?;
        let mut lexical =
            PreparedLexicalCompletionBatch::prepare(context, lexical_changes.values.drain(..))?;
        self.validate(context)?;
        base.commit_with_lexical(context, &mut lexical)
            .map_err(Box::new)
    }
}

impl RuntimeWorkspaceService for NativeCaptureIssuer {
    fn lookup(&self, name: &str) -> Option<Value> {
        self.service.lookup(name)
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.service.snapshot()
    }

    fn global_names(&self) -> Vec<String> {
        self.service.global_names()
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.service.assign(name, value)
    }

    fn clear(&self) -> Result<(), RuntimeError> {
        self.service.clear()
    }

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        self.service.remove(name)
    }

    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.service.ensure_effect_authority()
    }

    fn ensure_effect_authority_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.validate(context).map(|_| ()).map_err(|error| *error)
    }

    fn capture_workspace_completion_identity_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<Option<WorkspaceCompletionIdentity>, RuntimeError> {
        self.validate(context).map(Some).map_err(|error| *error)
    }

    fn apply_workspace_completion_with_context(
        &self,
        context: &RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<(), RuntimeError> {
        if self.validate(context).map_err(|error| *error)? != identity {
            return Err(capture_error("dynamic target completion identity changed"));
        }
        self.prepare_completion(context, source)
            .map_err(|error| *error)
    }
}

struct IndexedSource<'a> {
    source: &'a dyn WorkspaceCompletionSource,
    indices: &'a [usize],
}

impl WorkspaceCompletionSource for IndexedSource<'_> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn action(&self, index: usize) -> Option<WorkspaceCompletionAction<'_>> {
        self.source.action(*self.indices.get(index)?)
    }
}

struct ClaimedVec<T> {
    values: Vec<T>,
    limit: usize,
    // The Vec's elements and backing retire before the exact funding claim.
    _claim: Option<ExactHostAllocationClaim>,
}

impl<T> ClaimedVec<T> {
    fn prepare(context: &RuntimeContext, capacity: usize) -> CaptureResult<Self> {
        if capacity == 0 {
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| allocation_error(context, cause))?;
            return Ok(Self {
                values: Vec::new(),
                limit: 0,
                _claim: None,
            });
        }
        let bytes = Layout::array::<T>(capacity)
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?
            .size();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes)
                .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?,
            1,
        )
        .map_err(|cause| allocation_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| allocation_error(context, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| allocation_error(context, cause))?;
        match receipt {
            Some(receipt) => receipt
                .with_output_allocation(context, |permit| {
                    let claim = permit.reserve_exact_claim(bytes)?;
                    let values = exact_capacity_vec(capacity)?;
                    Ok(Self {
                        values,
                        limit: capacity,
                        _claim: Some(claim),
                    })
                })
                .map_err(|cause| allocation_error(context, cause)),
            None => Ok(Self {
                values: exact_capacity_vec(capacity).map_err(|cause| {
                    allocation_error(context, HostAllocationError::Numeric(cause))
                })?,
                limit: capacity,
                _claim: None,
            }),
        }
        .map_err(Box::new)
    }

    fn push(&mut self, value: T) -> CaptureResult<()> {
        if self.values.len() == self.limit {
            return Err(capture_error("dynamic source changed during preparation").into());
        }
        self.values.push(value);
        Ok(())
    }
}

fn allocation_error(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    crate::error::allocation_error(
        Some(context),
        format!("native dynamic target metadata admission failed: {cause}"),
        cause,
    )
}

fn capture_error(message: &'static str) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error("RunMat:Workspace:CaptureConflict", message)
}

fn lexical_scope(
    kind: runmat_native_codegen::NativeLexicalBindingKind,
) -> CaptureResult<LexicalCompletionScope> {
    match kind {
        runmat_native_codegen::NativeLexicalBindingKind::Owned => Ok(LexicalCompletionScope::Owned),
        runmat_native_codegen::NativeLexicalBindingKind::Inherited => {
            Ok(LexicalCompletionScope::Inherited)
        }
        runmat_native_codegen::NativeLexicalBindingKind::Transit => {
            Err(capture_error("transit binding became a lexical target").into())
        }
    }
}
