//! Immutable, admitted lexical overrides for one dynamic target capture.
//!
//! The source names and value leases already own their physical backings.
//! Only this sorted pointer array and its shared header are new allocations.

use std::alloc::Layout;

use runmat_execution::resource::ExactShared;
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit};

use crate::call::lexical::LexicalBindingRevision;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;

use crate::workspace::{WorkspaceName, WorkspaceSlotReadLease};

#[derive(Clone, Debug)]
pub struct DynamicWorkspaceOverlayEntry {
    name: WorkspaceName,
    value: Option<WorkspaceSlotReadLease>,
    revision: LexicalBindingRevision,
}

impl DynamicWorkspaceOverlayEntry {
    /// `None` is a lexical tombstone: the binding exists but has no current
    /// value, so a lower Base value with the same spelling is hidden.
    pub fn new(
        name: WorkspaceName,
        value: Option<WorkspaceSlotReadLease>,
        revision: LexicalBindingRevision,
    ) -> Self {
        Self {
            name,
            value,
            revision,
        }
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn value(&self) -> Option<&WorkspaceSlotReadLease> {
        self.value.as_ref()
    }

    pub fn validate_with_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        self.revision.validate_with_context(context)
    }
}

#[derive(Debug)]
struct OverlayPayload {
    entries: Vec<DynamicWorkspaceOverlayEntry>,
    // Both the Vec backing and ExactShared header retire before this refund.
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug, Default)]
pub struct DynamicWorkspaceTargetOverlay(Option<ExactShared<OverlayPayload>>);

impl DynamicWorkspaceTargetOverlay {
    pub fn prepare(
        context: &RuntimeContext,
        len: usize,
        mut source: impl Iterator<Item = Result<DynamicWorkspaceOverlayEntry, RuntimeError>>,
    ) -> Result<Self, RuntimeError> {
        if len == 0 {
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| admission_error(context, cause))?;
            if source.next().is_some() {
                return Err(crate::runtime_error::semantic_error(
                    "RunMat:Workspace:LexicalCaptureChanged",
                    "dynamic target lexical entry count changed during capture",
                ));
            }
            return Ok(Self(None));
        }
        let bytes = Layout::array::<DynamicWorkspaceOverlayEntry>(len)
            .map_err(|_| admission_error(context, HostAllocationError::Overflow))?
            .size()
            .checked_add(ExactShared::<OverlayPayload>::requested_bytes())
            .ok_or_else(|| admission_error(context, HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes)
                .map_err(|_| admission_error(context, HostAllocationError::Overflow))?,
            1,
        )
        .map_err(|cause| admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| admission_error(context, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| admission_error(context, cause))?;
        let build = |permit: Option<&dyn ExactHostAllocationPermit>| {
            let claim = permit
                .map(|permit| permit.reserve_exact_claim(bytes))
                .transpose()
                .map_err(|cause| admission_error(context, HostAllocationError::Numeric(cause)))?;
            let mut entries = exact_capacity_vec(len)
                .map_err(|cause| admission_error(context, HostAllocationError::Numeric(cause)))?;
            for entry in source {
                if entries.len() == len {
                    return Err(crate::runtime_error::semantic_error(
                        "RunMat:Workspace:LexicalCaptureChanged",
                        "dynamic target lexical entry count changed during capture",
                    ));
                }
                entries.push(entry?);
            }
            if entries.len() != len {
                return Err(crate::runtime_error::semantic_error(
                    "RunMat:Workspace:LexicalCaptureChanged",
                    "dynamic target lexical entry count changed during capture",
                ));
            }
            entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
            if entries.windows(2).any(|pair| pair[0].name == pair[1].name) {
                return Err(crate::runtime_error::semantic_error(
                    "RunMat:Workspace:DuplicateLexicalName",
                    "dynamic target contains duplicate lexical names",
                ));
            }
            let owner = ExactShared::try_new(OverlayPayload {
                entries,
                _claim: claim,
            })
            .map_err(|_| {
                admission_error(
                    context,
                    HostAllocationError::Numeric(
                        runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                    ),
                )
            })?;
            Ok(Self(Some(owner)))
        };
        match &receipt {
            Some(receipt) => {
                let permit = receipt
                    .output_permit(context)
                    .map_err(|cause| admission_error(context, cause))?;
                let overlay = build(Some(&permit))?;
                receipt
                    .checkpoint(context)
                    .map_err(|cause| admission_error(context, cause))?;
                Ok(overlay)
            }
            None => build(None),
        }
    }

    pub(super) fn get(&self, name: &str) -> Option<&DynamicWorkspaceOverlayEntry> {
        let entries = &self.0.as_ref()?.get().entries;
        entries
            .binary_search_by(|entry| entry.name().cmp(name))
            .ok()
            .map(|index| &entries[index])
    }

    pub fn contains_name(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = &DynamicWorkspaceOverlayEntry> {
        self.0.iter().flat_map(|owner| owner.get().entries.iter())
    }

    pub fn validate_with_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        for entry in self.iter() {
            entry.validate_with_context(context)?;
        }
        Ok(())
    }
}

fn admission_error(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    crate::build_runtime_error(format!(
        "dynamic workspace lexical overlay admission failed: {cause}"
    ))
    .with_identifier(format!(
        "{}:{}",
        context.error_namespace(),
        cause.runtime_identifier_suffix()
    ))
    .with_source(cause)
    .build()
}
