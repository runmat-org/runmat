use runmat_execution::resource::ExactShared;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use std::alloc::{alloc, Layout};
use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Debug)]
struct WorkspaceNamePayload {
    name: String,
    // ExactShared retires its header first; the String then releases its
    // backing before this claim refunds the complete name allocation.
    _claim: Option<ExactHostAllocationClaim>,
}

/// One allocation-stable workspace identity shared by live entries, session
/// globals, pending deltas, publication origins, and deterministic indexes.
#[derive(Clone, Debug)]
pub struct WorkspaceName(ExactShared<WorkspaceNamePayload>);

impl Borrow<str> for WorkspaceName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl WorkspaceName {
    pub fn requested_bytes(name: &str) -> Result<usize, ExactHostNumericAllocationError> {
        let backing = Layout::array::<u8>(name.len())
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size();
        backing
            .checked_add(ExactShared::<WorkspaceNamePayload>::requested_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub fn prepare(
        name: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let requested = Self::requested_bytes(name)?;
        let claim = permit.reserve_exact_claim(requested)?;
        let name = exact_string(name)?;
        ExactShared::try_new(WorkspaceNamePayload {
            name,
            _claim: Some(claim),
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    /// The existing unbounded workspace policy does not issue a finite claim.
    pub fn unbounded(name: &str) -> Self {
        Self(ExactShared::new(WorkspaceNamePayload {
            name: name.to_owned(),
            _claim: None,
        }))
    }

    pub fn as_str(&self) -> &str {
        &self.0.get().name
    }

    pub(crate) fn has_retained_claim(&self) -> bool {
        self.0.get()._claim.is_some()
    }

    #[cfg(test)]
    pub(crate) fn shares_owner_with(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl Deref for WorkspaceName {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PartialEq for WorkspaceName {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for WorkspaceName {}

impl PartialOrd for WorkspaceName {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkspaceName {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for WorkspaceName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl fmt::Display for WorkspaceName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

fn exact_string(source: &str) -> Result<String, ExactHostNumericAllocationError> {
    if source.is_empty() {
        return Ok(String::new());
    }
    let layout = Layout::array::<u8>(source.len())
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
    // SAFETY: a nonnull allocation has the exact byte layout. The source is
    // valid UTF-8 and all bytes are initialized before String publication.
    let pointer = unsafe { alloc(layout) };
    if pointer.is_null() {
        return Err(ExactHostNumericAllocationError::AllocationFailed);
    }
    unsafe {
        pointer.copy_from_nonoverlapping(source.as_ptr(), source.len());
        Ok(String::from_utf8_unchecked(Vec::from_raw_parts(
            pointer,
            source.len(),
            source.len(),
        )))
    }
}
