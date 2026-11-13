//! An original diagnostic display owner; ordinary source-owned Arc text is
//! retained without copying, while decoder-produced text has a known layout.
use super::{
    diagnostic::{exact_vec, text_end},
    DiagnosticConstructionError, DiagnosticCopyProgress, RetainedDiagnosticAllocation,
};
use runmat_execution::resource::ExactShared;
use std::{fmt, num::NonZeroUsize, ops::Deref, sync::Arc};

#[derive(Debug)]
struct Payload {
    text: String,
    _claim: RetainedDiagnosticAllocation,
}
#[derive(Clone)]
enum Storage {
    Source(Arc<str>),
    Exact(ExactShared<Payload>),
}
#[derive(Clone)]
pub struct DiagnosticText(Storage);
impl DiagnosticText {
    pub fn requested_bytes<E>(bytes: usize) -> Result<usize, DiagnosticConstructionError<E>> {
        ExactShared::<Payload>::requested_bytes()
            .checked_add(bytes)
            .ok_or(DiagnosticConstructionError::LayoutOverflow)
    }
    pub fn try_begin_copy_exact<E>(
        source: &str,
        admit: impl FnOnce(usize) -> Result<RetainedDiagnosticAllocation, E>,
    ) -> Result<DiagnosticTextCopy<'_>, DiagnosticConstructionError<E>> {
        let claim = admit(Self::requested_bytes(source.len())?)
            .map_err(DiagnosticConstructionError::Admission)?;
        let mut payload = Payload {
            text: String::new(),
            _claim: claim,
        };
        // Keep the claim alive through any fallible partial allocation.
        payload.text = unsafe { String::from_utf8_unchecked(exact_vec(source.len())?) };
        let owner = ExactShared::try_new(payload)
            .map_err(|_| DiagnosticConstructionError::AllocationFailed)?;
        Ok(DiagnosticTextCopy { source, owner })
    }
}
impl From<Arc<str>> for DiagnosticText {
    fn from(source: Arc<str>) -> Self {
        Self(Storage::Source(source))
    }
}
impl Deref for DiagnosticText {
    type Target = str;
    fn deref(&self) -> &str {
        match &self.0 {
            Storage::Source(source) => source,
            Storage::Exact(owner) => &owner.get().text,
        }
    }
}
impl fmt::Debug for DiagnosticText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}
impl PartialEq for DiagnosticText {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for DiagnosticText {}

pub struct DiagnosticTextCopy<'a> {
    source: &'a str,
    owner: ExactShared<Payload>,
}
impl DiagnosticTextCopy<'_> {
    pub fn advance(&mut self, work: NonZeroUsize) -> DiagnosticCopyProgress {
        let target = &mut self.owner.get_mut_unique().text;
        let start = target.len();
        let end = text_end(self.source, start, work.get());
        target.push_str(&self.source[start..end]);
        DiagnosticCopyProgress {
            copied_bytes: end - start,
            complete: end == self.source.len(),
        }
    }
    pub fn try_finish(self) -> Result<DiagnosticText, Self> {
        if self.owner.get().text.len() == self.source.len() {
            Ok(DiagnosticText(Storage::Exact(self.owner)))
        } else {
            Err(self)
        }
    }
}
