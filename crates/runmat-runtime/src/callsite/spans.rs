//! Retained caller-expression spans.
//!
//! Ordinary execution keeps the compiler-produced `Vec` unchanged. Managed
//! poll scopes copy once into an exactly claimed immutable owner; subsequent
//! poll-local projection and feval-prefix views are O(1).

use crate::context::host_allocation::{
    AdmittedCollectionAllocator, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    HostOutputPermit,
};
use crate::context::RuntimeContext;
use allocator_api2::vec::Vec as AllocatedVec;
use runmat_execution::resource::ExactShared;
use runmat_types::Span;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};
use std::alloc::Layout;
use std::fmt;
use std::ops::Deref;

type ManagedSpanVec = AllocatedVec<Span, AdmittedCollectionAllocator>;

pub struct CallsiteSpans {
    owner: SpanOwner,
    start: usize,
}

enum SpanOwner {
    Unbounded(Vec<Span>),
    Managed(ExactShared<ManagedSpans>),
}

struct ManagedSpans {
    spans: ManagedSpanVec,
    _header_claim: ExactHostAllocationClaim,
}

impl CallsiteSpans {
    fn retained_quote_for_len(len: usize) -> Result<(u64, usize), HostAllocationError> {
        if len == 0 {
            return Ok((0, 0));
        }
        let layout = Layout::array::<Span>(len).map_err(|_| HostAllocationError::Overflow)?;
        let backing = AdmittedCollectionAllocator::requested_bytes(layout)
            .map_err(|_| HostAllocationError::Overflow)?;
        let bytes = backing
            .checked_add(ExactShared::<ManagedSpans>::requested_bytes())
            .ok_or(HostAllocationError::Overflow)?;
        Ok((
            u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?,
            AdmittedCollectionAllocator::retained_claim_count() + 1,
        ))
    }

    pub(crate) fn retained_quote(&self) -> Result<(u64, usize), HostAllocationError> {
        if matches!(&self.owner, SpanOwner::Managed(_)) || self.is_empty() {
            return Ok((0, 0));
        }
        Self::retained_quote_for_len(self.len())
    }

    pub fn from_borrowed_in_context(
        spans: &[Span],
        context: &RuntimeContext,
    ) -> Result<Self, HostAllocationError> {
        Self::from_projected_in_context(spans, context, |span| *span)
    }

    pub fn from_projected_in_context<T>(
        source: &[T],
        context: &RuntimeContext,
        project: impl FnMut(&T) -> Span,
    ) -> Result<Self, HostAllocationError> {
        let policy = HostAllocationReceipt::policy(Some(context))?;
        if source.is_empty() {
            return Ok(Vec::new().into());
        }
        if policy == crate::context::host_allocation::HostAllocationPolicy::Unbounded {
            return Ok(source.iter().map(project).collect::<Vec<_>>().into());
        }
        let (bytes, claims) = Self::retained_quote_for_len(source.len())?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, claims)?;
        let receipt =
            HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, retained)?)?
                .ok_or(HostAllocationError::MissingManagedOwner)?;
        let result = receipt
            .output_permit(context)
            .and_then(|permit| Self::managed_from_projected(source, project, permit));
        receipt.checkpoint(context)?;
        result
    }

    pub(crate) fn admit(self, context: &RuntimeContext) -> Result<Self, HostAllocationError> {
        let policy = HostAllocationReceipt::policy(Some(context))?;
        if matches!(&self.owner, SpanOwner::Managed(_)) {
            return Ok(self);
        }
        if policy == crate::context::host_allocation::HostAllocationPolicy::Unbounded {
            return Ok(self);
        }
        if self.is_empty() {
            return Ok(Vec::new().into());
        }
        let (bytes, claims) = self.retained_quote()?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, claims)?;
        let receipt =
            HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, retained)?)?
                .ok_or(HostAllocationError::MissingManagedOwner)?;
        let result = receipt
            .output_permit(context)
            .and_then(|permit| self.admit_with_permit(permit));
        receipt.checkpoint(context)?;
        result
    }

    pub(crate) fn admit_with_permit(
        self,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        let Self { owner, start } = self;
        let source = match owner {
            SpanOwner::Unbounded(source) => source,
            SpanOwner::Managed(owner) => {
                return Ok(Self {
                    owner: SpanOwner::Managed(owner),
                    start,
                });
            }
        };
        if source.len() == start {
            return Ok(Vec::new().into());
        }
        Self::managed_from_slice(&source[start..], permit)
    }

    pub(crate) fn admitted_clone_with_permit(
        &self,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        match &self.owner {
            SpanOwner::Managed(owner) => Ok(Self {
                owner: SpanOwner::Managed(owner.clone()),
                start: self.start,
            }),
            SpanOwner::Unbounded(_) if self.is_empty() => Ok(Self {
                owner: SpanOwner::Unbounded(Vec::new()),
                start: 0,
            }),
            SpanOwner::Unbounded(_) => Self::managed_from_slice(self, permit),
        }
    }

    fn managed_from_slice(
        source: &[Span],
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        Self::managed_from_projected(source, |span| *span, permit)
    }

    fn managed_from_projected<T>(
        source: &[T],
        project: impl FnMut(&T) -> Span,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        if source.is_empty() {
            return Ok(Vec::new().into());
        }
        let mut spans = AllocatedVec::new_in(AdmittedCollectionAllocator::new(permit.clone()));
        spans
            .try_reserve_exact(source.len())
            .map_err(|_| HostAllocationError::Overflow)?;
        spans.extend(source.iter().map(project));
        let header_claim = permit
            .reserve_exact_claim(ExactShared::<ManagedSpans>::requested_bytes())
            .map_err(HostAllocationError::Numeric)?;
        let owner = ExactShared::try_new(ManagedSpans {
            spans,
            _header_claim: header_claim,
        })
        .map_err(|_| {
            HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            )
        })?;
        Ok(Self {
            owner: SpanOwner::Managed(owner),
            start: 0,
        })
    }

    pub fn without_first(mut self) -> Self {
        if !self.is_empty() {
            match &mut self.owner {
                SpanOwner::Unbounded(spans) => {
                    spans.remove(self.start);
                }
                SpanOwner::Managed(_) => self.start += 1,
            }
        }
        self
    }

    fn full_slice(&self) -> &[Span] {
        match &self.owner {
            SpanOwner::Unbounded(spans) => spans,
            SpanOwner::Managed(owner) => &owner.get().spans,
        }
    }
}

impl Clone for CallsiteSpans {
    fn clone(&self) -> Self {
        Self {
            owner: match &self.owner {
                SpanOwner::Unbounded(spans) => SpanOwner::Unbounded(spans.clone()),
                SpanOwner::Managed(owner) => SpanOwner::Managed(owner.clone()),
            },
            start: self.start,
        }
    }
}

impl Deref for CallsiteSpans {
    type Target = [Span];

    fn deref(&self) -> &Self::Target {
        &self.full_slice()[self.start..]
    }
}

impl From<Vec<Span>> for CallsiteSpans {
    fn from(spans: Vec<Span>) -> Self {
        Self {
            owner: SpanOwner::Unbounded(spans),
            start: 0,
        }
    }
}

impl fmt::Debug for CallsiteSpans {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(formatter)
    }
}

impl PartialEq for CallsiteSpans {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl PartialEq<Vec<Span>> for CallsiteSpans {
    fn eq(&self, other: &Vec<Span>) -> bool {
        self.deref() == other.as_slice()
    }
}

impl Eq for CallsiteSpans {}
