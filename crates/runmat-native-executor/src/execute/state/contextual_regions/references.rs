//! Exactly owned replay references for one completed expression or embedded call.
//! Scalar results remain inline. A variable-width result is admitted from its
//! actual width before any references are inserted into the arena.

use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::native::NativeValueRef;
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim, ExactHostNumericAllocationError};
use std::alloc::Layout;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::continuation_admission_error;

#[derive(Debug)]
pub(crate) struct ReferencePayload {
    values: Vec<NativeValueRef>,
    // ExactShared retires its header, then the Vec's exact backing, before
    // this one claim refunds their combined physical charge.
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub(crate) enum CachedRefs {
    Empty,
    One(NativeValueRef),
    Many(ExactShared<ReferencePayload>),
}

enum BuilderStorage {
    Empty,
    One(Option<NativeValueRef>),
    Many(ExactShared<ReferencePayload>),
}

pub(super) struct CachedRefsBuilder {
    width: usize,
    storage: BuilderStorage,
}

impl CachedRefsBuilder {
    pub(super) fn prepare(runtime: &RuntimeContext, width: usize) -> NativeExecutorResult<Self> {
        match width {
            0 => {
                return Ok(Self {
                    width,
                    storage: BuilderStorage::Empty,
                })
            }
            1 => {
                return Ok(Self {
                    width,
                    storage: BuilderStorage::One(None),
                })
            }
            _ => {}
        }
        let backing = Layout::array::<NativeValueRef>(width)
            .map_err(|_| NativeExecutorError::Host("native replay width overflow".into()))?
            .size();
        let physical = backing
            .checked_add(ExactShared::<ReferencePayload>::requested_bytes())
            .ok_or_else(|| NativeExecutorError::Host("native replay layout overflow".into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(physical)
                .map_err(|_| NativeExecutorError::Host("native replay layout overflow".into()))?,
            1,
        )
        .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let construct = |claim| {
            let values = exact_capacity_vec(width)?;
            ExactShared::try_new(ReferencePayload {
                values,
                _claim: claim,
            })
            .map(BuilderStorage::Many)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
        };
        let storage = match receipt {
            Some(receipt) => receipt
                .with_output_allocation(runtime, |permit| {
                    let claim = permit.reserve_exact_claim(physical)?;
                    construct(Some(claim))
                })
                .map_err(|cause| continuation_admission_error(runtime, cause)),
            None => construct(None).map_err(|cause| {
                continuation_admission_error(runtime, HostAllocationError::Numeric(cause))
            }),
        }?;
        Ok(Self { width, storage })
    }

    pub(super) fn push_initial(&mut self, value: NativeValueRef) -> NativeExecutorResult<()> {
        match &mut self.storage {
            BuilderStorage::Empty => Err(NativeExecutorError::Host(
                "empty native replay received a reference".into(),
            )),
            BuilderStorage::One(slot) => {
                if slot.is_some() {
                    return Err(NativeExecutorError::Host(
                        "native scalar replay received multiple references".into(),
                    ));
                }
                *slot = Some(value);
                Ok(())
            }
            BuilderStorage::Many(shared) => {
                let values = &mut shared.get_mut_unique().values;
                if values.len() == values.capacity() {
                    return Err(NativeExecutorError::Host(
                        "admitted native replay width was exceeded".into(),
                    ));
                }
                values.push(value);
                Ok(())
            }
        }
    }

    pub(super) fn as_slice(&self) -> &[NativeValueRef] {
        match &self.storage {
            BuilderStorage::Empty | BuilderStorage::One(None) => &[],
            BuilderStorage::One(Some(value)) => std::slice::from_ref(value),
            BuilderStorage::Many(shared) => &shared.get().values,
        }
    }

    pub(super) fn finish(&mut self) -> NativeExecutorResult<CachedRefs> {
        if self.as_slice().len() != self.width {
            return Err(NativeExecutorError::Host(
                "native replay width does not match its admitted owner".into(),
            ));
        }
        match std::mem::replace(&mut self.storage, BuilderStorage::Empty) {
            BuilderStorage::Empty => Ok(CachedRefs::Empty),
            BuilderStorage::One(Some(value)) => Ok(CachedRefs::One(value)),
            BuilderStorage::Many(shared) => Ok(CachedRefs::Many(shared)),
            BuilderStorage::One(None) => Err(NativeExecutorError::Host(
                "native scalar replay has no reference".into(),
            )),
        }
    }
}

impl CachedRefs {
    pub(crate) fn as_slice(&self) -> &[NativeValueRef] {
        match self {
            Self::Empty => &[],
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => &values.get().values,
        }
    }
}
