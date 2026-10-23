//! Claimed sparse row support with deterministic minimum-column traversal.

use super::{ClaimedVec, ManagedSparseScalar};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::{SparseWorkError, Work};

pub(super) struct ActiveSupport {
    heap: ClaimedVec<usize>,
    touched: ClaimedVec<usize>,
    seen: ClaimedVec<u8>,
    queued: ClaimedVec<u8>,
}

impl ActiveSupport {
    pub(super) async fn new(order: usize, work: &mut Work<'_>) -> Result<Self, SparseWorkError> {
        Ok(Self {
            heap: ClaimedVec::new(),
            touched: ClaimedVec::new(),
            seen: ClaimedVec::zeroed(order, work).await?,
            queued: ClaimedVec::zeroed(order, work).await?,
        })
    }

    pub(super) async fn activate(
        &mut self,
        column: usize,
        work: &mut Work<'_>,
    ) -> Result<(), SparseWorkError> {
        if self.seen.as_slice()[column] == 0 {
            self.touched.ensure(1, work).await?;
            self.touched.push_reserved(column);
            self.seen.as_mut_slice()[column] = 1;
        }
        self.queue(column, work).await
    }

    async fn queue(&mut self, column: usize, work: &mut Work<'_>) -> Result<(), SparseWorkError> {
        if self.queued.as_slice()[column] != 0 {
            return Ok(());
        }
        self.heap.ensure(1, work).await?;
        self.heap.push_reserved(column);
        self.queued.as_mut_slice()[column] = 1;
        let mut at = self.heap.len() - 1;
        while at > 0 {
            let parent = (at - 1) / 2;
            if self.heap.as_slice()[parent] <= self.heap.as_slice()[at] {
                break;
            }
            self.heap.as_mut_slice().swap(parent, at);
            at = parent;
            work.tick().await?;
        }
        Ok(())
    }

    pub(super) fn peek(&self) -> Option<usize> {
        self.heap.as_slice().first().copied()
    }

    pub(super) async fn pop_min(
        &mut self,
        work: &mut Work<'_>,
    ) -> Result<Option<usize>, SparseWorkError> {
        let Some(last) = self.heap.pop() else {
            return Ok(None);
        };
        if self.heap.len() == 0 {
            self.queued.as_mut_slice()[last] = 0;
            return Ok(Some(last));
        }
        let result = self.heap.as_slice()[0];
        self.heap.as_mut_slice()[0] = last;
        self.queued.as_mut_slice()[result] = 0;
        let mut at = 0;
        loop {
            let left = at * 2 + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.heap.len()
                && self.heap.as_slice()[right] < self.heap.as_slice()[left]
            {
                right
            } else {
                left
            };
            if self.heap.as_slice()[at] <= self.heap.as_slice()[child] {
                break;
            }
            self.heap.as_mut_slice().swap(at, child);
            at = child;
            work.tick().await?;
        }
        Ok(Some(result))
    }

    pub(super) fn queued_columns(&self) -> &[usize] {
        self.heap.as_slice()
    }

    pub(super) async fn rebuild<T: ManagedSparseScalar>(
        &mut self,
        values: &[T],
        work: &mut Work<'_>,
    ) -> Result<(), SparseWorkError> {
        while let Some(column) = self.heap.pop() {
            self.queued.as_mut_slice()[column] = 0;
            work.tick().await?;
        }
        let count = self.touched.len();
        for at in 0..count {
            let column = self.touched.as_slice()[at];
            if values[column] != T::default() {
                self.queue(column, work).await?;
            }
            work.tick().await?;
        }
        Ok(())
    }

    pub(super) fn pop_touched(&mut self) -> Option<usize> {
        let column = self.touched.pop()?;
        self.seen.as_mut_slice()[column] = 0;
        self.queued.as_mut_slice()[column] = 0;
        Some(column)
    }
}
