//! Sparse transpose metadata over the one retained repair model. Entries own
//! only (original column, column-local ordinal), never copied coefficients.
use super::super::super::intlinprog::feasibility::OriginalLinear;
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;

pub(super) struct Rows {
    offsets: Vec<usize>,
    entries: Vec<(usize, usize)>,
    variables: usize,
    rows: usize,
    widest: usize,
}
impl Rows {
    pub async fn new(
        view: OriginalLinear<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if view.rows() == 0 {
            return Ok(Self {
                offsets: Vec::new(),
                entries: Vec::new(),
                variables: view.variables(),
                rows: 0,
                widest: 0,
            });
        }
        let mut counts = POLICY.capacity(view.rows(), cooperation.runtime())?;
        for _ in 0..view.rows() {
            counts.push(0usize);
            cooperation.completed(1).await?;
        }
        let mut total = 0usize;
        for column in 0..view.variables() {
            for (row, _) in view
                .column(column)
                .ok_or_else(|| invalid("original linear column is unavailable"))?
            {
                counts[row] = counts[row]
                    .checked_add(1)
                    .ok_or_else(|| invalid("sparse row extent overflow"))?;
                total = total
                    .checked_add(1)
                    .ok_or_else(|| invalid("original sparse extent overflow"))?;
                cooperation.completed(1).await?;
            }
        }
        let mut offsets = POLICY.capacity(
            view.rows()
                .checked_add(1)
                .ok_or_else(|| invalid("sparse row pointer extent overflow"))?,
            cooperation.runtime(),
        )?;
        offsets.push(0);
        let mut widest = 0usize;
        let mut offset = 0usize;
        for count in &mut counts {
            widest = widest.max(*count);
            let next = offset
                .checked_add(*count)
                .ok_or_else(|| invalid("sparse row pointer overflow"))?;
            *count = offset;
            offsets.push(next);
            offset = next;
            cooperation.completed(1).await?;
        }
        let mut entries = POLICY.capacity(total, cooperation.runtime())?;
        for _ in 0..total {
            entries.push((0, 0));
            cooperation.completed(1).await?;
        }
        // Ascending column traversal preserves the sorted sparse row order
        // required by the exact cone owner without another sorting workspace.
        for column in 0..view.variables() {
            for (ordinal, (row, _)) in view
                .column(column)
                .ok_or_else(|| invalid("original linear column is unavailable"))?
                .enumerate()
            {
                entries[counts[row]] = (column, ordinal);
                counts[row] += 1;
                cooperation.completed(1).await?;
            }
        }
        Ok(Self {
            offsets,
            entries,
            variables: view.variables(),
            rows: view.rows(),
            widest,
        })
    }
    pub fn scratch(&self, cooperation: &ScalarCooperation) -> BuiltinResult<Vec<(usize, f64)>> {
        POLICY.capacity(self.widest, cooperation.runtime())
    }
    pub async fn fill(
        &self,
        view: OriginalLinear<'_>,
        row: usize,
        scratch: &mut Vec<(usize, f64)>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if view.variables() != self.variables || view.rows() != self.rows || row >= self.rows {
            return Err(invalid(
                "original sparse row metadata differs from its retained model",
            ));
        }
        let entries = &self.entries[self.offsets[row]..self.offsets[row + 1]];
        if scratch.capacity() < entries.len() {
            return Err(invalid("original sparse row scratch was not admitted"));
        }
        scratch.clear();
        for &(column, ordinal) in entries {
            let (original_row, value) = view
                .entry(column, ordinal)
                .ok_or_else(|| invalid("original sparse entry is unavailable"))?;
            if original_row != row {
                return Err(invalid(
                    "original sparse entry generation differs from row metadata",
                ));
            }
            scratch.push((column, value));
            cooperation.completed(1).await?;
        }
        Ok(())
    }
}
