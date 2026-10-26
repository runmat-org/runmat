//! Exact numeric duplicate metadata borrows its immutable generation. This
//! optimization is not a purity declaration or equality proof for objects.
use super::super::{invalid, population::Generation, ScalarCooperation, POLICY};
use crate::BuiltinResult;

pub(in super::super) struct Aliases<'a> {
    source: &'a Generation,
    selected: Vec<usize>,
    positions: Vec<usize>,
}
impl<'a> Aliases<'a> {
    pub async fn new(
        source: &'a Generation,
        enabled: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let Some(population) = source.numeric().filter(|_| enabled) else {
            return Ok(None);
        };
        let mut order = POLICY.capacity(population.members, cooperation.runtime())?;
        let mut positions = POLICY.capacity(population.members, cooperation.runtime())?;
        for row in 0..population.members {
            order.push(row);
            positions.push(0);
            cooperation.completed(1).await?;
        }
        use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
        // Stable least-significant-column passes keep each comparator O(1).
        // Whole-row comparison inside a library sort would hide nvars work
        // from its scheduling quantum. Total work remains O(nvars*P*log(P)).
        for column in (0..population.variables).rev() {
            let mut sorting = StableSortWork::new(&mut order, |&a: &usize, &b: &usize| {
                population
                    .get(a, column)
                    .to_bits()
                    .cmp(&population.get(b, column).to_bits())
            });
            loop {
                let (done, work) = match sorting.advance().map_err(|source| {
                    crate::build_runtime_error("ga: duplicate index sort allocation failed")
                        .with_builtin("ga")
                        .with_identifier("RunMat:ga:AllocationFailed")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(source.0)
                        .build()
                })? {
                    SortProgress::Pending(work) => (false, work),
                    SortProgress::Complete(work) => (true, work),
                };
                cooperation.completed(work).await?;
                if done {
                    break;
                }
            }
        }
        let mut representative = None;
        for row in order {
            let mut equal = representative.is_some();
            if let Some(first) = representative {
                for column in 0..population.variables {
                    equal &= population.get(first, column).to_bits()
                        == population.get(row, column).to_bits();
                    cooperation.completed(1).await?;
                    if !equal {
                        break;
                    }
                }
            }
            if !equal {
                representative = Some(row);
            }
            positions[row] = representative.expect("nonempty duplicate group");
            cooperation.completed(1).await?;
        }
        let mut selected = POLICY.capacity(population.members, cooperation.runtime())?;
        // Ties were broken by original row, so each representative precedes
        // every alias. Reuse the same array for dense positions, in place.
        for row in 0..population.members {
            let representative = positions[row];
            positions[row] = if representative == row {
                let position = selected.len();
                selected.push(row);
                position
            } else {
                debug_assert!(representative < row);
                positions[representative]
            };
            cooperation.completed(1).await?;
        }
        Ok(Some(Self {
            source,
            selected,
            positions,
        }))
    }
    pub fn validate(&self, source: &Generation) -> BuiltinResult<()> {
        if std::ptr::eq(self.source, source) {
            Ok(())
        } else {
            Err(invalid(
                "duplicate row metadata belongs to another generation",
            ))
        }
    }
    pub fn selected(&self) -> &[usize] {
        &self.selected
    }
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }
    pub fn representative(&self, row: usize) -> usize {
        self.selected[self.positions[row]]
    }
}
