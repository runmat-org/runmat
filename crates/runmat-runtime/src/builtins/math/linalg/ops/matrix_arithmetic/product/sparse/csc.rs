//! Stored-column multiplication with sparse touched-row accumulation.
use super::*;
use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
use std::collections::HashMap;
pub(super) async fn multiply<T: Element>(
    a: Operand<'_>,
    b: Operand<'_>,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<Value> {
    let mut work = runtime
        .map(OperatorWork::new)
        .transpose()
        .map_err(|e| operator_error(e, runtime))?;
    let mut columns =
        runmat_value::exact_capacity_vec(b.cols().checked_add(1).ok_or_else(|| {
            failure(
                crate::context::host_allocation::HostAllocationError::Overflow,
                runtime,
            )
        })?)
        .map_err(|e| failure(e, runtime))?;
    let (mut rows, mut values) = (Vec::new(), Vec::new());
    let mut accumulated = HashMap::<usize, T>::new();
    let mut touched = Vec::new();
    columns.push(0);
    for column in 0..b.cols() {
        for (inner, coefficient) in b.column(column) {
            let coefficient = T::convert(coefficient).expect("selected compatible class");
            for (row, value) in a.column(inner) {
                if !accumulated.contains_key(&row) {
                    accumulated
                        .try_reserve(1)
                        .map_err(|e| failure(e, runtime))?;
                    touched.try_reserve(1).map_err(|e| failure(e, runtime))?;
                    accumulated.insert(row, T::zero());
                    touched.push(row);
                }
                T::add_product(
                    accumulated.get_mut(&row).expect("touched row"),
                    T::convert(value).expect("selected compatible class"),
                    coefficient,
                );
                tick(&mut work, runtime).await?;
            }
            tick(&mut work, runtime).await?;
        }
        let mut sort = StableSortWork::new(&mut touched, usize::cmp);
        loop {
            let progress = sort.advance().map_err(|e| failure(e.0, runtime))?;
            let (done, count) = match progress {
                SortProgress::Pending(count) => (false, count),
                SortProgress::Complete(count) => (true, count),
            };
            for _ in 0..count {
                tick(&mut work, runtime).await?;
            }
            if done {
                break;
            }
        }
        drop(sort);
        for row in touched.drain(..) {
            let value = accumulated.remove(&row).expect("touched accumulator");
            if !T::is_zero(value) {
                rows.try_reserve(1).map_err(|e| failure(e, runtime))?;
                values.try_reserve(1).map_err(|e| failure(e, runtime))?;
                rows.push(row);
                values.push(value);
            }
            tick(&mut work, runtime).await?;
        }
        columns.push(values.len());
        tick(&mut work, runtime).await?;
    }
    if let Some(work) = &mut work {
        work.flush().await.map_err(|e| operator_error(e, runtime))?;
    }
    let mut validation = runmat_value::SparseCscValidation::new(
        a.rows(),
        b.cols(),
        columns.into(),
        rows.into(),
        values.len(),
    )
    .map_err(|e| crate::build_runtime_error(e).with_builtin(NAME).build())?;
    let unit = std::num::NonZeroUsize::new(1).expect("one CSC validation item");
    loop {
        let done = validation
            .advance(unit)
            .map_err(|e| crate::build_runtime_error(e).with_builtin(NAME).build())?;
        tick(&mut work, runtime).await?;
        if done {
            break;
        }
    }
    if let Some(work) = &mut work {
        work.flush().await.map_err(|e| operator_error(e, runtime))?;
    }
    let parts = validation.into_validated().map_err(|_| {
        crate::build_runtime_error("mtimes: incomplete output topology")
            .with_builtin(NAME)
            .build()
    })?;
    T::sparse(values, parts).map_err(|e| crate::build_runtime_error(e).with_builtin(NAME).build())
}
