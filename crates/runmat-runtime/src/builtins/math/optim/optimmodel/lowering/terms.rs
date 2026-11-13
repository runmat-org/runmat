//! Compact active original-coordinate columns. Constants have no global CSC
//! pointer allocation; global columns are expanded only at final assembly.
use super::*;
use clarabel::algebra::CscMatrix;
#[derive(Clone, Copy)]
pub(super) struct Term {
    pub column: usize,
    pub row: usize,
    pub value: f64,
}
pub(super) struct Table {
    columns: Vec<usize>,
    matrix: CscMatrix<f64>,
}
#[derive(Clone)]
pub(super) enum Coefficients {
    Zero,
    Sparse(std::rc::Rc<Table>),
}
impl Coefficients {
    pub(super) async fn from_entries(
        mut entries: Vec<Term>,
        rows: usize,
        original_columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        if entries.is_empty() {
            return Ok(Ok(Self::Zero));
        }
        let mut ordered = true;
        let mut previous = None;
        for entry in &entries {
            if entry.column >= original_columns || entry.row >= rows {
                return Err(
                    policy.invalid("compiled coefficient coordinate exceeds its original layout")
                );
            }
            if !entry.value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            let key = (entry.column, entry.row);
            if previous.is_some_and(|old| old > key) {
                ordered = false
            }
            previous = Some(key);
            cooperation.completed(1).await?;
        }
        if !ordered {
            use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
            let mut sort = StableSortWork::new(&mut entries, |left, right| {
                (left.column, left.row).cmp(&(right.column, right.row))
            });
            loop {
                match sort.advance().map_err(|source| policy.reserve(source.0))? {
                    SortProgress::Pending(work) => cooperation.completed(work).await?,
                    SortProgress::Complete(work) => {
                        cooperation.completed(work).await?;
                        break;
                    }
                }
            }
        }
        let maximum_columns = entries.len().min(original_columns);
        let mut columns =
            runmat_value::exact_capacity_vec(maximum_columns).map_err(|e| policy.storage(e))?;
        let mut pointers = runmat_value::exact_capacity_vec(
            maximum_columns
                .checked_add(1)
                .ok_or_else(|| policy.invalid("compiled column pointer count overflows usize"))?,
        )
        .map_err(|e| policy.storage(e))?;
        let mut indices =
            runmat_value::exact_capacity_vec(entries.len()).map_err(|e| policy.storage(e))?;
        let mut values =
            runmat_value::exact_capacity_vec(entries.len()).map_err(|e| policy.storage(e))?;
        pointers.push(0);
        let mut cursor = 0;
        while cursor < entries.len() {
            let column = entries[cursor].column;
            let first = values.len();
            while cursor < entries.len() && entries[cursor].column == column {
                let row = entries[cursor].row;
                let mut sum = 0.0_f64;
                while cursor < entries.len()
                    && entries[cursor].column == column
                    && entries[cursor].row == row
                {
                    sum += entries[cursor].value;
                    if !sum.is_finite() {
                        return Ok(Err(Stop::NonFinite));
                    }
                    cursor += 1;
                    cooperation.completed(1).await?;
                }
                if sum != 0.0 {
                    indices.push(row);
                    values.push(sum);
                }
            }
            if values.len() != first {
                columns.push(column);
                pointers.push(values.len());
            }
        }
        if columns.is_empty() {
            return Ok(Ok(Self::Zero));
        }
        let matrix = CscMatrix::new(rows, columns.len(), pointers, indices, values);
        Ok(Ok(Self::Sparse(std::rc::Rc::new(Table {
            columns,
            matrix,
        }))))
    }
    pub(super) fn is_zero(&self) -> bool {
        matches!(self, Self::Zero)
    }
    pub(super) fn nonzeros(&self) -> usize {
        match self {
            Self::Zero => 0,
            Self::Sparse(table) => table.matrix.nzval.len(),
        }
    }
    pub(super) fn columns(&self) -> &[usize] {
        match self {
            Self::Zero => &[],
            Self::Sparse(table) => &table.columns,
        }
    }
    pub(super) fn column(&self, column: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let selected = match self {
            Self::Zero => None,
            Self::Sparse(table) => table
                .columns
                .binary_search(&column)
                .ok()
                .map(|index| (table, index)),
        };
        selected.into_iter().flat_map(|(table, index)| {
            (table.matrix.colptr[index]..table.matrix.colptr[index + 1])
                .map(|entry| (table.matrix.rowval[entry], table.matrix.nzval[entry]))
        })
    }
    pub(super) fn entries(&self) -> impl Iterator<Item = Term> + '_ {
        let table = match self {
            Self::Zero => None,
            Self::Sparse(table) => Some(table),
        };
        table.into_iter().flat_map(|table| {
            table
                .columns
                .iter()
                .enumerate()
                .flat_map(move |(local, &column)| {
                    (table.matrix.colptr[local]..table.matrix.colptr[local + 1]).map(move |index| {
                        Term {
                            column,
                            row: table.matrix.rowval[index],
                            value: table.matrix.nzval[index],
                        }
                    })
                })
        })
    }
}

mod offset;
