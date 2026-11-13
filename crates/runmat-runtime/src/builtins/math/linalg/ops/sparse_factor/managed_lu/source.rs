//! Borrowed shifted-pencil columns. No merged CSC or dense coefficient copy.
mod dense;
use super::{ManagedSparseLuError, ManagedSparseScalar, Work};
use dense::Dense;
use runmat_value::{SparseTensor, Value};
pub(super) enum PencilB<'a> {
    Sparse(&'a SparseTensor),
    Dense(Dense<'a>),
}
enum Rows<'a> {
    Sparse(std::iter::Copied<std::slice::Iter<'a, usize>>),
    Dense(std::ops::Range<usize>),
}
impl Iterator for Rows<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Sparse(v) => v.next(),
            Self::Dense(v) => v.next(),
        }
    }
}
impl PencilB<'_> {
    fn rows(&self, column: usize) -> Rows<'_> {
        match self {
            Self::Sparse(b) => Rows::Sparse(
                b.row_indices[b.col_ptrs[column]..b.col_ptrs[column + 1]]
                    .iter()
                    .copied(),
            ),
            Self::Dense(b) => Rows::Dense(0..b.order),
        }
    }
}
pub(super) enum Source<'a, T> {
    Matrix(&'a SparseTensor),
    Pencil {
        a: &'a SparseTensor,
        b: Option<PencilB<'a>>,
        shift: T,
    },
}
impl<'a, T: ManagedSparseScalar> Source<'a, T> {
    pub(super) fn pencil(
        a: &'a SparseTensor,
        b: Option<&'a Value>,
        shift: T,
    ) -> Result<Self, ManagedSparseLuError> {
        let b = b
            .map(|value| match value {
                Value::SparseTensor(value) => Ok(PencilB::Sparse(value)),
                value => Dense::borrow(value).map(PencilB::Dense),
            })
            .transpose()?;
        Ok(Self::Pencil { a, b, shift })
    }
    pub(super) fn sparse_pencil(
        a: &'a SparseTensor,
        b: Option<&'a SparseTensor>,
        shift: T,
    ) -> Self {
        Self::Pencil {
            a,
            b: b.map(PencilB::Sparse),
            shift,
        }
    }
    pub(super) fn primary(&self) -> &'a SparseTensor {
        match self {
            Self::Matrix(a) | Self::Pencil { a, .. } => a,
        }
    }
    pub(super) fn matrices(&self) -> [Option<&'a SparseTensor>; 2] {
        [
            Some(self.primary()),
            match self {
                Self::Pencil {
                    b: Some(PencilB::Sparse(b)),
                    ..
                } => Some(*b),
                _ => None,
            },
        ]
    }
    pub(super) fn validate(&self) -> Result<(), ManagedSparseLuError> {
        let n = self.primary().rows;
        if self.primary().cols != n {
            return Err(ManagedSparseLuError::NotSquare);
        }
        for matrix in self.matrices().into_iter().flatten() {
            if matrix.rows != n || matrix.cols != n {
                return Err(ManagedSparseLuError::DimensionMismatch);
            }
            let matches = match self {
                Self::Matrix(_) => T::matches(matrix),
                Self::Pencil { .. } => T::matches_pencil(matrix),
            };
            if !matches {
                return Err(ManagedSparseLuError::UnsupportedClass);
            }
        }
        if let Self::Pencil { b, shift, .. } = self {
            if !shift.finite() {
                return Err(ManagedSparseLuError::NonFiniteInput);
            }
            if let Some(PencilB::Dense(b)) = b {
                if b.order != n {
                    return Err(ManagedSparseLuError::DimensionMismatch);
                }
                if !b.matches::<T>() {
                    return Err(ManagedSparseLuError::UnsupportedClass);
                }
            }
        }
        Ok(())
    }
    pub(super) async fn validate_dense(
        &self,
        work: &mut Work<'_>,
    ) -> Result<(), ManagedSparseLuError> {
        if let Self::Pencil {
            b: Some(PencilB::Dense(b)),
            ..
        } = self
        {
            for column in 0..b.order {
                for row in 0..b.order {
                    b.read::<T>(row, column)?;
                    work.tick().await?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn rows(&self, column: usize) -> impl Iterator<Item = usize> + '_ {
        let a = self.primary();
        let (b, identity) = match self {
            Self::Pencil { b, shift, .. } if *shift != T::default() => (b.as_ref(), b.is_none()),
            _ => (None, false),
        };
        a.row_indices[a.col_ptrs[column]..a.col_ptrs[column + 1]]
            .iter()
            .copied()
            .chain(b.into_iter().flat_map(move |b| b.rows(column)))
            .chain(identity.then_some(column))
    }
    pub(super) fn read(&self, matrix: &SparseTensor, index: usize) -> T {
        match self {
            Self::Matrix(_) => T::read(matrix, index),
            Self::Pencil { .. } => T::read_pencil(matrix, index),
        }
    }
    pub(super) async fn scatter(
        &self,
        column: usize,
        x: &mut [T],
        work: &mut Work<'_>,
    ) -> Result<(), ManagedSparseLuError> {
        let a = self.primary();
        for index in a.col_ptrs[column]..a.col_ptrs[column + 1] {
            x[a.row_indices[index]] = self.read(a, index);
            work.tick().await?;
        }
        if let Self::Pencil { b, shift, .. } = self {
            if *shift != T::default() {
                match b {
                    Some(PencilB::Sparse(b)) => {
                        for index in b.col_ptrs[column]..b.col_ptrs[column + 1] {
                            let row = b.row_indices[index];
                            x[row] = x[row] - *shift * self.read(b, index);
                            work.tick().await?;
                        }
                    }
                    Some(PencilB::Dense(b)) => {
                        for (row, value) in x.iter_mut().enumerate() {
                            *value = *value - *shift * b.read::<T>(row, column)?;
                            work.tick().await?;
                        }
                    }
                    None => {
                        x[column] = x[column] - *shift;
                        work.tick().await?;
                    }
                }
            }
        }
        for row in self.rows(column) {
            if !x[row].finite() {
                return Err(ManagedSparseLuError::NonFiniteInput);
            }
            work.tick().await?;
        }
        Ok(())
    }
}
