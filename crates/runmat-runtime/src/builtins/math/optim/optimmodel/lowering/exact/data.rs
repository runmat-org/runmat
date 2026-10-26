//! Immutable exact slots, with lazy original numeric constants and uniform
//! symbolic values. Missing assignment slots are not numerical zeros.
use super::*;
#[derive(Clone)]
enum Storage {
    Uniform(Option<Rational>),
    Original(array::Offset),
    Sparse(std::rc::Rc<HashMap<usize, Rational>>),
    Explicit(std::rc::Rc<Vec<Option<Rational>>>),
}
#[derive(Clone)]
pub(super) struct Data {
    pub shape: SharedShape,
    storage: Storage,
}
impl Data {
    pub(super) fn uniform(shape: SharedShape, value: Option<Rational>) -> Self {
        Self {
            shape,
            storage: Storage::Uniform(value),
        }
    }
    pub(super) fn constant(shape: SharedShape, offset: array::Offset) -> Self {
        Self {
            shape,
            storage: Storage::Original(offset),
        }
    }
    pub(super) fn explicit(
        shape: SharedShape,
        values: Vec<Option<Rational>>,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        if evaluation::element_count(&shape, policy)? != values.len() {
            return Err(policy.invalid("exact slots differ from original expression shape"));
        }
        Ok(Self {
            shape,
            storage: Storage::Explicit(std::rc::Rc::new(values)),
        })
    }
    pub(super) fn reshaped(mut self, shape: SharedShape, policy: Policy) -> BuiltinResult<Self> {
        if self.count(policy)? != evaluation::element_count(&shape, policy)? {
            return Err(policy.invalid("exact reshape differs from original slot count"));
        }
        self.shape = shape;
        Ok(self)
    }
    pub(super) fn count(&self, policy: Policy) -> BuiltinResult<usize> {
        evaluation::element_count(&self.shape, policy)
    }
    pub(super) fn at(&self, row: usize, policy: Policy) -> BuiltinResult<Option<Rational>> {
        if row >= self.count(policy)? {
            return Err(policy.invalid("exact slot exceeds original shape"));
        }
        match &self.storage {
            Storage::Uniform(value) => Ok(value.clone()),
            Storage::Sparse(values) => Ok(Some(
                values.get(&row).cloned().unwrap_or_else(Rational::zero),
            )),
            Storage::Original(offset) => Rational::from_float(offset.at(row, policy)?)
                .map(Some)
                .ok_or_else(|| policy.invalid("exact model constant must be finite")),
            Storage::Explicit(values) => Ok(values[row].clone()),
        }
    }
    pub(super) fn routed(shape: SharedShape, values: HashMap<usize, Rational>) -> Self {
        Self {
            shape,
            storage: Storage::Sparse(std::rc::Rc::new(values)),
        }
    }
    pub(super) async fn diagonal(
        &self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let n = self.shape.first().copied().unwrap_or(1);
        if n <= 1 {
            return Ok(true);
        }
        match &self.storage {
            Storage::Uniform(value) => Ok(value.as_ref().is_some_and(Rational::is_zero)),
            Storage::Original(array::Offset::Uniform(value)) => Ok(*value == 0.),
            Storage::Sparse(values) => {
                for (&row, value) in values.iter() {
                    if row % n != row / n && !value.is_zero() {
                        return Ok(false);
                    }
                    c.completed(1).await?;
                }
                Ok(true)
            }
            Storage::Original(array::Offset::Sparse(matrix)) => {
                for column in 0..matrix.cols {
                    for entry in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
                        let row = matrix.row_indices[entry] + column * matrix.rows;
                        if row % n != row / n
                            && self.at(row, policy)?.is_none_or(|value| !value.is_zero())
                        {
                            return Ok(false);
                        }
                        c.completed(1).await?;
                    }
                    c.completed(1).await?;
                }
                Ok(true)
            }
            _ => {
                for row in 0..self.count(policy)? {
                    if row % n != row / n
                        && self.at(row, policy)?.is_none_or(|value| !value.is_zero())
                    {
                        return Ok(false);
                    }
                    c.completed(1).await?;
                }
                Ok(true)
            }
        }
    }
    pub(super) fn sparse(&self) -> Option<&runmat_value::SparseTensor> {
        match &self.storage {
            Storage::Original(array::Offset::Sparse(array))
                if self.shape.first().copied().unwrap_or(1) == array.rows
                    && self.shape.get(1).copied().unwrap_or(1) == array.cols
                    && self.shape.iter().skip(2).all(|&n| n == 1) =>
            {
                Some(array)
            }
            _ => None,
        }
    }
    pub(super) fn uniform_value(&self) -> Option<&Option<Rational>> {
        match &self.storage {
            Storage::Uniform(value) => Some(value),
            _ => None,
        }
    }
}
pub(super) fn buffer(count: usize, policy: Policy) -> BuiltinResult<Vec<Option<Rational>>> {
    runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))
}
