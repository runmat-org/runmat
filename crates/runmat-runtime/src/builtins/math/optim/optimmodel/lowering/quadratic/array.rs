//! Coefficient slots share immutable source constants and selected references.
//! Only active algebra results own physical polynomial rows.
use super::*;
#[derive(Clone)]
enum Storage {
    Uniform(Option<Polynomial>),
    Constant(super::super::array::Offset),
    Variable(Value, usize),
    Rows(std::rc::Rc<Vec<Option<Polynomial>>>),
}
#[derive(Clone)]
pub(super) struct Array {
    pub shape: SharedShape,
    storage: Storage,
}
impl Array {
    pub fn sparse_constant(&self) -> Option<&runmat_value::SparseTensor> {
        match &self.storage {
            Storage::Constant(super::super::array::Offset::Sparse(value)) => Some(value),
            _ => None,
        }
    }
    pub async fn all_assigned(
        &self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        for row in 0..self.count(policy)? {
            if self.at(row, policy)?.is_none() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
    pub fn zero(shape: SharedShape) -> Self {
        Self {
            shape,
            storage: Storage::Uniform(Some(Polynomial::zero())),
        }
    }
    pub fn missing(shape: SharedShape) -> Self {
        Self {
            shape,
            storage: Storage::Uniform(None),
        }
    }
    pub fn constant(shape: SharedShape, offset: super::super::array::Offset) -> Self {
        Self {
            shape,
            storage: Storage::Constant(offset),
        }
    }
    pub fn variable(value: Value, start: usize, shape: SharedShape) -> Self {
        Self {
            shape,
            storage: Storage::Variable(value, start),
        }
    }
    pub fn count(&self, policy: Policy) -> BuiltinResult<usize> {
        evaluation::element_count(&self.shape, policy)
    }
    pub fn at(&self, row: usize, policy: Policy) -> BuiltinResult<Option<Polynomial>> {
        if row >= self.count(policy)? {
            return Err(policy.invalid("quadratic coefficient row exceeds original shape"));
        }
        Ok(match &self.storage {
            Storage::Uniform(value) => value.clone(),
            Storage::Constant(offset) => Some(Polynomial::scalar(offset.at(row, policy)?, policy)?),
            Storage::Rows(rows) => rows
                .get(row)
                .ok_or_else(|| {
                    policy.invalid("quadratic coefficient storage exceeds original shape")
                })?
                .clone(),
            Storage::Variable(Value::Object(object), start) => {
                let Some(
                    runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
                        selection,
                    ),
                ) = object.optimization_variable_payload()
                else {
                    return Err(policy.invalid("quadratic variable lost original reference"));
                };
                let index = selection
                    .definition_index(row)
                    .and_then(|index| start.checked_add(index))
                    .ok_or_else(|| {
                        policy.invalid("quadratic variable coordinate exceeds original layout")
                    })?;
                Some(Polynomial::variable(index))
            }
            Storage::Variable(..) => {
                return Err(policy.invalid("quadratic variable lost native object"))
            }
        })
    }
    pub fn rows(
        shape: SharedShape,
        rows: Vec<Option<Polynomial>>,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        if rows.len() != evaluation::element_count(&shape, policy)? {
            return Err(policy.invalid("quadratic coefficient storage differs from original shape"));
        }
        Ok(Self {
            shape,
            storage: Storage::Rows(std::rc::Rc::new(rows)),
        })
    }
    pub fn reshaped(mut self, shape: SharedShape, policy: Policy) -> BuiltinResult<Self> {
        if self.count(policy)? != evaluation::element_count(&shape, policy)? {
            return Err(policy.invalid("quadratic reshape changes original element count"));
        }
        self.shape = shape;
        Ok(self)
    }
    pub async fn gather(
        &self,
        indices: &[usize],
        shape: SharedShape,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut rows =
            runmat_value::exact_capacity_vec(indices.len()).map_err(|e| policy.storage(e))?;
        for &index in indices {
            rows.push(self.at(index, policy)?);
            c.completed(1).await?;
        }
        Self::rows(shape, rows, policy)
    }
    pub async fn scaled(
        &self,
        factor: &Weight,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut rows =
            runmat_value::exact_capacity_vec(self.count(policy)?).map_err(|e| policy.storage(e))?;
        for row in 0..self.count(policy)? {
            rows.push(match self.at(row, policy)? {
                Some(value) => Some(value.scaled(factor, policy, c).await?),
                None => None,
            });
            c.completed(1).await?;
        }
        Self::rows(self.shape.clone(), rows, policy)
    }
}
