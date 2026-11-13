//! Affine arrays retain real immutable numeric backing or a uniform constant
//! polynomial part. No dense array of numerical probe zeros is constructed.
use super::terms::Coefficients;
use super::*;
mod projection;
mod degree;
pub(super) use degree::Degree;
mod arithmetic;
mod reduction;
mod matrix;
mod product;
mod permutation;
mod assignment;
mod concatenate;
use runmat_value::{NumericDType, NumericScalar, SparseTensor, Tensor};

#[derive(Clone)]
pub(super) enum Offset {
    Uniform(f64),
    Dense(Tensor),
    Sparse(SparseTensor),
}
impl Offset {
    pub(super) fn from_value(value: &Value, policy: Policy) -> Lowered<Self> {
        let result = match value {
            Value::Num(value) => Self::Uniform(*value),
            Value::Tensor(value) if value.numeric_dtype() == NumericDType::F64 => {
                Self::Dense(value.clone())
            }
            Value::SparseTensor(value)
                if value.numeric_dtype() == Some(NumericDType::F64) && !value.is_complex() =>
            {
                Self::Sparse(value.clone())
            }
            _ => return Err(policy.invalid("compiled constant lost its real double backing")),
        };
        Ok(Ok(result))
    }
    pub(super) fn value(&self) -> Value {
        match self {
            Self::Uniform(value) => Value::Num(*value),
            Self::Dense(value) => Value::Tensor(value.clone()),
            Self::Sparse(value) => Value::SparseTensor(value.clone()),
        }
    }
    pub(super) fn at(&self, index: usize, policy: Policy) -> BuiltinResult<f64> {
        match self {
            Self::Uniform(value) => Ok(*value),
            Self::Dense(value) => match value.numeric_value_at(index) {
                Some(NumericScalar::F64(value)) => Ok(value),
                _ => Err(policy.invalid("compiled constant coordinate exceeds its storage")),
            },
            Self::Sparse(value) => {
                let count = value
                    .rows
                    .checked_mul(value.cols)
                    .ok_or_else(|| policy.invalid("sparse constant dimensions overflow usize"))?;
                if index >= count {
                    return Err(
                        policy.invalid("compiled constant coordinate exceeds its sparse shape")
                    );
                }
                // The immutable backing was class-checked on construction and
                // the coordinate is in bounds. Missing CSC entries are zeros,
                // not missing numeric storage or a changed class.
                Ok(value
                    .get(index % value.rows, index / value.rows)
                    .unwrap_or(0.0))
            }
        }
    }
}
#[derive(Clone)]
pub(super) enum Assigned {
    All,
    Missing,
    Partial(std::rc::Rc<Vec<bool>>),
}
impl Assigned {
    pub(super) fn at(&self, index: usize, policy: Policy) -> BuiltinResult<bool> {
        match self {
            Self::All => Ok(true),
            Self::Missing => Ok(false),
            Self::Partial(values) => values
                .get(index)
                .copied()
                .ok_or_else(|| policy.invalid("compiled assignment mask exceeds its shape")),
        }
    }
}
#[derive(Clone)]
pub(super) struct Array {
    pub shape: SharedShape,
    pub offset: Offset,
    pub coefficients: Coefficients,
    pub assigned: Assigned,
    pub degree: Degree,
}
impl Array {
    pub(super) async fn reshaped(
        mut self,
        arguments: &[Value],
        shape: SharedShape,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        if self.count(policy)? != evaluation::element_count(&shape, policy)? {
            return Err(
                policy.invalid("reshape differs from the original expression element count")
            );
        }
        match &mut self.offset {
            Offset::Uniform(_) => {}
            Offset::Dense(value) => value
                .reshape_with_shared_shape(shape.clone())
                .map_err(|e| policy.invalid(e))?,
            Offset::Sparse(value) => {
                let mut inputs = runmat_value::exact_capacity_vec(
                    arguments
                        .len()
                        .checked_add(1)
                        .ok_or_else(|| policy.invalid("reshape argument count overflow"))?,
                )
                .map_err(|e| policy.storage(e))?;
                inputs.push(Value::SparseTensor(value.clone()));
                for argument in arguments {
                    inputs.push(argument.clone());
                    cooperation.completed(1).await?;
                }
                let result = evaluation::invoke("reshape", &inputs, policy, cooperation).await?;
                self.offset = match Offset::from_value(result.value(), policy)? {
                    Ok(value) => value,
                    Err(stop) => return Ok(Err(stop)),
                };
            }
        }
        self.shape = shape;
        Ok(Ok(self))
    }
    pub(super) async fn concatenate(
        inputs: Vec<Self>,
        dimension: usize,
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        concatenate::concatenate(inputs, dimension, shape, columns, policy, cooperation).await
    }
    pub(super) async fn assign(
        self,
        replacement: Self,
        indices: &[usize],
        selection_shape: &[usize],
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        assignment::assign(
            self,
            replacement,
            indices,
            selection_shape,
            columns,
            policy,
            cooperation,
        )
        .await
    }
    pub(super) async fn transposed(
        self,
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        permutation::transpose(self, shape, columns, policy, cooperation).await
    }
    pub(super) async fn product(
        self,
        right: Self,
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        product::multiply(self, right, shape, columns, policy, cooperation).await
    }
    pub(super) async fn matrix_product(
        self,
        right: Self,
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        matrix::multiply(self, right, shape, columns, policy, cooperation).await
    }

    pub(super) async fn sum(
        self,
        arguments: &[Value],
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        reduction::sum(self, arguments, shape, columns, policy, cooperation).await
    }

    pub(super) fn scalar_constant(&self, policy: Policy) -> BuiltinResult<Option<f64>> {
        if self.degree != Degree::Constant
            || self.count(policy)? != 1
            || !self.assigned.at(0, policy)?
        {
            return Ok(None);
        }
        if !self.coefficients.is_zero() {
            return Err(
                policy.invalid("constant source unexpectedly contains decision coefficients")
            );
        }
        self.offset.at(0, policy).map(Some)
    }

    pub(super) async fn add(
        self,
        right: Self,
        operator: runmat_types::OperatorKind,
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        arithmetic::add(self, right, operator, shape, columns, policy, cooperation).await
    }

    pub(super) async fn scaled(
        self,
        factor: f64,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        arithmetic::scale(self, factor, columns, policy, cooperation).await
    }

    pub(super) fn unassigned(shape: SharedShape) -> Self {
        Self {
            shape,
            offset: Offset::Uniform(0.),
            coefficients: Coefficients::Zero,
            assigned: Assigned::Missing,
            degree: Degree::Constant,
        }
    }
    pub(super) fn zero(shape: SharedShape) -> Self {
        Self {
            shape,
            offset: Offset::Uniform(0.),
            coefficients: Coefficients::Zero,
            assigned: Assigned::All,
            degree: Degree::Constant,
        }
    }
    pub(super) async fn gather(
        self,
        indices: &[usize],
        shape: SharedShape,
        columns: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        if matches!(self.offset, Offset::Uniform(_))
            && matches!(self.assigned, Assigned::All)
            && self.coefficients.is_zero()
        {
            let count = self.count(policy)?;
            if indices.len() != evaluation::element_count(&shape, policy)? {
                return Err(
                    policy.invalid("compiled selection differs from its original result shape")
                );
            }
            for &index in indices {
                if index >= count {
                    return Err(
                        policy.invalid("compiled selector exceeds original expression rows")
                    );
                }
                cooperation.completed(1).await?;
            }
            let degree = Degree::collect(
                indices.len(),
                |row| Ok(self.degree.at(indices[row])),
                policy,
                cooperation,
            )
            .await?;
            return Ok(Ok(Self {
                shape,
                degree,
                ..self
            }));
        }
        let map =
            match projection::RowMap::gather(indices, self.count(policy)?, policy, cooperation)
                .await?
            {
                Ok(map) => map,
                Err(stop) => return Ok(Err(stop)),
            };
        projection::project(self, &map, shape, columns, policy, cooperation).await
    }
    pub(super) async fn variable(
        object: &ObjectInstance,
        layout: &super::layout::Layout,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> Lowered<Self> {
        let view = super::super::super::optimvar::View::new(object, cooperation).await?;
        let original = layout.variable(view.selection().definition(), policy)?;
        // The view's stored selection maps actual original coordinates. The
        // layout's retained definition generation, not a refreshed view, owns
        // this solve's domain and bound snapshot.
        let count = view.selection().element_count();
        let mut entries = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for row in 0..count {
            let coordinate = view.selection().definition_index(row).ok_or_else(|| {
                policy.invalid("variable selection exceeds its original coordinate map")
            })?;
            if coordinate >= original.count {
                return Err(
                    policy.invalid("variable selection differs from original compiled dimensions")
                );
            }
            let column = original
                .start
                .checked_add(coordinate)
                .ok_or_else(|| policy.invalid("decision-coordinate offset overflows usize"))?;
            entries.push(super::terms::Term {
                column,
                row,
                value: 1.,
            });
            cooperation.completed(1).await?;
        }
        let coefficients =
            match Coefficients::from_entries(entries, count, layout.columns, policy, cooperation)
                .await?
            {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
        Ok(Ok(Self {
            shape: view.selection().shape().clone(),
            offset: Offset::Uniform(0.),
            coefficients,
            assigned: Assigned::All,
            degree: if count == 0 {
                Degree::Constant
            } else {
                Degree::Affine
            },
        }))
    }
    pub(super) fn constant(value: &Value, shape: SharedShape, policy: Policy) -> Lowered<Self> {
        let offset = match Offset::from_value(value, policy)? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        Ok(Ok(Self {
            shape,
            offset,
            coefficients: Coefficients::Zero,
            assigned: Assigned::All,
            degree: Degree::Constant,
        }))
    }
    pub(super) fn count(&self, policy: Policy) -> BuiltinResult<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| policy.invalid("compiled expression dimensions overflow usize"))
    }
}
