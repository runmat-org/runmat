//! Executor-neutral MATLAB `for` iteration semantics.
//!
//! MATLAB iterates a value by columns: the iterable is evaluated once, its
//! second dimension determines the iteration count, and each iteration binds
//! the equivalent of `value(:, column)`. Executors own control-flow state;
//! this module owns the value semantics of that operation.

use runmat_value::{CellArray, CharArray, LogicalArray, ObjectArray, SymbolicArray, Value};

use crate::indexing::plan::{build_index_plan, IndexPlan};
use crate::indexing::read_slice::{
    gather_string_slice, read_complex_slice_from_plan, read_gpu_slice_from_plan,
    read_sparse_slice_from_plan, read_tensor_slice_from_plan,
};
use crate::indexing::selectors::SliceSelector;
use crate::object::dispatch::call_object_index_descriptor_method;
use crate::object::indexing::ObjectIndexDescriptor;
use crate::{call_builtin_async, RuntimeError};

/// A snapshot of one MATLAB `for` iterable.
///
/// The source value is cloned once when the loop is entered. Later workspace
/// mutation therefore cannot change the loop's remaining values.
#[derive(Debug, Clone)]
pub struct ForColumnIterator {
    source: Value,
    next_column: usize,
    column_count: usize,
}

impl ForColumnIterator {
    /// Capture an iterable and resolve its MATLAB-visible second dimension.
    pub async fn new(source: Value) -> Result<Self, RuntimeError> {
        let size = call_builtin_async("size", &[source.clone(), Value::Num(2.0)]).await?;
        let column_count = dimension_extent(&size)?;
        Ok(Self {
            source,
            next_column: 0,
            column_count,
        })
    }

    /// Return the next column, or `None` once the captured iterable is exhausted.
    pub async fn next(&mut self) -> Result<Option<Value>, RuntimeError> {
        if self.next_column >= self.column_count {
            return Ok(None);
        }
        let column = self.next_column;
        self.next_column += 1;
        read_column(&self.source, column).await.map(Some)
    }
}

fn dimension_extent(value: &Value) -> Result<usize, RuntimeError> {
    match value {
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && *value >= 0.0
                && *value <= usize::MAX as f64 =>
        {
            Ok(*value as usize)
        }
        Value::Int(value) => value.try_to_usize().ok_or_else(invalid_dimension),
        other => Err(crate::runtime_error::semantic_error(
            "InvalidLoopDimension",
            format!("size returned a non-scalar loop dimension: {other:?}"),
        )),
    }
}

fn invalid_dimension() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "InvalidLoopDimension",
        "size returned a loop dimension outside platform limits",
    )
}

async fn read_column(source: &Value, column: usize) -> Result<Value, RuntimeError> {
    match source {
        Value::Tensor(value) => {
            read_tensor_slice_from_plan(value, &column_plan(&value.shape, column)?)
        }
        Value::ComplexTensor(value) => {
            read_complex_slice_from_plan(value, &column_plan(&value.shape, column)?)
        }
        Value::SparseTensor(value) => {
            read_sparse_slice_from_plan(value, &column_plan(&value.shape(), column)?)
        }
        Value::GpuTensor(value) => {
            read_gpu_slice_from_plan(value, &column_plan(&value.shape, column)?)
        }
        Value::StringArray(value) => {
            gather_string_slice(value, &column_plan(&value.shape, column)?)
        }
        Value::LogicalArray(value) => gather_logical(value, &column_plan(&value.shape, column)?),
        Value::CharArray(value) => gather_char(value, &column_plan(value.shape(), column)?),
        Value::Cell(value) => gather_cell(value, &column_plan(&value.shape, column)?),
        Value::ObjectArray(value) => {
            gather_object_array(value, &column_plan(value.shape(), column)?)
        }
        Value::SymbolicArray(value) => gather_symbolic(value, &column_plan(&value.shape, column)?),
        Value::Object(_) | Value::HandleObject(_) => {
            let descriptor = ObjectIndexDescriptor::subsref_paren_from_slice(
                source.clone(),
                2,
                1,
                0,
                &[Value::Num((column + 1) as f64)],
            )?;
            call_object_index_descriptor_method(descriptor).await
        }
        // MATLAB scalars have size 1x1 and bind as themselves.
        value => Ok(value.clone()),
    }
}

fn column_plan(shape: &[usize], column: usize) -> Result<IndexPlan, RuntimeError> {
    build_index_plan(
        &[SliceSelector::Colon, SliceSelector::Scalar(column + 1)],
        2,
        shape,
    )
}

fn gather_logical(value: &LogicalArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let data = gather(&value.data, plan, "logical")?;
    if data.len() == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    LogicalArray::new(data, plan.output_shape.clone())
        .map(Value::LogicalArray)
        .map_err(shape_error)
}

fn gather_char(value: &CharArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let data = gather(&value.to_column_major(), plan, "character")?;
    CharArray::from_column_major(data, plan.output_shape.clone())
        .map(Value::CharArray)
        .map_err(shape_error)
}

fn gather_cell(value: &CellArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let indices = plan
        .indices
        .iter()
        .map(|index| *index as usize + 1)
        .collect::<Vec<_>>();
    crate::object::cell::gather_cell_paren_linear_indices(value, &indices, &plan.output_shape)
}

fn gather_object_array(value: &ObjectArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    if let [index] = plan.indices.as_slice() {
        return value.get_linear(*index as usize).cloned().ok_or_else(|| {
            crate::runtime_error::semantic_error("IndexOutOfBounds", "Index out of bounds")
        });
    }
    let indices = plan
        .indices
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    value
        .select_linear(&indices, plan.output_shape.clone())
        .map(Value::ObjectArray)
        .map_err(shape_error)
}

fn gather_symbolic(value: &SymbolicArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let data = gather(&value.data, plan, "symbolic")?;
    SymbolicArray::new(data, plan.output_shape.clone())
        .map(Value::SymbolicArray)
        .map_err(shape_error)
}

fn gather<T: Clone>(data: &[T], plan: &IndexPlan, kind: &str) -> Result<Vec<T>, RuntimeError> {
    plan.indices
        .iter()
        .map(|index| {
            data.get(*index as usize).cloned().ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "IndexOutOfBounds",
                    format!("{kind} loop column index is out of bounds"),
                )
            })
        })
        .collect()
}

fn shape_error(error: impl std::fmt::Display) -> RuntimeError {
    crate::runtime_error::semantic_error("ShapeMismatch", error.to_string())
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use futures::executor::block_on;
    use runmat_value::{CellArray, IntValue, IntegerStorage, Tensor, Value};

    use super::ForColumnIterator;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;

    fn collect(source: Value) -> Vec<Value> {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(runtime.scope(async move {
            let mut columns = ForColumnIterator::new(source).await.unwrap();
            let mut values = Vec::new();
            while let Some(value) = columns.next().await.unwrap() {
                values.push(value);
            }
            values
        }))
    }

    #[test]
    fn iterates_dense_columns_without_losing_integer_storage() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX, 7, 8]),
            vec![2, 2],
        )
        .unwrap();
        let columns = collect(Value::Tensor(tensor));
        assert_eq!(columns.len(), 2);
        let Value::Tensor(first) = &columns[0] else {
            panic!("expected a tensor column")
        };
        assert_eq!(
            first.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]))
        );
    }

    #[test]
    fn iterates_cell_columns_with_matlab_paren_shape() {
        let cell = CellArray::from_column_major(
            vec![
                Value::Int(IntValue::I32(1)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(3)),
                Value::Int(IntValue::I32(4)),
            ],
            vec![2, 2],
        )
        .unwrap();
        let columns = collect(Value::Cell(cell));
        let Value::Cell(second) = &columns[1] else {
            panic!("expected a cell column")
        };
        assert_eq!(second.shape, vec![2, 1]);
        assert_eq!(
            second.to_column_major(),
            vec![Value::Int(IntValue::I32(3)), Value::Int(IntValue::I32(4))]
        );
    }
}
