mod data;
mod extract;

use crate::builtins::common::broadcast;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) use data::ArrayData;

pub(super) struct ArrayInput {
    pub(super) data: ArrayData,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
}

impl ArrayInput {
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }

    pub(super) fn value_at(&self, index: usize, output_shape: &[usize]) -> BuiltinResult<Value> {
        let source_index =
            broadcast::broadcast_index(index, output_shape, &self.shape, &self.strides);
        self.data.value_at(source_index)
    }

    pub(super) fn scalar_fact(&self) -> runmat_types::ValueFact {
        self.data.scalar_fact()
    }
}
