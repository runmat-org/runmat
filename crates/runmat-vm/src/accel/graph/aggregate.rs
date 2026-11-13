//! Known-arity aggregate tracking; overloaded/captured construction is a barrier.
use super::*;
use crate::bytecode::AggregateElementSpec;

impl GraphBuilder<'_> {
    pub(super) fn handle_matrix_aggregate(
        &mut self,
        pc: usize,
        rows: usize,
        row_lengths: &[usize],
        elements: &[AggregateElementSpec],
    ) {
        let count = row_lengths
            .iter()
            .try_fold(0usize, |sum, length| sum.checked_add(*length));
        if rows != row_lengths.len()
            || count != Some(elements.len())
            || elements
                .iter()
                .any(|element| !matches!(element, AggregateElementSpec::Single))
            || self.stack.len() < elements.len()
        {
            self.mark_unknown();
            return;
        }
        let mut inputs = Vec::with_capacity(elements.len());
        for _ in elements {
            inputs.push(self.pop_value().expect("checked aggregate operand arity"));
        }
        inputs.reverse();
        let numeric = inputs.iter().all(|id| {
            matches!(
                self.values[*id as usize].ty,
                Type::Num | Type::Int | Type::Bool | Type::Tensor { .. } | Type::Logical { .. }
            )
        });
        if !numeric {
            // horzcat/vertcat on objects can run user code and mutate state.
            // Do not preserve stale graph bindings across an unproven call.
            self.mark_unknown();
            return;
        }
        let mut offset = 0;
        let types: Vec<Vec<Type>> = row_lengths
            .iter()
            .map(|length| {
                let row = inputs[offset..offset + length]
                    .iter()
                    .map(|id| self.values[*id as usize].ty.clone())
                    .collect();
                offset += length;
                row
            })
            .collect();
        let shape = checked_concat_shape(&types);
        let logical = !inputs.is_empty()
            && inputs.iter().all(|id| {
                matches!(
                    self.values[*id as usize].ty,
                    Type::Bool | Type::Logical { .. }
                )
            });
        let ty = if logical {
            Type::Logical { shape }
        } else {
            Type::Tensor { shape }
        };

        // Preserve the existing acceleration-only f64 literal fold. Native
        // integer, Single, logical and complex owners remain ordinary values;
        // they are never converted through a binary64 compatibility mirror.
        let columns = row_lengths.first().copied().unwrap_or(0);
        let uniform = row_lengths.iter().all(|length| *length == columns);
        let constants: Option<Vec<f64>> = inputs
            .iter()
            .map(|id| match self.values[*id as usize].constant.as_ref() {
                Some(Value::Num(number)) => Some(*number),
                _ => None,
            })
            .collect();
        let folded = constants
            .filter(|_| uniform && !inputs.is_empty())
            .and_then(|values| {
                let mut data = vec![0.0; values.len()];
                for row in 0..rows {
                    for column in 0..columns {
                        data[row + column * rows] = values[row * columns + column];
                    }
                }
                runmat_value::Tensor::new(data, vec![rows, columns]).ok()
            });
        let node_id = self.nodes.len() as NodeId;
        // This opaque aggregate is not an elementwise execution recipe. Only
        // a proved literal receives the existing FoldedLiteralArray tag.
        let tags = if folded.is_some() {
            vec![AccelGraphTag::FoldedLiteralArray]
        } else {
            Vec::new()
        };
        let output = match folded {
            Some(tensor) => self.new_value(
                ValueOrigin::NodeOutput {
                    node: node_id,
                    output: 0,
                },
                ty,
                Some(Value::Tensor(tensor)),
            ),
            None => self.new_node_output(node_id, 0, ty),
        };
        self.nodes.push(AccelNode {
            id: node_id,
            label: AccelNodeLabel::Primitive(PrimitiveOp::UPlus),
            category: AccelOpCategory::Other,
            inputs,
            outputs: vec![output],
            span: InstrSpan { start: pc, end: pc },
            tags,
        });
        self.stack.push(output);
    }
}

fn checked_concat_shape(rows: &[Vec<Type>]) -> Option<Vec<Option<usize>>> {
    // Canonical shape inference owns concatenation rules. Avoid calling it
    // with ambiguous empty/ND facts or extents whose sum could overflow its
    // metadata arithmetic; those retain numeric type but unknown shape.
    rows.iter().flatten().try_fold(0usize, |sum, ty| match ty {
        Type::Num | Type::Int | Type::Bool => sum.checked_add(2),
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) }
            if shape.len() <= 2 && !shape.contains(&Some(0)) =>
        {
            shape
                .iter()
                .flatten()
                .try_fold(sum, |sum, dimension| sum.checked_add(*dimension))
        }
        _ => None,
    })?;
    runmat_builtins::shape_rules::concat_shape(rows)
}

#[cfg(test)]
#[path = "aggregate/tests.rs"]
mod tests;
