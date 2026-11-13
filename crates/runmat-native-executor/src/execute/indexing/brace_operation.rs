//! Direct Cell brace consumers. Selector evaluation remains at the shared
//! Native materializer; Runtime owns selection geometry and final backing.

use runmat_mir::MirIndexing;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RootedValues;
use runmat_types::IndexResultContext;
use runmat_value::{Value, ValueSequenceKind};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{semantic_error, MaterializedSelectors};

pub(super) fn read_brace(
    runtime: &RuntimeContext,
    base: Value,
    selectors: MaterializedSelectors,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<RootedValues> {
    let Value::Cell(cell) = base else {
        return Err(NativeExecutorError::from(semantic_error(
            "CellIndexType",
            "Brace indexing requires a cell array",
        )));
    };
    let indices = selectors.positional_in_context(runtime)?;
    let values = runmat_runtime::object::cell::expand_cell_indices_in_context(
        runtime,
        &cell,
        indices.values(),
    )?;
    if indexing.result_context == IndexResultContext::ReadCommaList || indexing.cell_expand_all {
        if values.values().len() != requested_outputs {
            return Err(NativeExecutorError::from(semantic_error(
                "OutputArityMismatch",
                format!(
                    "cell comma-list produced {} values for {requested_outputs} outputs",
                    values.values().len()
                ),
            )));
        }
        return Ok(values);
    }
    if values.values().len() == 1 && requested_outputs == 1 {
        Ok(values)
    } else {
        Err(NativeExecutorError::from(semantic_error(
            "CellIndexArity",
            "Cell brace indexing must select exactly one value in scalar context",
        )))
    }
}

pub(super) fn assign_brace(
    runtime: &RuntimeContext,
    base: Value,
    selectors: MaterializedSelectors,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    if delete {
        return Err(NativeExecutorError::from(semantic_error(
            "UnsupportedCellBraceDeletion",
            "Cell brace assignment does not support deletion",
        )));
    }
    let Value::Cell(cell) = base else {
        return Err(NativeExecutorError::from(semantic_error(
            "CellIndexType",
            "Brace assignment requires a cell array",
        )));
    };
    let indices = selectors.positional_in_context(runtime)?;
    let positions = runmat_runtime::object::cell::resolve_cell_assignment_positions_in_context(
        Some(runtime),
        &cell,
        indices.values(),
    )?;
    // PendingAssign retains the original receiver/RHS arena roots through
    // this call. Keep the canonical sequence backing borrowed through mutation;
    // neither a singleton output collection nor a K-element replica is needed.
    let sequence = runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?;
    match sequence.kind() {
        ValueSequenceKind::CommaSeparated => {
            runmat_runtime::object::cell::assign_cell_value_multi_in_context(
                Some(runtime),
                cell,
                positions.as_slice(),
                sequence.as_slice(),
                |_, _| {},
            )
        }
        ValueSequenceKind::Single => {
            let value = sequence.as_slice().first().ok_or_else(|| {
                semantic_error(
                    "CellAssignmentArityMismatch",
                    "single cell assignment value is missing",
                )
            })?;
            runmat_runtime::object::cell::assign_cell_value_scalar_in_context(
                Some(runtime),
                cell,
                positions.as_slice(),
                value,
                |_, _| {},
            )
        }
    }
    .map_err(NativeExecutorError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_mir::{MirIndexPlan, MirIndexing};
    use runmat_types::IndexKind;
    use runmat_value::{CellArray, Tensor};
    use std::rc::Rc;

    fn selectors(values: Vec<Value>, dims: usize) -> MaterializedSelectors {
        MaterializedSelectors {
            dims,
            colon_mask: 0,
            end_mask: 0,
            numeric: RootedValues::new(values).expect("selector roots"),
        }
    }

    #[test]
    fn brace_views_keep_csl_order_scalar_replication_and_index_error_priority() {
        runmat_gc::gc_test_context(|| {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let cell = || {
                Value::Cell(
                    CellArray::new(
                        vec![
                            Value::Num(11.),
                            Value::Num(12.),
                            Value::Num(21.),
                            Value::Num(22.),
                        ],
                        2,
                        2,
                    )
                    .unwrap(),
                )
            };
            let indexing = MirIndexing {
                argument_spans: None,
                kind: IndexKind::Brace,
                plan: MirIndexPlan::Slice,
                components: vec![],
                result_context: IndexResultContext::ReadCommaList,
                cell_expand_all: true,
            };
            let positions = || {
                selectors(
                    vec![Value::Tensor(
                        Tensor::new(vec![4., 1., 4.], vec![1, 3]).unwrap(),
                    )],
                    1,
                )
            };
            let values = read_brace(&runtime, cell(), positions(), &indexing, 3).unwrap();
            assert_eq!(
                values.values(),
                &[Value::Num(22.), Value::Num(11.), Value::Num(22.)]
            );
            let Value::Cell(updated) =
                assign_brace(&runtime, cell(), positions(), Value::Num(9.), false).unwrap()
            else {
                panic!("cell assignment receiver");
            };
            assert_eq!(
                updated.data(),
                &[
                    Value::Num(9.),
                    Value::Num(12.),
                    Value::Num(21.),
                    Value::Num(9.)
                ]
            );
            let selected = read_brace(
                &runtime,
                cell(),
                selectors(vec![Value::Num(2.), Value::Num(1.)], 2),
                &indexing,
                1,
            )
            .unwrap();
            assert_eq!(selected.values(), &[Value::Num(21.)]);
            let bad = selectors(
                vec![Value::Tensor(
                    Tensor::new(vec![1., 5.], vec![1, 2]).unwrap(),
                )],
                1,
            );
            let error =
                read_brace(&runtime, cell(), bad, &indexing, 1).expect_err("invalid later index");
            let NativeExecutorError::Runtime(error) = error else {
                panic!("original Runtime index error");
            };
            assert_eq!(error.identifier(), Some("RunMat:CellIndexOutOfBounds"));
        });
    }
}
