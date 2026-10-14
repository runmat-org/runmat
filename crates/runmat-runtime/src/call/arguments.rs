use serde::{Deserialize, Serialize};

use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueSequence, RootedValues};
use crate::object::indexing::{ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath};
use crate::sequence::ResolveValueSequence;
use crate::{runtime_error::semantic_error, RuntimeError};
use runmat_value::{CellArray, IntValue, Value, ValueSequence};

pub(crate) mod roots;

/// Describes one source-level call argument after lowering.
///
/// The descriptor is shared by bytecode and native execution. It captures only
/// language-level expansion semantics and deliberately contains no operand-stack
/// or instruction-decoding state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentSpec {
    Single,
    Expansion(ArgumentExpansionSpec),
    CapturedSequence { slot: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentExpansionSpec {
    CellContents {
        num_indices: usize,
        expand_all: bool,
    },
    ReturnedOutputs,
    Member(runmat_types::MemberName),
    DynamicMember,
}

impl ArgumentSpec {
    pub const fn stack_operand_count(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::Expansion(ArgumentExpansionSpec::CellContents { num_indices, .. }) => {
                1 + *num_indices
            }
            Self::Expansion(ArgumentExpansionSpec::ReturnedOutputs)
            | Self::Expansion(ArgumentExpansionSpec::Member(_)) => 1,
            Self::Expansion(ArgumentExpansionSpec::DynamicMember) => 2,
            Self::CapturedSequence { .. } => 0,
        }
    }
}

/// One source-level argument after an executor has materialized its operands.
///
/// VM stack order and Native IR value identities stay in their executors; the
/// expansion rules themselves are shared Runtime language semantics.
#[derive(Debug)]
pub enum MaterializedArgument {
    Single(Value),
    Expansion(MaterializedExpansionSource),
    Sequence(ValueSequence),
}

#[derive(Debug, Clone)]
pub enum MaterializedExpansionSource {
    CellContents {
        base: Value,
        indices: Vec<Value>,
        expand_all: bool,
    },
    ReturnedOutputs(ValueSequence),
    Member {
        base: Value,
        member: runmat_types::MemberName,
    },
    DynamicMember {
        base: Value,
        member: Value,
    },
}

pub async fn expand_arguments(
    runtime: &RuntimeContext,
    arguments: Vec<MaterializedArgument>,
) -> Result<RootedValues, RuntimeError> {
    let roots = roots::ExpansionRoots::new()?;
    roots.retain_arguments(&arguments);
    let mut expanded_arguments = Vec::new();
    for argument in arguments {
        match argument {
            MaterializedArgument::Single(value) => expanded_arguments.push(value),
            MaterializedArgument::Expansion(source) => {
                materialize_expansion(runtime, source)
                    .await?
                    .with_owned_sequence(|sequence| {
                        let values = sequence.resolve(
                            runmat_types::SequenceUse::ExpandAll,
                            crate::sequence::SequenceResolutionContext::default(),
                        )?;
                        roots.retain_values(&values);
                        expanded_arguments.extend(values);
                        Ok::<_, RuntimeError>(())
                    })?;
            }
            MaterializedArgument::Sequence(sequence) => {
                let values = sequence.resolve(
                    runmat_types::SequenceUse::ExpandAll,
                    crate::sequence::SequenceResolutionContext::default(),
                )?;
                roots.retain_values(&values);
                expanded_arguments.extend(values);
            }
        }
    }
    RootedValues::new(expanded_arguments).map_err(|error| error.into_runtime_error())
}

pub async fn materialize_expansion(
    runtime: &RuntimeContext,
    source: MaterializedExpansionSource,
) -> Result<RootedValueSequence, RuntimeError> {
    let source_roots = roots::ExpansionRoots::new()?;
    source_roots.retain_source(&source);
    let values = match source {
        MaterializedExpansionSource::CellContents {
            base,
            indices,
            expand_all,
        } => {
            if expand_all {
                match base {
                    Value::Cell(cell) => crate::object::cell::expand_all_cell_values(&cell)?,
                    Value::Composite(handle) => {
                        return expand_composite_values(runtime, *handle, &[], true)
                            .await?
                            .with_owned_values(rooted_sequence_from_values);
                    }
                    base @ (Value::Object(_) | Value::HandleObject(_)) => {
                        return expand_brace_values(base, &[], None)
                            .await?
                            .with_owned_values(rooted_sequence_from_values);
                    }
                    _ => {
                        return Err(semantic_error(
                            "InvalidExpandAllTarget",
                            "Comma-separated-list expansion requires a cell array or object",
                        ));
                    }
                }
            } else {
                match (base, indices.len()) {
                    (Value::Cell(cell), 1..) => {
                        crate::object::cell::expand_cell_indices(&cell, &indices)?
                    }
                    (Value::Composite(handle), _) => {
                        return expand_composite_values(runtime, *handle, &indices, false)
                            .await?
                            .with_owned_values(rooted_sequence_from_values);
                    }
                    (base @ (Value::Object(_) | Value::HandleObject(_)), _) => {
                        return expand_brace_values(base, &indices, None)
                            .await?
                            .with_owned_values(rooted_sequence_from_values);
                    }
                    _ => {
                        return Err(semantic_error(
                            "InvalidExpandTarget",
                            "Indexed comma-separated-list expansion requires a cell array or object",
                        ));
                    }
                }
            }
        }
        MaterializedExpansionSource::ReturnedOutputs(sequence) => {
            return RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error());
        }
        MaterializedExpansionSource::Member { base, member } => {
            return crate::object::resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                member.0,
                false,
                None,
            )
            .await;
        }
        MaterializedExpansionSource::DynamicMember { base, member } => {
            let member = String::try_from(&member)
                .map_err(|error| semantic_error("DynamicFieldName", error))?;
            return crate::object::resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                member,
                false,
                None,
            )
            .await;
        }
    };
    rooted_sequence_from_values(values)
}

fn rooted_sequence_from_values(values: Vec<Value>) -> Result<RootedValueSequence, RuntimeError> {
    RootedValueSequence::new(
        crate::sequence::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?,
    )
    .map_err(|error| error.into_runtime_error())
}

/// The sole executor bridge from legacy builtin multi-output values into the
/// transient sequence carrier. Builtins remain unmigrated for now, but an
/// OutputList can never be accepted as one of another OutputList's outputs.
pub fn adapt_legacy_builtin_result(value: Value) -> Result<ValueSequence, RuntimeError> {
    match value {
        Value::OutputList(values) => ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime),
        value => ValueSequence::single(value).map_err(crate::sequence::sequence_error_to_runtime),
    }
}

/// Raw projection for the legacy builtin `Value` shape. Managed outputs may
/// pass through this function only while a rooted source owner remains live and
/// a rooted replacement is registered before that owner retires.
pub(crate) fn project_legacy_builtin_value_abi(sequence: ValueSequence) -> Value {
    let kind = sequence.kind();
    let mut values = sequence.into_values();
    match kind {
        runmat_value::ValueSequenceKind::Single => values.remove(0),
        runmat_value::ValueSequenceKind::CommaSeparated => Value::OutputList(values),
    }
}

/// Project an owning sequence at the legacy scalar ABI boundary without
/// retiring its managed roots before the replacement value is registered.
pub(crate) fn project_rooted_legacy_builtin_value_abi(
    sequence: RootedValueSequence,
) -> Result<RootedValue, RuntimeError> {
    sequence.try_map_owned(
        |sequence| Ok(project_legacy_builtin_value_abi(sequence)),
        |error| error.into_runtime_error(),
    )
}

async fn expand_composite_values(
    runtime: &RuntimeContext,
    handle: runmat_execution::CompositeHandle,
    indices: &[Value],
    expand_all: bool,
) -> Result<RootedValues, RuntimeError> {
    crate::parallel::lease::validate_composite(runtime, &handle)?;
    if !expand_all && indices.len() != 1 {
        return Err(semantic_error(
            "CompositeIndexArity",
            "Composite brace indexing requires one lab index",
        ));
    }

    let lab_count = handle.gang.labs.0;
    let rank_values = (1..=lab_count)
        .map(|rank| Value::Int(IntValue::U32(rank)))
        .collect::<Vec<_>>();
    let rank_cell = CellArray::new(rank_values, 1, lab_count as usize).map_err(|error| {
        semantic_error(
            "ShapeMismatch",
            format!("Composite lab selector construction failed: {error}"),
        )
    })?;
    let selected_ranks = if expand_all {
        crate::object::cell::expand_all_cell_values(&rank_cell)?
    } else {
        crate::object::cell::expand_cell_indices(&rank_cell, indices)?
    };

    let distributed = runtime
        .service_ports()
        .require_distributed("Composite comma-separated-list expansion")
        .map_err(|error| semantic_error("RuntimeCapabilityUnavailable", error.to_string()))?;
    let mut values = crate::execution::RootedValuesBuilder::with_capacity(selected_ranks.len());
    for selected_rank in selected_ranks {
        let Value::Int(IntValue::U32(rank)) = selected_rank else {
            return Err(semantic_error(
                "CompositeIndexType",
                "Composite lab selector did not resolve to an exact lab rank",
            ));
        };
        let value = distributed
            .composite_entry(handle.clone(), runmat_types::LabRank(rank))
            .await?
            .ok_or_else(|| {
                semantic_error(
                    "CompositeEntryUnavailable",
                    "the selected lab did not assign this Composite value",
                )
            })?;
        values.push_owned(value);
    }
    values.finish().map_err(|error| error.into_runtime_error())
}

pub async fn expand_brace_values(
    base: Value,
    indices: &[Value],
    pad_to_outputs: Option<usize>,
) -> Result<RootedValues, RuntimeError> {
    let input_roots = roots::ExpansionRoots::new()?;
    input_roots.retain_iter(std::iter::once(&base).chain(indices.iter()));
    let mut values = match base {
        Value::Cell(cell) => {
            if indices.is_empty() {
                if let Some(output_count) = pad_to_outputs {
                    crate::object::cell::expand_cell_values(&cell, &[], output_count)?
                } else {
                    crate::object::cell::expand_all_cell_values(&cell)?
                }
            } else {
                crate::object::cell::expand_cell_indices(&cell, indices)?
            }
        }
        base @ (Value::Object(_) | Value::HandleObject(_)) => {
            let path = ObjectSubscriptPath::single(ObjectSubscript::braces(
                ObjectIndexSelector::IndexValues {
                    components: indices.iter().cloned().map(Into::into).collect(),
                },
            ));
            let resolution = crate::object::dispatch::resolve_object_index_protocol(
                &base,
                crate::object::indexing::ObjectIndexOp::Subsref,
                None,
            )?;
            let value = crate::object::dispatch::invoke_resolved_object_index_path_method(
                &resolution,
                base,
                path,
                pad_to_outputs.unwrap_or(1),
            )
            .await?;
            return value.with_owned_sequence(|sequence| {
                let mut values = sequence.into_values();
                pad_brace_values(&mut values, pad_to_outputs);
                RootedValues::new(values).map_err(|error| error.into_runtime_error())
            });
        }
        _ => {
            return Err(semantic_error(
                "CellExpansionOnNonCell",
                "Cell expansion on non-cell",
            ));
        }
    };
    pad_brace_values(&mut values, pad_to_outputs);
    RootedValues::new(values).map_err(|error| error.into_runtime_error())
}

fn pad_brace_values(values: &mut Vec<Value>, pad_to_outputs: Option<usize>) {
    if let Some(output_count) = pad_to_outputs {
        if values.len() > output_count {
            values.truncate(output_count);
        } else {
            values.resize(output_count, Value::Num(0.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::{CellArray, HandleRef, Value, ValueSequence};

    use super::{expand_arguments, MaterializedArgument, MaterializedExpansionSource};

    fn runtime() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
    }

    #[test]
    fn indexed_expansion_still_rejects_zero_selectors_and_noncell_values() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for (base, indices) in [
            (Value::Cell(CellArray::new(vec![], 0, 0).unwrap()), vec![]),
            (Value::Num(1.0), vec![Value::Num(1.0); 3]),
        ] {
            let error = block_on(expand_arguments(
                &runtime(),
                vec![MaterializedArgument::Expansion(
                    MaterializedExpansionSource::CellContents {
                        base,
                        indices,
                        expand_all: false,
                    },
                )],
            ))
            .expect_err("invalid expansion remains rejected");
            assert_eq!(error.identifier(), Some("RunMat:InvalidExpandTarget"));
        }
    }

    #[test]
    fn expansion_preserves_source_and_comma_list_order() {
        let cell = CellArray::new(vec![Value::Num(2.0), Value::Num(3.0)], 1, 2).unwrap();
        let values = block_on(expand_arguments(
            &runtime(),
            vec![
                MaterializedArgument::Single(Value::Num(1.0)),
                MaterializedArgument::Expansion(MaterializedExpansionSource::CellContents {
                    base: Value::Cell(cell),
                    indices: Vec::new(),
                    expand_all: true,
                }),
                MaterializedArgument::Single(Value::Num(4.0)),
            ],
        ))
        .expect("expand arguments");
        assert_eq!(
            values,
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ]
        );
    }

    #[test]
    fn returned_output_sequence_expands_without_becoming_a_value() {
        let values = block_on(expand_arguments(
            &runtime(),
            vec![MaterializedArgument::Expansion(
                MaterializedExpansionSource::ReturnedOutputs(
                    ValueSequence::comma_separated(vec![Value::Num(9.0), Value::Num(2.0)]).unwrap(),
                ),
            )],
        ))
        .expect("expand output list");
        assert_eq!(values, vec![Value::Num(9.0), Value::Num(2.0)]);
    }

    #[test]
    fn legacy_builtin_adapter_rejects_nested_output_lists() {
        let error =
            super::adapt_legacy_builtin_result(Value::OutputList(vec![Value::OutputList(vec![
                Value::Num(1.0),
            ])]))
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NestedLegacyOutputList"));
    }

    #[test]
    fn legacy_builtin_projector_preserves_an_empty_comma_separated_output_list() {
        let sequence = ValueSequence::comma_separated(Vec::new()).expect("empty output sequence");
        let projected = super::project_legacy_builtin_value_abi(sequence);
        assert!(matches!(projected, Value::OutputList(values) if values.is_empty()));
    }

    #[test]
    fn rooted_legacy_projector_preserves_zero_single_and_multiple_managed_outputs() {
        runmat_gc::gc_test_context(|| {
            for count in 0..=2 {
                let mut initial_roots = Vec::new();
                let mut handles = Vec::new();
                let mut values = Vec::new();
                for index in 0..count {
                    let initial = runmat_gc::gc_allocate_rooted(Value::String(
                        format!("projected output {index}").into(),
                    ))
                    .expect("allocate managed output");
                    let handle = initial.handle();
                    values.push(Value::HandleObject(HandleRef {
                        class_name: "ProjectedOutput".into(),
                        target: handle,
                        valid: true,
                    }));
                    handles.push(handle);
                    initial_roots.push(initial);
                }
                let sequence = if count == 1 {
                    ValueSequence::single(values.remove(0)).expect("single output")
                } else {
                    ValueSequence::comma_separated(values).expect("comma-separated outputs")
                };
                let source =
                    crate::execution::RootedValueSequence::new(sequence).expect("source roots");
                for initial in initial_roots {
                    initial.unroot().expect("transfer source roots");
                }
                let projected = super::project_rooted_legacy_builtin_value_abi(source)
                    .expect("project under the original roots");
                runmat_gc::gc_collect_major().expect("collect with projected owner");
                for handle in &handles {
                    assert!(runmat_gc::gc_clone_value(handle).is_ok());
                }
                match count {
                    1 => assert!(matches!(projected.value(), Value::HandleObject(_))),
                    0 | 2 => assert!(matches!(
                        projected.value(),
                        Value::OutputList(outputs) if outputs.len() == count
                    )),
                    _ => unreachable!(),
                }
                drop(projected);
                runmat_gc::gc_collect_major().expect("collect after projected owner");
                for handle in &handles {
                    assert!(runmat_gc::gc_clone_value(handle).is_err());
                }
            }
        });
    }

    #[test]
    fn invalid_expansion_retains_stable_identifier() {
        let error = block_on(expand_arguments(
            &runtime(),
            vec![MaterializedArgument::Expansion(
                MaterializedExpansionSource::CellContents {
                    base: Value::Num(1.0),
                    indices: Vec::new(),
                    expand_all: true,
                },
            )],
        ))
        .expect_err("numeric expansion must fail");
        assert_eq!(error.identifier(), Some("RunMat:InvalidExpandAllTarget"));
    }
}
