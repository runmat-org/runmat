//! Callback state is a rooted view of the completed generation. Numeric
//! population, score and history payloads retain their existing backing.
use super::super::{
    assessment::Assessment, evaluation::generation::Evaluated, history::History, inputs::Problem,
    invalid, lineage::Lineage, population::Generation, ScalarCooperation, POLICY,
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{
    OptimGeneticAction, OptimGeneticObserverState as Phase, OptimGeneticStateField as Field,
};
use runmat_value::{CellArray, IntValue, Tensor, Value};
use std::time::Duration;

pub(super) struct Flags {
    pub eval_elites: bool,
    pub duplicates: bool,
}
pub(super) enum How<'a> {
    Initial { supplied: usize },
    Evolution(&'a [Lineage]),
}
pub(super) struct Snapshot<'a> {
    pub population: &'a Generation,
    pub evaluated: &'a Evaluated,
    pub assessment: &'a Assessment,
    pub problem: &'a Problem,
    pub history: &'a History,
    pub generation: usize,
    pub calls: usize,
    pub start_token: &'a Value,
    pub last_improvement: usize,
    pub last_improvement_time: Duration,
    pub flags: &'a Flags,
    pub expectation: Option<&'a Tensor>,
    pub selection: &'a [usize],
    pub how: How<'a>,
    pub augmented: bool,
    pub phase: Phase,
    pub inner_generation: usize,
    pub total_inner_generations: usize,
    pub previous: Option<&'a RootedValue>,
    pub stop_flag: Option<&'a Value>,
    pub action: Option<OptimGeneticAction>,
    pub ancestors: &'a [Option<super::super::lineage::Parents>],
}
pub(super) fn start_token(
    timing: &super::super::common::solver_timing::SolverTiming,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Value> {
    let runtime = cooperation.runtime().ok_or_else(|| {
        let mut error = crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::Builtin,
            "genetic state requires its selected session",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    // Reuse the session's existing tic/toc owner and the SAME admitted start;
    // there is no second timer, physical-clock fallback or fabricated token.
    let token = runtime
        .state()
        .timing
        .borrow_mut()
        .start(timing.started(), true)
        .ok_or_else(|| {
            let error = &runmat_builtins::TIC_ERROR_TOKEN_EXHAUSTED;
            let mut builder = crate::build_runtime_error(error.message)
                .with_builtin("ga")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
            if let Some(identifier) = error.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        })?;
    Ok(Value::Int(IntValue::U64(token.raw())))
}
pub(super) fn count(value: usize) -> BuiltinResult<Value> {
    u64::try_from(value)
        .ok()
        .and_then(|value| IntValue::U64(value).to_exact_f64())
        .map(Value::Num)
        .ok_or_else(|| invalid("public genetic counter cannot be represented exactly as double"))
}
pub(super) async fn publish(
    snapshot: Snapshot<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    use super::super::common::population::projection::Arguments;
    let rows = snapshot.population.members();
    let mut result = if let Some(previous) = snapshot.previous {
        let Value::Struct(previous) = previous.value() else {
            return Err(invalid("observer state must remain a scalar structure"));
        };
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        for (name, value) in &previous.fields {
            let unavailable = name == Field::How.name()
                || name == Field::NonlinearInequality.name()
                || name == Field::NonlinearEquality.name()
                || snapshot.augmented && name == Field::Fitness.name();
            if !unavailable {
                fields.insert(name, value.clone());
            }
        }
        fields
    } else {
        runmat_value::UnboundedStructValueBuilder::new()
    };
    result.insert(Field::Generation.name(), count(snapshot.generation)?);
    result.insert(Field::FunEval.name(), count(snapshot.calls)?);
    result.insert(Field::StartTime.name(), snapshot.start_token.clone());
    result.insert(
        Field::StopFlag.name(),
        snapshot
            .stop_flag
            .cloned()
            .unwrap_or_else(|| Value::from("")),
    );
    result.insert(
        Field::InnerGeneration.name(),
        count(snapshot.inner_generation)?,
    );
    result.insert(
        Field::TotalInnerGenerations.name(),
        count(snapshot.total_inner_generations)?,
    );
    result.insert(
        Field::LastImprovement.name(),
        count(snapshot.last_improvement)?,
    );
    result.insert(
        Field::LastImprovementTime.name(),
        Value::Num(snapshot.last_improvement_time.as_secs_f64()),
    );
    result.insert(Field::Best.name(), snapshot.history.value());
    if let Some(action) = snapshot.action {
        result.insert(Field::How.name(), Value::from(action.name()));
    }
    result.insert(
        Field::EvalElites.name(),
        Value::Bool(snapshot.flags.eval_elites),
    );
    result.insert(
        Field::HaveDuplicates.name(),
        Value::Bool(snapshot.flags.duplicates),
    );
    result.insert(Field::Population.name(), snapshot.population.value());
    result.insert(
        Field::Score.name(),
        snapshot.assessment.public_scores.clone(),
    );
    result.insert(
        Field::Expectation.name(),
        snapshot
            .expectation
            .map(|value| Value::Tensor(value.clone()))
            .unwrap_or_else(super::super::inputs::raw::empty),
    );
    result.insert(
        Field::LinearConstraintType.name(),
        Value::from(if snapshot.problem.has_linear {
            "linearconstraints"
        } else if snapshot.problem.has_bounds {
            "boundconstraints"
        } else {
            "unconstrained"
        }),
    );
    result.insert(
        Field::MixedInteger.name(),
        Value::Bool(snapshot.problem.mixed_integer),
    );
    let mut selection = POLICY.capacity(snapshot.selection.len(), cooperation.runtime())?;
    for &index in snapshot.selection {
        if index >= rows {
            return Err(invalid("public selection contains an invalid parent row"));
        }
        let Value::Num(index) = count(
            index
                .checked_add(1)
                .ok_or_else(|| invalid("public parent index overflow"))?,
        )?
        else {
            unreachable!("double public count");
        };
        selection.push(index);
        cooperation.completed(1).await?;
    }
    result.insert(
        Field::Selection.name(),
        Value::Tensor(Tensor::new(selection, vec![1, snapshot.selection.len()]).map_err(invalid)?),
    );
    if let How::Evolution(lineage) = &snapshot.how {
        if lineage.len() != rows {
            return Err(invalid("public row lineage extent changed"));
        }
    }
    let mut how = POLICY.capacity(rows, cooperation.runtime())?;
    for row in 0..rows {
        let kind = match &snapshot.how {
            How::Initial { supplied } => {
                if *supplied > rows {
                    return Err(invalid("initial supplied prefix exceeds population"));
                }
                if row < *supplied {
                    Lineage::Supplied
                } else {
                    Lineage::Creation
                }
            }
            How::Evolution(lineage) => lineage[row],
        };
        how.push(Value::from(kind.name()));
        cooperation.completed(1).await?;
    }
    result.insert(
        Field::GeneticOrigin.name(),
        Value::Cell(CellArray::new_with_shape(how, vec![rows, 1]).map_err(invalid)?),
    );
    if !snapshot.ancestors.is_empty() && snapshot.ancestors.len() != rows {
        return Err(invalid("public genetic parentage extent changed"));
    }
    let mut ancestors = POLICY.capacity(
        rows.checked_mul(2)
            .ok_or_else(|| invalid("public parentage extent overflow"))?,
        cooperation.runtime(),
    )?;
    for column in 0..2 {
        for row in 0..rows {
            let parent = snapshot
                .ancestors
                .get(row)
                .copied()
                .flatten()
                .and_then(|parents| {
                    if column == 0 {
                        Some(parents.first)
                    } else {
                        parents.second
                    }
                });
            let value = if let Some(parent) = parent {
                if parent >= rows {
                    return Err(invalid(
                        "public genetic parent is outside its source generation",
                    ));
                }
                let Value::Num(parent) = count(
                    parent
                        .checked_add(1)
                        .ok_or_else(|| invalid("parentage index overflow"))?,
                )?
                else {
                    unreachable!("double public index");
                };
                parent
            } else {
                f64::NAN
            };
            ancestors.push(value);
            cooperation.completed(1).await?;
        }
    }
    result.insert(
        Field::GeneticParents.name(),
        Value::Tensor(Tensor::new(ancestors, vec![rows, 2]).map_err(invalid)?),
    );
    if snapshot.phase != Phase::Interrupt {
        result.insert(
            Field::NonlinearInequality.name(),
            Value::Tensor(snapshot.evaluated.constraints.inequalities.clone()),
        );
        result.insert(
            Field::NonlinearEquality.name(),
            Value::Tensor(snapshot.evaluated.constraints.equalities.clone()),
        );
    }
    if !snapshot.augmented {
        result.insert(
            Field::Fitness.name(),
            snapshot.evaluated.fitness.values.value(),
        );
    }
    RootedValue::in_optional_context(Value::Struct(result.finish()), cooperation.runtime())
}

/// Retire only this controller's old Best edge before numeric append. Other
/// user-added state fields remain rooted; their aliases still require COW.
pub(super) async fn retire_history_edge(
    state: RootedValue,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = cooperation.runtime().cloned();
    state
        .with_owned_async(|value| async move {
            let Value::Struct(state) = value else {
                return Err(invalid("observer state must remain a scalar structure"));
            };
            crate::execution::construct_struct_from_fields_in_optional_context(
                runtime.as_ref(),
                state.fields.iter().filter_map(|(name, value)| {
                    (name != Field::Best.name()).then_some((name, value))
                }),
            )
        })
        .await
}
