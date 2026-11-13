//! Six requested outputs share their retained generations until the result
//! forest has acquired its own roots. No reporting objective call occurs.
use super::super::linear::Geometry;
use super::super::{evaluation::fitness::Origin, inputs::raw::empty, ScalarCooperation, POLICY};
use super::{driver::Finished, frame::count};
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) fn finalize(
    finished: Finished,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let requested = crate::output_count::current_output_count_in(cooperation.runtime());
    let outputs = requested.unwrap_or(1);
    let mut values = POLICY.capacity(outputs, cooperation.runtime())?;
    let search = finished.search.as_ref();
    let best = search.map(|search| &search.best);
    if outputs > 0 {
        values.push(
            best.and_then(|best| best.point.as_ref())
                .map_or_else(empty, |point| point.value().clone()),
        );
    }
    if outputs >= 2 {
        values.push(
            best.and_then(|best| best.score())
                .map_or_else(empty, Value::Num),
        );
    }
    if outputs >= 3 {
        values.push(Value::Num(finished.reason.flag() as f64));
    }
    if outputs >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "generations",
            count(search.map_or(0, |search| search.progress.generation))?,
        );
        fields.insert(
            "funccount",
            count(search.map_or(0, |search| search.objective_counts.points))?,
        );
        fields.insert(
            "objectiveCallCount",
            count(search.map_or(0, |search| search.objective_counts.callbacks))?,
        );
        fields.insert(
            "nonlinearPointCount",
            count(search.map_or_else(
                || {
                    finished
                        .initial
                        .as_ref()
                        .map_or(0, |initial| initial.nonlinear_points)
                },
                |search| search.nonlinear_counts.points,
            ))?,
        );
        fields.insert(
            "nonlinearCallCount",
            count(search.map_or_else(
                || {
                    finished
                        .initial
                        .as_ref()
                        .map_or(0, |initial| initial.nonlinear_calls)
                },
                |search| search.nonlinear_counts.callbacks,
            ))?,
        );
        fields.insert(
            "innergenerations",
            count(search.map_or(0, |search| search.progress.total_inner))?,
        );
        fields.insert("totaltime", Value::Num(finished.elapsed.as_secs_f64()));
        fields.insert(
            "linearnodes",
            count(
                search
                    .and_then(|search| search.domain.as_ref())
                    .map_or(0, |domain| domain.repair_nodes()),
            )?,
        );
        fields.insert(
            "maxconstraint",
            best.filter(|best| best.point.is_some())
                .map_or_else(empty, |best| Value::Num(best.violation)),
        );
        fields.insert("message", Value::from(finished.reason.message()));
        fields.insert("rngstate", finished.rng.value().clone());
        fields.insert(
            "hybridflag",
            finished
                .hybrid
                .as_ref()
                .map_or_else(empty, |hybrid| Value::Num(hybrid.flag)),
        );
        fields.insert(
            "hybridfunccount",
            count(
                finished
                    .hybrid
                    .as_ref()
                    .map_or(0, |hybrid| hybrid.evaluations),
            )?,
        );
        fields.insert(
            "hybridaccepted",
            Value::Bool(
                finished
                    .hybrid
                    .as_ref()
                    .is_some_and(|hybrid| hybrid.accepted),
            ),
        );
        fields.insert(
            "fitnessprovenance",
            best.and_then(|best| best.fitness)
                .map_or_else(empty, |fitness| {
                    Value::from(match fitness.origin {
                        Origin::Measured => "measured",
                        Origin::SuppliedInitial => "initial-score",
                        Origin::ObserverSupplied => "observer-score",
                        Origin::HybridReported => "hybrid-report",
                        Origin::ConstraintRejected => "constraint-rejected",
                    })
                }),
        );
        fields.insert(
            "problemtype",
            Value::from(search.map_or("unmeasured", |search| {
                if search.problem.mixed_integer {
                    "integerconstraints"
                } else if search.problem.nonlinear.is_some() {
                    "nonlinearconstraints"
                } else if search.problem.has_linear {
                    "linearconstraints"
                } else if search.problem.has_bounds {
                    "boundconstraints"
                } else {
                    "unconstrained"
                }
            })),
        );
        values.push(Value::Struct(fields.finish()));
    }
    if outputs >= 5 {
        values.push(search.map_or_else(empty, |search| search.generation.value()));
    }
    if outputs >= 6 {
        values.push(search.map_or_else(empty, |search| {
            if search.problem.mixed_integer {
                search.assessment.public_scores.clone()
            } else {
                search.evaluated.fitness.values.value()
            }
        }));
    }
    let sequence = if requested.is_none() {
        runmat_value::ValueSequence::single(values.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(values)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match cooperation.runtime() {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => crate::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error()),
    }
}
