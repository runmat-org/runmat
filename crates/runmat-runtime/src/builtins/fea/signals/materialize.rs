use runmat_time::Instant;
use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, GeneratorCallable, PhysicalSignal};

use super::bindings::{FrozenGeneratorInventory, FrozenStudyGeneratorInventory};
use super::*;

pub(in crate::builtins::fea) async fn materialize_study(
    study: &mut crate::analysis::AnalysisStudySpec,
    bindings: &FrozenGeneratorInventory,
) -> BuiltinResult<()> {
    let wall_millis = materialization_wall_millis(&study.model);
    let model = &mut study.model;
    model.signals = materialize_signals(&model.signals, bindings, wall_millis).await?;
    Ok(())
}

pub(in crate::builtins::fea) async fn materialize_sweep(
    sweep: &mut crate::analysis::AnalysisStudySweepSpec,
    bindings: &FrozenStudyGeneratorInventory,
) -> BuiltinResult<()> {
    for study in &mut sweep.studies {
        let study_id = study.study_id.clone();
        let study_bindings = bindings.for_study(&study_id);
        materialize_study(study, &study_bindings).await?;
    }
    Ok(())
}

async fn materialize_signals(
    signals: &[PhysicalSignal],
    bindings: &FrozenGeneratorInventory,
    maximum_wall_millis: u64,
) -> BuiltinResult<Vec<PhysicalSignal>> {
    if signals.iter().all(|signal| signal.generator().is_none()) {
        return Ok(signals.to_vec());
    }
    let by_id = bindings
        .bindings()
        .iter()
        .map(|binding| (&binding.signal_id, binding))
        .collect::<BTreeMap<_, _>>();
    let context = crate::context::legacy::active().ok_or_else(|| {
        materialization_error("signal materialization requires an active RunMat runtime context")
    })?;
    let program_service = context
        .service_ports()
        .require_program("freeze an FEA signal callable")
        .map_err(|error| materialization_error(error.to_string()))?;
    let analysis_service = context
        .service_ports()
        .require_analysis("materialize an FEA signal")
        .map_err(|error| materialization_error(error.to_string()))?;
    let started = Instant::now();
    let mut materialized = Vec::with_capacity(signals.len());
    for signal in signals {
        let Some(generator) = signal.generator() else {
            materialized.push(signal.clone());
            continue;
        };
        let binding = by_id.get(signal.signal_id()).ok_or_else(|| {
            materialization_error(format!(
                "generated signal `{}` has no frozen runtime callable; named .fea generators must be run through the FEA command host",
                signal.signal_id().0
            ))
        })?;
        if crate::interrupt::is_cancelled() {
            return Err(materialization_error(
                "signal materialization was cancelled",
            ));
        }
        if elapsed_millis(started) >= maximum_wall_millis {
            return Err(materialization_error(format!(
                "signal materialization exceeded its {maximum_wall_millis} ms wall-time limit"
            )));
        }
        let (function, display_name) = match &generator.invocation.callable {
            GeneratorCallable::BoundFunction {
                function,
                display_name,
            } => (*function, display_name.as_str()),
            GeneratorCallable::ProjectFunction { .. } | GeneratorCallable::ImportedFunction(_) => {
                return Err(materialization_error(
                    "named .fea generators must be run through the FEA command host",
                ))
            }
        };
        let runmat_execution::value::CallableOwner::Bound(origin) = binding.callable.owner else {
            return Err(materialization_error(
                "signal callable has no authenticated bound origin",
            ));
        };
        if origin.function != function {
            return Err(materialization_error(
                "signal generator function differs from its authenticated callable origin",
            ));
        }
        let program = program_service
            .freeze_bound_callable(origin, display_name)
            .map_err(|error| materialization_error(error.to_string()))?;
        let request = crate::context::RuntimeSignalMaterializationRequest {
            signal: signal.clone(),
            callable: binding.callable.clone(),
            program,
            maximum_wall_millis: maximum_wall_millis
                .saturating_sub(elapsed_millis(started))
                .max(1),
        };
        materialized.push(
            analysis_service
                .materialize_signal(request)
                .await
                .map_err(|error| materialization_error(error.to_string()))?,
        );
    }
    Ok(materialized)
}

fn materialization_wall_millis(model: &AnalysisModel) -> u64 {
    model
        .sequence
        .steps
        .iter()
        .map(|step| step.accuracy.resources.maximum_wall_time_milliseconds)
        .max()
        .unwrap_or(60_000)
        .max(1)
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn materialization_error(message: impl Into<String>) -> crate::RuntimeError {
    builtin_error(RUN_NAME, &ERROR_OPERATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precomputed_signals_need_no_generator_host() {
        let bindings = FrozenGeneratorInventory::default();
        let empty = futures::executor::block_on(materialize_signals(&[], &bindings, 1));
        assert!(empty.unwrap().is_empty());
        let signal = constant_signal(
            "constant-force".into(),
            SignalShape::Scalar,
            SignalValue::Scalar(Force::new(1.0)),
            PhysicalSignal::Force,
        );
        let signals = vec![signal];
        let materialized = futures::executor::block_on(materialize_signals(&signals, &bindings, 1));
        assert_eq!(materialized.unwrap(), signals);
    }
}
