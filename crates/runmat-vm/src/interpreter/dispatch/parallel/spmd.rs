use crate::bytecode::program::ExecutionContext;
use crate::{InterpreterOutcome, InterpreterResumeState};
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

use super::{capability_error, distributed, execution_error, tasks};

pub(super) async fn execute(
    bytecode: &crate::Bytecode,
    executable: &crate::BytecodeSpmdRegion,
    header: crate::BytecodeSpmdHeader,
    operands: Vec<Value>,
    vars: &mut [Value],
    execution: &ExecutionContext,
    current_function_name: &str,
) -> Result<(), RuntimeError> {
    let (pool, labs) = spmd_request(header, operands, execution)?;
    let available_labs = runmat_types::LabCount(
        execution
            .runtime
            .execution()
            .current_pool()
            .map_err(execution_error)?
            .filter(|snapshot| snapshot.handle == pool)
            .map(|snapshot| snapshot.workers)
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdPoolUnavailable",
                    "SPMD requires a live pool in the current execution scope",
                )
            })?,
    );
    let captures = executable
        .captures
        .iter()
        .map(|capture| {
            vars.get(capture.slot).cloned().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdFrame",
                    "SPMD capture is outside its compiler-bound VM frame",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let spmd = execution
        .runtime
        .service_ports()
        .require_spmd("SPMD execution")
        .map_err(capability_error)?
        .clone();
    let admission = spmd
        .admit(
            runmat_execution::GangRequest { pool, labs },
            available_labs,
            executable.contract.id,
        )
        .await?;
    admission.validate()?;
    let gang = admission.gang.handle.clone();
    let execution_result = async {
        let rank_results = match execution.runtime.execution().spmd_execution_mode() {
            runmat_runtime::execution::SpmdExecutionMode::IsolatedWorkers => {
                runmat_runtime::execution::validate_spawn_captures(&captures)?;
                let program = crate::encode_interpreter_script_v2(bytecode).map_err(|error| {
                    crate::interpreter::errors::mex(
                        "ExecutionProgram",
                        &format!("failed to encode the exact SPMD program: {error}"),
                    )
                })?;
                execution
                    .runtime
                    .execution()
                    .execute_spmd_gang(runmat_runtime::execution::SpmdGangCall {
                        gang: gang.clone(),
                        region: executable.contract.id,
                        captures: captures.clone(),
                        requested_outputs: executable.outputs.len(),
                        program_revision: execution.runtime.program_revision().cloned(),
                        capabilities: executable.contract.capabilities.clone(),
                        program,
                    })
                    .await
                    .map_err(execution_error)?
            }
            runmat_runtime::execution::SpmdExecutionMode::Cooperative => {
                execute_spmd_in_process(
                    bytecode,
                    executable,
                    captures,
                    admission.labs,
                    execution,
                    current_function_name,
                )
                .await?
            }
        };
        validate_spmd_rank_results(&gang, executable.outputs.len(), &rank_results)?;
        let outputs = executable
            .outputs
            .iter()
            .enumerate()
            .map(|(output_index, output)| {
                let entries = rank_results
                    .iter()
                    .map(|rank| rank.outputs[output_index].clone())
                    .collect();
                Ok(runmat_runtime::context::RuntimeSpmdOutput {
                    value: output.contract.value,
                    fact: output.contract.fact.clone(),
                    entries,
                })
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        let handles = spmd
            .retain_outputs(gang.clone(), executable.contract.id, outputs)
            .await?;
        if handles.len() != executable.outputs.len() {
            return Err(crate::interpreter::errors::mex(
                "SpmdOutputContract",
                "SPMD runtime returned a different number of outputs than the compiler contract",
            ));
        }
        for (output, retained) in executable.outputs.iter().zip(handles) {
            let destination = vars.get_mut(output.slot).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdFrame",
                    "SPMD output is outside its compiler-bound VM frame",
                )
            })?;
            *destination = match retained {
                runmat_runtime::context::RuntimeSpmdRetainedOutput::Composite(handle) => {
                    Value::Composite(Box::new(handle))
                }
                runmat_runtime::context::RuntimeSpmdRetainedOutput::Distributed(handle) => {
                    Value::Distributed(Box::new(handle))
                }
            };
            crate::runtime::workspace::mark_workspace_assigned(output.slot);
        }
        Ok(())
    }
    .await;
    let retirement = spmd.retire(gang).await;
    match execution_result {
        Err(error) => Err(error),
        Ok(()) => retirement,
    }
}

async fn execute_spmd_in_process(
    bytecode: &crate::Bytecode,
    executable: &crate::BytecodeSpmdRegion,
    captures: Vec<Value>,
    labs: Vec<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    execution: &ExecutionContext,
    current_function_name: &str,
) -> Result<Vec<runmat_runtime::execution::SpmdRankResult>, RuntimeError> {
    let mut tasks = Vec::with_capacity(labs.len());
    for collective in labs {
        let rank = collective.context().rank;
        let gang = collective.context().gang.clone();
        let mut frame = vec![None; bytecode.var_count];
        for (capture, value) in executable.captures.iter().zip(&captures) {
            frame[capture.slot] = Some(value.clone());
        }
        let services = execution
            .runtime
            .service_ports()
            .clone()
            .with_collective(collective);
        let runtime = execution.runtime.fork_parallel_lab(services);
        let spmd = execution
            .runtime
            .service_ports()
            .require_spmd("SPMD execution")
            .map_err(capability_error)?
            .clone();
        let bytecode = bytecode.clone();
        let function_name = current_function_name.to_string();
        let body_pc = executable.body.pc;
        tasks.push(async move {
            let resume = InterpreterResumeState {
                pc: body_pc,
                completion_boundary: Some(
                    crate::interpreter::state::InterpreterCompletionBoundary::before(
                        executable.exit.pc,
                    ),
                ),
                vars: frame,
                supplied_inputs: 0,
                requested_outputs: 0,
                missing_input_slots: HashSet::new(),
                global_aliases: HashMap::new(),
                persistent_aliases: HashMap::new(),
                side_effect_epoch: 0,
            };
            let result = crate::interpreter::runner::interpret_resume_in_context(
                &bytecode,
                resume,
                Some(&function_name),
                runtime.clone(),
            )
            .await;
            match result {
                Ok(InterpreterOutcome::Completed(completion)) => {
                    spmd.rank_finished(&gang, rank)?;
                    Ok((rank, completion, runtime))
                }
                Err(error) => {
                    // Wake peers that may be waiting in a collective, but keep
                    // the rank's source-mapped failure as the public result.
                    let _ = spmd.rank_failed(&gang, rank);
                    Err(error)
                }
            }
        });
    }
    let completions = futures::future::try_join_all(tasks).await?;
    let mut results = Vec::with_capacity(completions.len());
    for (rank, completion, runtime) in completions {
        let mut outputs = Vec::with_capacity(executable.outputs.len());
        for output in &executable.outputs {
            if !completion.assigned_slots.contains(&output.slot) {
                outputs.push(None);
                continue;
            }
            let value = completion.values.get(output.slot).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "SpmdFrame",
                    "an assigned SPMD output is outside its completed VM frame",
                )
            })?;
            outputs.push(Some(
                distributed::encode_spmd_output(&runtime, value).await?,
            ));
        }
        results.push(runmat_runtime::execution::SpmdRankResult { rank, outputs });
    }
    Ok(results)
}

fn validate_spmd_rank_results(
    gang: &runmat_execution::GangHandle,
    output_count: usize,
    ranks: &[runmat_runtime::execution::SpmdRankResult],
) -> Result<(), RuntimeError> {
    if ranks.len() != gang.labs.0 as usize {
        return Err(crate::interpreter::errors::mex(
            "SpmdOutputContract",
            "SPMD execution returned a different number of ranks than the admitted gang",
        ));
    }
    for (index, result) in ranks.iter().enumerate() {
        let expected_rank = runmat_types::LabRank(u32::try_from(index + 1).map_err(|_| {
            crate::interpreter::errors::mex(
                "SpmdOutputContract",
                "SPMD rank index exceeds the supported range",
            )
        })?);
        if result.rank != expected_rank || result.outputs.len() != output_count {
            return Err(crate::interpreter::errors::mex(
                "SpmdOutputContract",
                "SPMD rank results differ from the admitted rank or compiler output contract",
            ));
        }
    }
    Ok(())
}

fn spmd_request(
    header: crate::BytecodeSpmdHeader,
    mut operands: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<
    (
        runmat_execution::PoolHandle,
        runmat_types::SpmdLabRequirement,
    ),
    RuntimeError,
> {
    use runmat_types::{LabCount, SpmdLabRequirement};

    let explicit_pool = matches!(header, crate::BytecodeSpmdHeader::PoolRange);
    let pool = if explicit_pool {
        tasks::pool_handle(operands.remove(0))?
    } else {
        execution
            .runtime
            .execution()
            .ensure_pool(runmat_execution::PoolRequest::automatic())
            .map_err(execution_error)?
            .handle
    };
    let labs = match header {
        crate::BytecodeSpmdHeader::Default => SpmdLabRequirement::Default,
        crate::BytecodeSpmdHeader::Exact => SpmdLabRequirement::Exact {
            labs: LabCount(spmd_lab_count(operands.remove(0))?),
        },
        crate::BytecodeSpmdHeader::Range | crate::BytecodeSpmdHeader::PoolRange => {
            let minimum = spmd_lab_count(operands.remove(0))?;
            let maximum = spmd_lab_count(operands.remove(0))?;
            if minimum > maximum {
                return Err(crate::interpreter::errors::mex(
                    "InvalidSpmdRange",
                    "SPMD minimum lab count cannot exceed its maximum",
                ));
            }
            SpmdLabRequirement::Range {
                minimum: LabCount(minimum),
                maximum: LabCount(maximum),
            }
        }
    };
    Ok((pool, labs))
}

fn spmd_lab_count(value: Value) -> Result<u32, RuntimeError> {
    let count = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 1.0
                && value <= u32::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    };
    count
        .filter(|count| *count > 0)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "InvalidSpmdLabCount",
                "SPMD lab counts must be positive integer scalars",
            )
        })
}
