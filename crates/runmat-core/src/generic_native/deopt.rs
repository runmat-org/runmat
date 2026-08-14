use std::rc::Rc;

use runmat_jit::{
    deopt::{DeoptimizationPolicy, MaterializedFrame, ResumeTarget},
    execute::{GenericExecution, GenericInvocationStep},
    GenericExecutor,
};
use runmat_value::Value;

use crate::ExecutableUnit;

struct DeoptimizationInvocation {
    executor: Rc<GenericExecutor>,
    function: runmat_types::ProgramFunctionId,
    captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    policy: DeoptimizationPolicy,
}

pub(super) async fn invoke(
    unit: &ExecutableUnit,
    executor: Rc<GenericExecutor>,
    function: runmat_types::ProgramFunctionId,
    captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<GenericExecution, runmat_runtime::RuntimeError> {
    let policy = DeoptimizationPolicy {
        target: ResumeTarget::Interpreter,
        ..DeoptimizationPolicy::default()
    };
    invoke_with_policy(
        unit,
        DeoptimizationInvocation {
            executor,
            function,
            captures,
            arguments,
            requested_outputs,
            runtime,
            policy,
        },
    )
    .await
}

async fn invoke_with_policy(
    unit: &ExecutableUnit,
    invocation: DeoptimizationInvocation,
) -> Result<GenericExecution, runmat_runtime::RuntimeError> {
    let DeoptimizationInvocation {
        executor,
        function,
        captures,
        arguments,
        requested_outputs,
        runtime,
        policy,
    } = invocation;
    let mut invocation = executor
        .begin_with_deoptimization(
            function,
            captures,
            arguments,
            requested_outputs,
            runtime.clone(),
            policy,
        )
        .map_err(super::error::from_jit_error)?;
    loop {
        match invocation.advance().map_err(super::error::from_jit_error)? {
            GenericInvocationStep::Completed(execution) => return Ok(execution),
            GenericInvocationStep::Suspended {
                continuation,
                generation,
            } => invocation
                .resume_suspension(continuation, generation)
                .await
                .map_err(super::error::from_jit_error)?,
            GenericInvocationStep::Deoptimized { target, .. }
                if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
            {
                invocation
                    .resume_deoptimization()
                    .map_err(super::error::from_jit_error)?;
            }
            GenericInvocationStep::Deoptimized { frame, .. } => {
                return resume_interpreter(unit, frame, requested_outputs, runtime).await;
            }
        }
    }
}

async fn resume_interpreter(
    unit: &ExecutableUnit,
    frame: MaterializedFrame,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<GenericExecution, runmat_runtime::RuntimeError> {
    let function_id = runmat_hir::FunctionId(frame.function.0 as usize);
    let layout = unit
        .vm_layout()
        .functions
        .get(&function_id)
        .ok_or_else(|| super::error::stage("NativeDeoptimization", "VM layout is unavailable"))?;
    let (bytecode, function_name, var_count) =
        if let Some(function) = unit.functions().get(function_id) {
            (
                runmat_vm::Bytecode::for_function(
                    function,
                    unit.functions().clone(),
                    unit.vm_layout().clone(),
                ),
                function.display_name.clone(),
                function.var_count,
            )
        } else {
            (
                unit.bytecode().clone(),
                unit.mir()
                    .functions
                    .get(&function_id)
                    .map(|metadata| metadata.name.0.clone())
                    .unwrap_or_else(|| "<main>".into()),
                unit.bytecode().var_count,
            )
        };
    let pc = frame.site.bytecode_pc.ok_or_else(|| {
        super::error::stage(
            "NativeDeoptimization",
            "interpreter target has no verified bytecode resume point",
        )
    })?;
    let pc = usize::try_from(pc).map_err(|_| {
        super::error::stage(
            "NativeDeoptimization",
            "bytecode resume PC exceeds this host",
        )
    })?;
    let mut vars = vec![None; var_count];
    for local in frame.locals {
        let slot = layout
            .mir_local_slots
            .get(&runmat_mir::MirLocalId(local.local.0 as usize))
            .ok_or_else(|| {
                super::error::stage(
                    "NativeDeoptimization",
                    "native local has no canonical VM slot",
                )
            })?;
        vars[slot.0] = local.value;
    }
    // Native frame operands describe the MIR operation about to execute and
    // remain available to the generic-native continuation. The VM resumes at
    // the compiler-verified bytecode boundary before that operation, where its
    // own operand stack is empty, so it reconstructs those operands normally.
    let aliases = |bindings: std::collections::BTreeMap<usize, String>| {
        bindings
            .into_iter()
            .map(|(local, name)| {
                layout
                    .mir_local_slots
                    .get(&runmat_mir::MirLocalId(local))
                    .map(|slot| (slot.0, name))
                    .ok_or_else(|| {
                        super::error::stage(
                            "NativeDeoptimization",
                            "native storage alias has no canonical VM slot",
                        )
                    })
            })
            .collect::<Result<std::collections::HashMap<_, _>, _>>()
    };
    let missing_input_slots = frame
        .missing_input_locals
        .iter()
        .map(|local| {
            layout
                .mir_local_slots
                .get(&runmat_mir::MirLocalId(local.0 as usize))
                .map(|slot| slot.0)
                .ok_or_else(|| {
                    super::error::stage(
                        "NativeDeoptimization",
                        "omitted native input has no canonical VM slot",
                    )
                })
        })
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    let resume = runmat_vm::InterpreterResumeState {
        pc,
        vars,
        supplied_inputs: frame.supplied_inputs,
        requested_outputs: frame.requested_outputs,
        missing_input_slots,
        global_aliases: aliases(frame.global_bindings)?,
        persistent_aliases: aliases(frame.persistent_bindings)?,
        side_effect_epoch: frame.site.side_effect_epoch,
    };
    let runmat_vm::InterpreterOutcome::Completed(values) =
        runmat_vm::interpret_resume_in_context(&bytecode, resume, Some(&function_name), runtime)
            .await?;
    let body = unit.mir().bodies.get(&function_id).ok_or_else(|| {
        super::error::stage("NativeDeoptimization", "MIR function body is unavailable")
    })?;
    let fixed_outputs = body
        .abi
        .fixed_outputs
        .iter()
        .map(|local| {
            layout
                .binding_slots
                .get(local)
                .and_then(|slot| values.get(slot.0))
                .cloned()
                .unwrap_or(Value::Num(0.0))
        })
        .collect::<Vec<_>>();
    let varargout = body
        .abi
        .varargout
        .and_then(|binding| layout.binding_slots.get(&binding))
        .and_then(|slot| values.get(slot.0));
    let outputs = runmat_runtime::call::function_abi::collect_function_outputs(
        &function_name,
        &fixed_outputs,
        varargout,
        requested_outputs,
    )?;
    let captures = unit
        .mir()
        .functions
        .get(&function_id)
        .into_iter()
        .flat_map(|metadata| &metadata.captures)
        .filter_map(|capture| {
            let local = body
                .locals
                .iter()
                .find(|local| local.binding == Some(capture.binding))?;
            let slot = layout.mir_local_slots.get(&local.id)?;
            Some(runmat_runtime::call::lexical::LexicalCapture {
                binding: capture.binding,
                value: values.get(slot.0).cloned().unwrap_or(Value::Num(0.0)),
            })
        })
        .collect();
    Ok(GenericExecution { outputs, captures })
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_jit::deopt::{DeoptimizationPolicy, FaultInjection, ResumeTarget};
    use runmat_value::Value;

    use crate::{ExecutableSource, RunMatSession};

    #[test]
    fn core_resumes_native_execution_in_interpreter_with_exact_call_context() {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        session.configure_runtime_context();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "core-native-deopt-test@1",
                "nativeDeopt.m",
                "function [y, observed] = nativeDeopt(x, optional)\ny = sqrt(x);\nif nargin < 2\nobserved = nargout;\nelse\nobserved = optional;\nend\nend\n",
            ),
            None,
        ))
        .expect("compile deoptimization fixture");
        let compiled = super::super::compile::compile(&unit, Some("nativeDeopt"))
            .expect("compile generic-native fixture");
        let safepoint = compiled
            .safepoints
            .get(&compiled.entrypoint)
            .and_then(|points| points.first())
            .copied()
            .expect("sqrt call should publish a native safepoint");
        let policy = DeoptimizationPolicy {
            target: ResumeTarget::Interpreter,
            ..DeoptimizationPolicy::default()
        }
        .inject(FaultInjection::Safepoint(safepoint));
        let execution = block_on(super::invoke_with_policy(
            &unit,
            super::DeoptimizationInvocation {
                executor: compiled.executor,
                function: compiled.entrypoint,
                captures: Vec::new(),
                arguments: vec![Value::Num(81.0)],
                requested_outputs: 2,
                runtime: session.runtime_context().clone(),
                policy,
            },
        ))
        .expect("native execution should resume in the interpreter");
        assert_eq!(execution.outputs, vec![Value::Num(9.0), Value::Num(2.0)]);
    }
}
