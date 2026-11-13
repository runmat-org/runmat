//! Selected native source-host preparation. Physical downloads/outputs reuse
//! their existing owners; native occurrences retain their own authority.
use super::{prepare_selected_host_inputs, NativeHostAdmission, RuntimeContext};
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_runtime::numeric_region::service::{
    root_values, NativeNumericRegionHostExecution, NativeNumericRegionProgramOwner,
    NativeNumericRegionRight,
};
use runmat_value::Value;

fn invalid() -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Placement:InvalidNativeRegion",
        "selected native host input/site/ownership no longer matches its admitted program",
    )
}
fn phase(error: runmat_runtime::RuntimeError, after_effect: bool) -> ProviderExecutionError {
    let error = ProviderExecutionError::from_source(error);
    if after_effect {
        error.after_effect()
    } else {
        error
    }
}
fn retain_numeric(value: &Value) -> Option<Value> {
    match value {
        Value::Num(value) => Some(Value::Num(*value)),
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
            ) =>
        {
            Some(Value::Tensor(tensor.clone()))
        }
        Value::GpuTensor(handle) => Some(Value::GpuTensor(handle.clone())),
        _ => None,
    }
}

pub(super) async fn prepare_native_host(
    runtime: &RuntimeContext,
    program: &NativeNumericRegionProgramOwner,
    right: &NativeNumericRegionRight,
    inputs: &[Value],
    mut captured: NativeHostAdmission,
) -> ProviderExecutionResult<NativeNumericRegionHostExecution> {
    if !right.is_current(runtime) || !program.has_native_operation_sites() {
        return Err(phase(invalid(), false));
    }
    program
        .checkpoint(runtime)
        .map_err(ProviderExecutionError::from_source)?;
    let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|source| {
        phase(
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Placement:NativeHostRegion",
                "native host policy changed",
                source,
            ),
            false,
        )
    })?;
    let outputs = if policy == HostAllocationPolicy::Managed {
        super::super::elementwise::host_readback::prepare_outputs(
            runtime,
            &captured.backing.get().admission.outputs,
        )
        .map_err(|source| {
            phase(
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Placement:NativeHostRegion",
                    "native numerical output preparation failed",
                    source,
                ),
                false,
            )
        })?
    } else {
        None
    };
    let operations = program
        .native_host_operations(&captured.backing.get().admission.canonical)
        .ok_or_else(|| phase(invalid(), false))?;
    let route = runmat_runtime::context::NativeRegionHostRoute::prepare(
        runtime,
        right.selection_authority(),
        program.native_operation_count(),
        operations,
        outputs,
    )
    .map_err(ProviderExecutionError::from_source)?;
    // Admit/allocate the exact private argument header before any readback.
    // The placeholders and their replacements are all closed numeric values:
    // no new managed GC identity can appear in this unique transformation.
    let private = root_values(runtime, program.input_count(), |_| Ok(Value::Num(0.0)))
        .map_err(ProviderExecutionError::from_source)?;
    private
        .with_owned_output_storage_async(|mut storage| async move {
            let plan = &mut captured.backing.get_mut_unique().admission.plan;
            let had_effects = !plan.sources.is_empty();
            let prepared = if had_effects {
                let empty = super::super::host_readback::CpuReadbackPlan {
                    sources: Vec::new(),
                    stack_len: 0,
                    resident_offsets: Vec::new(),
                };
                let plan = std::mem::replace(plan, empty);
                Some(
                    prepare_selected_host_inputs(plan, inputs, Some(runtime))
                        .await
                        .map_err(|failure| match failure {
                            super::CpuReadbackFailure::BeforeEffect(error) => phase(error, false),
                            super::CpuReadbackFailure::AfterEffect(error) => phase(error, true),
                        })?,
                )
            } else {
                None
            };
            if !right.is_current(runtime) {
                return Err(phase(invalid(), had_effects));
            }
            let slots = storage
                .try_as_mut_slice_unique()
                .ok_or_else(|| phase(invalid(), had_effects))?;
            let mut mirrors = [None; runmat_runtime::numeric_region::MAX_NODES];
            if let Some(prepared) = &prepared {
                for (offset, value) in &prepared.entry {
                    let mirror = mirrors
                        .get_mut(*offset)
                        .ok_or_else(|| phase(invalid(), had_effects))?;
                    if mirror.is_some() {
                        return Err(phase(invalid(), had_effects));
                    }
                    *mirror = Some(value);
                }
            }
            let mut filled = [false; runmat_runtime::numeric_region::MAX_NODES];
            let mut index = 0;
            for operation in &program.program().nodes {
                let original = match operation {
                    runmat_runtime::numeric_region::NumericRegionNode::Input(original) => {
                        Some(*original)
                    }
                    runmat_runtime::numeric_region::NumericRegionNode::Constant(_) => None,
                    _ => continue,
                };
                if let Some(original) = original {
                    let actual = inputs
                        .get(index)
                        .ok_or_else(|| phase(invalid(), had_effects))?;
                    let value = match actual {
                        Value::GpuTensor(_) => {
                            let value = mirrors
                                .get(index)
                                .copied()
                                .flatten()
                                .ok_or_else(|| phase(invalid(), had_effects))?;
                            if !(matches!(value, Value::Num(_)) || matches!(value, Value::Tensor(tensor)
                                if matches!(tensor.numeric_dtype(), runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64))) {
                                return Err(phase(invalid(), had_effects));
                            }
                            value
                        }
                        _ => actual,
                    };
                    let slot = slots
                        .get_mut(original)
                        .ok_or_else(|| phase(invalid(), had_effects))?;
                    *slot = retain_numeric(value).ok_or_else(|| phase(invalid(), had_effects))?;
                    *filled
                        .get_mut(original)
                        .ok_or_else(|| phase(invalid(), had_effects))? = true;
                }
                index += 1;
            }
            if index != inputs.len() || filled[..slots.len()].iter().any(|filled| !filled) {
                return Err(phase(invalid(), had_effects));
            }
            // These numeric values have no managed GC children. Canonical
            // publication retains their physical exact collection and claims.
            let private = runmat_runtime::execution::RootedValues::from_output_storage_in_context(
                storage, runtime,
            )
            .map_err(|error| phase(error, had_effects))?;
            drop(prepared);
            drop(captured);
            Ok(NativeNumericRegionHostExecution::new(private, route))
        })
        .await
}
