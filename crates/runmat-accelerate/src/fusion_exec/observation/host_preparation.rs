//! Pre-effect ownership of selected VM Host outputs and private readbacks.
use super::{CpuReadbackFailure, CpuReadbackPlan, RuntimeContext, VmCpuCompletion};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_value::Value;

pub(super) async fn prepare_selected_host_inputs(
    plan: CpuReadbackPlan,
    stack: &[Value],
    runtime: Option<&RuntimeContext>,
) -> Result<crate::fusion_exec::host_readback::PreparedVmHostInputs, CpuReadbackFailure> {
    let Some(runtime) = runtime else {
        return plan.prepare_inputs(stack).await;
    };
    let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|cause| {
        CpuReadbackFailure::BeforeEffect(host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Placement:HostReadback",
            "selected host readback policy changed",
            cause,
        ))
    })?;
    if policy == HostAllocationPolicy::Managed {
        plan.prepare_inputs_managed(stack, runtime).await
    } else {
        plan.prepare_inputs(stack).await
    }
}

/// Keep the synchronous result small; allocate the rich failure only on refusal.
pub(super) fn prepare_vm_outputs(vm: &mut VmCpuCompletion) -> Result<(), Box<CpuReadbackFailure>> {
    let Some(specs) = vm.outputs.take() else {
        return Ok(());
    };
    let runtime = vm.runtime.as_ref().ok_or_else(|| {
        Box::new(CpuReadbackFailure::BeforeEffect(
            runmat_runtime::build_runtime_error(
                "selected VM host numerical outputs have no runtime owner",
            )
            .build(),
        ))
    })?;
    let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|cause| {
        Box::new(CpuReadbackFailure::BeforeEffect(
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Placement:HostNumericalOutput",
                "selected VM host numerical output policy changed",
                cause,
            ),
        ))
    })?;
    if policy == HostAllocationPolicy::Unbounded {
        return Ok(());
    }
    let route = vm.host_route.as_ref().ok_or_else(|| {
        Box::new(CpuReadbackFailure::BeforeEffect(
            runmat_runtime::build_runtime_error(
                "selected VM host numerical outputs lost their route",
            )
            .build(),
        ))
    })?;
    let outputs = crate::fusion_exec::elementwise::host_readback::prepare_outputs(runtime, &specs)
        .map_err(|cause| {
            Box::new(CpuReadbackFailure::BeforeEffect(
                host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Placement:HostNumericalOutput",
                    "selected VM host numerical output admission failed",
                    cause,
                ),
            ))
        })?
        .ok_or_else(|| {
            Box::new(CpuReadbackFailure::BeforeEffect(
                runmat_runtime::build_runtime_error(
                    "selected VM host numerical output lost its finite owner",
                )
                .build(),
            ))
        })?;
    route
        .install_numerical_outputs(runtime, outputs)
        .map_err(|error| Box::new(CpuReadbackFailure::BeforeEffect(error)))
}
