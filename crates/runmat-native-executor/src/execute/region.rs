pub(super) mod observation;

use super::state::HostState;
use crate::region::RegionPlacement;
use crate::NativeExecutorResult;
use runmat_runtime::native::{NativeExit, NativeSiteOutcome};

pub(super) fn checkpoint(
    state: &mut HostState,
    request: runmat_runtime::native::NativeSiteRequest,
    exit: &mut NativeExit,
) -> NativeExecutorResult<Option<NativeSiteOutcome>> {
    use runmat_runtime::numeric_region::service::{
        NativeNumericRegionOutcome, NativeNumericRegionRequest, NumericRegionCapability,
    };
    let resumed = super::call_suspension::numeric_region::take(state, request)?;
    if super::call_suspension::has_completed_operation_at(state, request)? {
        // A nonnumeric completion proves this source occurrence already
        // entered its ordinary operation. Its exact typed/embedded consumer,
        // not a new region selection, must adopt that original owner first.
        // This also covers a declined region whose ordinary operation yielded
        // without installing an observation. Do not reconstruct decline state.
        return Ok(None);
    }
    let declined = resumed.is_some();
    if let Some(completed) = resumed {
        match completed.outcome {
            NativeNumericRegionOutcome::Ineligible => {}
            NativeNumericRegionOutcome::Original {
                right,
                observation: selected,
            } => {
                if !right.matches(completed.plan.region, request)
                    || !right.is_current(&state.runtime)
                {
                    if state.runtime.is_cancelled() {
                        *exit = super::control::cancelled(&state.runtime.cancellation());
                        return Ok(Some(NativeSiteOutcome::exit()));
                    }
                    return Err(crate::NativeExecutorError::Host(
                        "original native region authority changed before continuation".into(),
                    ));
                }
                for (input, original) in completed.plan.inputs.iter().zip(completed.inputs.iter()) {
                    if state.local_reference(input.local as usize)? != *original {
                        return Err(runmat_runtime::runtime_error::semantic_error(
                            "NativeRegionInputGenerationChanged",
                            "original native region input changed while selection was suspended",
                        )
                        .into());
                    }
                }
                let observation = crate::region::RegionObservation::original(
                    state.runtime.clone(),
                    right,
                    selected,
                );
                state.region_observation = Some(observation::GenericRegionObservation::new(
                    completed.plan,
                    observation,
                ));
                return Ok(None);
            }
            NativeNumericRegionOutcome::NativeCpu {
                kind,
                right,
                observation: selected_observation,
                host,
            } => {
                if !right.matches(completed.plan.region, request)
                    || !right.is_current(&state.runtime)
                {
                    if state.runtime.is_cancelled() {
                        *exit = super::control::cancelled(&state.runtime.cancellation());
                        return Ok(Some(NativeSiteOutcome::exit()));
                    }
                    return Err(crate::NativeExecutorError::Host(
                        "native CPU region authority changed before continuation".into(),
                    ));
                }
                let selected_observation = crate::region::RegionObservation::composed(
                    state.runtime.clone(),
                    right,
                    selected_observation,
                );
                match kind {
                    runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic => {
                        let host = host.ok_or_else(|| crate::NativeExecutorError::Host("selected native source-host route has no admitted host owner".into()))?;
                        if host.input_count() != completed.plan.inputs.len() {
                            return Err(crate::NativeExecutorError::Host("selected native source-host input count changed".into()));
                        }
                        for (input, original) in completed.plan.inputs.iter().zip(completed.inputs.iter()) {
                            if state.local_reference(input.local as usize)? != *original {
                                return Err(runmat_runtime::runtime_error::semantic_error("NativeRegionInputGenerationChanged", "native generic region input changed while its selection was suspended").into());
                            }
                        }
                        state.region_observation = Some(observation::GenericRegionObservation::new(completed.plan, selected_observation).with_host(host));
                        return Ok(None);
                    }
                    runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Vectorized => {
                        let inputs = crate::region::NumericRegionInputViews::try_from_iter(completed.inputs.iter().map(|reference| state.arena.get(*reference)))?;
                        let outputs = runmat_runtime::numeric_region::execute_in_context(&completed.plan.program, &inputs, &state.runtime)?;
                        match outputs {
                            runmat_runtime::numeric_region::NumericRegionOwnedExecution::Completed(outputs) => {
                                outputs.with_owned_output_storage(|storage| {
                                    state.publish_optimized_region_from_inputs(&completed.plan, &completed.inputs, super::call_suspension::own_output_storage(storage)?.into_iter())
                                })?;
                                selected_observation.complete();
                                return Ok(Some(NativeSiteOutcome::continue_execution()));
                            }
                            runmat_runtime::numeric_region::NumericRegionOwnedExecution::Cancelled => {
                                *exit = super::control::cancelled(&state.runtime.cancellation());
                                return Ok(Some(NativeSiteOutcome::exit()));
                            }
                            runmat_runtime::numeric_region::NumericRegionOwnedExecution::Ineligible => return Err(crate::NativeExecutorError::Host("selected native CPU inputs no longer match the admitted region".into())),
                        }
                    }
                }
            }
            NativeNumericRegionOutcome::Completed {
                outputs,
                right,
                publication,
            } => {
                if !right.matches(completed.plan.region, request)
                    || !right.is_current(&state.runtime)
                {
                    if state.runtime.is_cancelled() {
                        *exit = super::control::cancelled(&state.runtime.cancellation());
                        return Ok(Some(NativeSiteOutcome::exit()));
                    }
                    return Err(crate::NativeExecutorError::Host(
                        "native numeric region completion authority changed".into(),
                    ));
                }
                outputs.with_owned_output_storage(|storage| {
                    let values = super::call_suspension::own_output_storage(storage)?.into_iter();
                    state.publish_optimized_region_from_inputs(
                        &completed.plan,
                        &completed.inputs,
                        values,
                    )
                })?;
                if let Some(publication) = publication {
                    publication.complete(&state.runtime, right);
                }
                return Ok(Some(NativeSiteOutcome::continue_execution()));
            }
        }
    }
    // An enclosing generic region owns this execution interval. Do not time
    // overlapping nested alternatives or replace its pending observation.
    if state.region_observation.is_some() {
        return Ok(None);
    }
    let Some(plan) = state.optimized_region(request) else {
        return Ok(None);
    };
    // Refreshing canonical bindings mutably borrows the frame while the input
    // views are live. Retain its shared runtime services without copying inputs.
    let runtime = state.runtime.clone();
    if !declined
        && runtime
            .service_ports()
            .acceleration()
            .is_some_and(|service| {
                matches!(
                    service.numeric_region_capability(),
                    Some(NumericRegionCapability::DenseF64 | NumericRegionCapability::RealFloating)
                )
            })
    {
        let references = state.optimized_region_input_references(&plan)?;
        if !references.iter().any(|reference| reference.is_null()) {
            let inputs = crate::region::NumericRegionInputViews::try_from_iter(
                references
                    .iter()
                    .map(|reference| state.arena.get(*reference)),
            )?;
            let accepts_input_domain =
                runtime
                    .service_ports()
                    .acceleration()
                    .is_some_and(|service| {
                        service.numeric_region_capability()
                            == Some(NumericRegionCapability::RealFloating)
                            || runmat_runtime::numeric_region::workload(&inputs).is_some()
                    });
            if let Some(request) = accepts_input_domain
                .then(|| {
                    NativeNumericRegionRequest::capture_verified(
                        &runtime,
                        plan.region,
                        request,
                        &plan.program,
                        &inputs,
                    )
                })
                .transpose()?
                .flatten()
            {
                let request =
                    request.with_native_operation_sites(plan.operation_sites.iter().map(
                        |site| runmat_runtime::native::NativeSiteRequest {
                            function: site.point.function.0,
                            block: site.point.block,
                            position: site.point.position,
                            phase: if site.phase == runmat_native_codegen::NativeSitePhase::Rvalue {
                                runmat_runtime::native::NativeSitePhase::RVALUE
                            } else {
                                runmat_runtime::native::NativeSitePhase::TERMINATOR
                            },
                            ordinal: site.ordinal,
                            reserved: 0,
                        },
                    ))?;
                super::call_suspension::numeric_region::begin(state, plan, references, request)?;
                unreachable!("installing a numeric region always suspends")
            }
        }
    }
    let Some(inputs) = state.optimized_region_inputs(&plan)? else {
        return Ok(None);
    };
    let Some(workload) = runmat_runtime::numeric_region::workload(&inputs) else {
        return Ok(None);
    };
    let placement = match crate::region::choose_region(&runtime, &plan, &inputs, workload)? {
        RegionPlacement::Unavailable => return Ok(None),
        RegionPlacement::Generic(observation) => {
            state.region_observation = Some(observation::GenericRegionObservation::new(
                plan,
                observation,
            ));
            return Ok(None);
        }
        RegionPlacement::Vectorized(observation) => observation,
    };
    // The observation guard records failure on execution/publication errors,
    // cancellation or unwinding, and success only after publication completes.
    match runmat_runtime::numeric_region::execute_in_context(&plan.program, &inputs, &runtime)? {
        runmat_runtime::numeric_region::NumericRegionOwnedExecution::Completed(outputs) => {
            outputs.with_owned_output_storage(|storage| {
                state.publish_optimized_region(
                    &plan,
                    super::call_suspension::own_output_storage(storage)?.into_iter(),
                )
            })?;
            placement.complete();
            Ok(Some(NativeSiteOutcome::continue_execution()))
        }
        runmat_runtime::numeric_region::NumericRegionOwnedExecution::Ineligible => Ok(None),
        runmat_runtime::numeric_region::NumericRegionOwnedExecution::Cancelled => {
            *exit = super::control::cancelled(&runtime.cancellation());
            Ok(Some(NativeSiteOutcome::exit()))
        }
    }
}
