//! Ordered source-store image execution. A later stage failure never rolls
//! back the mean, sigma, or output prefix already published by the VM.

use super::*;
use crate::bytecode::{ImageCheckpoint, ImageLiveOutSource, ImageStoreDestination};
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::fusion_exec::{ImageVmPreparation, ImageVmSelectedRoute, ImageVmStages};
use runmat_accelerate::graph::ValueOrigin;

pub(super) enum ImageStageOutcome {
    Declined,
    Executed {
        next_pc: usize,
    },
    #[cfg(feature = "wgpu")]
    Original {
        span: runmat_accelerate::InstrSpan,
        publication: std::rc::Rc<crate::bytecode::ImageNormalizePublicationJoin>,
        route: Box<runmat_accelerate::fusion_exec::ImageVmOriginalRoute>,
    },
}

pub(super) struct ImageStageFrame<'a> {
    pub vars: &'a mut Vec<Value>,
    pub context: &'a mut crate::bytecode::program::ExecutionContext,
    pub var_names: &'a HashMap<usize, String>,
    pub completion_assigned_slots: &'a mut Option<HashSet<usize>>,
    pub global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    pub persistent_aliases: &'a crate::interpreter::PersistentAliases,
    pub current_function_name: &'a str,
    pub missing_input_slots: &'a mut HashSet<usize>,
}

fn selected_stage_pc(
    graph: &runmat_accelerate::graph::AccelGraph,
    entry: &crate::bytecode::ImageRegionStore,
) -> Option<usize> {
    let ValueOrigin::NodeOutput { node, .. } = &graph.value(entry.raw_stage_dependency?)?.origin
    else {
        return None;
    };
    Some(graph.node(*node)?.span.start)
}

fn publish(
    entry: &crate::bytecode::ImageRegionStore,
    handle: runmat_accelerate_api::GpuTensorHandle,
    frame: &mut ImageStageFrame<'_>,
) -> Result<(), RuntimeError> {
    set_vm_pc(entry.store.pc);
    publish_fusion_stores(
        vec![(
            FusionStoreMaterialization {
                value_id: entry.public_output,
                binding: entry.store.binding.clone(),
            },
            Value::GpuTensor(handle),
        )],
        FusionPublicationFrame {
            vars: &mut *frame.vars,
            context: &mut *frame.context,
            var_names: frame.var_names,
            completion_assigned_slots: &mut *frame.completion_assigned_slots,
            global_aliases: frame.global_aliases,
            persistent_aliases: frame.persistent_aliases,
            current_function_name: frame.current_function_name,
            missing_input_slots: &mut *frame.missing_input_slots,
        },
    )
}

pub(super) async fn execute(
    installation: &crate::bytecode::program::VmImagePlanInstallation,
    pc: usize,
    mut frame: ImageStageFrame<'_>,
) -> Result<ImageStageOutcome, RuntimeError> {
    let Some(selected) = installation.region_starting_at(pc, &frame.context.runtime) else {
        return Ok(ImageStageOutcome::Declined);
    };
    let outer_started = runmat_time::Instant::now();
    let join = &selected.region.publication;
    // Only a selected source checkpoint can satisfy a region live-out. An
    // intermediate MIR slot cannot be omitted merely because it is unnamed.
    if join.live_outs().iter().any(|live| match live.source {
        ImageLiveOutSource::PreserveEntry => false,
        ImageLiveOutSource::LastStore(ordinal) => join
            .store_at(ordinal)
            .is_none_or(|entry| matches!(entry.destination, ImageStoreDestination::Intermediate)),
    }) {
        log::debug!(target: "runmat_vm::accel::fusion::image_stages", "selected image declined: live-out source");
        return Ok(ImageStageOutcome::Declined);
    }
    if join.ordered_stores().iter().any(|entry| {
        matches!(entry.destination, ImageStoreDestination::Checkpoint(_))
            && selected_stage_pc(selected.graph, entry).is_none()
    }) {
        log::debug!(target: "runmat_vm::accel::fusion::image_stages", "selected image declined: stage PC");
        return Ok(ImageStageOutcome::Declined);
    }
    let Some(captured) = installation
        .capture_inputs(
            selected.region,
            frame.vars,
            frame.context,
            &frame.context.runtime,
        )
        .map_err(|error| {
            accel_fusion::fusion_execution_error(
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?
    else {
        log::debug!(target: "runmat_vm::accel::fusion::image_stages", "selected image declined: input capture");
        return Ok(ImageStageOutcome::Declined);
    };
    let prepared = ImageVmStages::prepare(ImageVmPreparation {
        graph: selected.graph,
        plan: &selected.region.plan,
        source: std::sync::Arc::clone(&selected.region.source),
        inputs: &captured.values,
        publications: &selected.region.source_publications,
        runtime: &frame.context.runtime,
        placement: None,
        decline: Some(&selected.decline),
        completion: selected.completion,
        started: outer_started,
    })
    .await;
    // The selected continuation retains its own authenticated preparation;
    // release VM input clones and their reservation before stage execution.
    drop(captured);
    let mut stage = match prepared {
        Ok(ImageVmSelectedRoute::Provider(stage)) => stage,
        #[cfg(feature = "wgpu")]
        Ok(ImageVmSelectedRoute::Original(route)) => {
            return Ok(ImageStageOutcome::Original {
                span: selected.region.plan.group.span.clone(),
                publication: std::rc::Rc::clone(&selected.region.publication),
                route,
            });
        }
        Ok(ImageVmSelectedRoute::Declined) => {
            log::debug!(target: "runmat_vm::accel::fusion::image_stages", "selected image declined: preparation");
            return Ok(ImageStageOutcome::Declined);
        }
        Err(error) if error.is_unsupported() => {
            log::debug!(target: "runmat_vm::accel::fusion::image_stages", "selected image declined: unsupported preparation {error:?}");
            return Ok(ImageStageOutcome::Declined);
        }
        Err(error) => return Err(accel_fusion::fusion_execution_error(error)),
    };
    let checkpoint = |role| {
        join.ordered_stores()
            .iter()
            .find(|entry| entry.destination == ImageStoreDestination::Checkpoint(role))
    };
    let mean_store = checkpoint(ImageCheckpoint::Mu).ok_or_else(|| {
        mex(
            "FusionImagePublication",
            "selected mean has no source store",
        )
    })?;
    set_vm_pc(selected_stage_pc(selected.graph, mean_store).expect("selected source PC preflight"));
    interp_engine::check_cancelled(&frame.context.runtime)?;
    let mean = stage
        .mean()
        .await
        .map_err(accel_fusion::fusion_execution_error)?;
    interp_engine::check_cancelled(&frame.context.runtime)?;
    publish(mean_store, mean, &mut frame)?;

    let sigma_store = checkpoint(ImageCheckpoint::Sigma).ok_or_else(|| {
        mex(
            "FusionImagePublication",
            "selected sigma has no source store",
        )
    })?;
    set_vm_pc(
        selected_stage_pc(selected.graph, sigma_store).expect("selected source PC preflight"),
    );
    interp_engine::check_cancelled(&frame.context.runtime)?;
    let sigma = stage
        .sigma()
        .await
        .map_err(accel_fusion::fusion_execution_error)?;
    interp_engine::check_cancelled(&frame.context.runtime)?;
    publish(sigma_store, sigma, &mut frame)?;

    for entry in join.ordered_stores().iter().filter(|entry| {
        matches!(
            entry.destination,
            ImageStoreDestination::Checkpoint(
                ImageCheckpoint::PreClamp | ImageCheckpoint::Clamp | ImageCheckpoint::Final
            )
        )
    }) {
        set_vm_pc(selected_stage_pc(selected.graph, entry).expect("selected source PC preflight"));
        interp_engine::check_cancelled(&frame.context.runtime)?;
        let output = stage
            .next_output_checkpoint()
            .await
            .map_err(accel_fusion::fusion_execution_error)?;
        if output.source_output != u64::from(entry.public_output) {
            return Err(mex(
                "FusionImagePublication",
                "selected output checkpoint differs from the source store",
            ));
        }
        interp_engine::check_cancelled(&frame.context.runtime)?;
        publish(entry, output.handle, &mut frame)?;
        stage
            .acknowledge_output_checkpoint(output.source_output)
            .map_err(accel_fusion::fusion_execution_error)?;
    }
    if let Some(completion) = stage.into_vm_completion_after_tail().await {
        completion.complete(outer_started);
    }
    Ok(ImageStageOutcome::Executed {
        next_pc: join.pcs().end,
    })
}
