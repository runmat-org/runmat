//! Ordered source-store image execution. A later stage failure never rolls
//! back the mean, sigma, or output prefix already published by the VM.

use super::*;
use crate::bytecode::{ImageCheckpoint, ImageLiveOutSource, ImageStoreDestination};
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::fusion_exec::{ImageVmPreparation, ImageVmStages};
use runmat_accelerate::graph::ValueOrigin;

pub(super) enum ImageStageOutcome {
    Declined,
    Executed { next_pc: usize },
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
    let join = &selected.region.publication;
    // Only a selected source checkpoint can satisfy a region live-out. An
    // intermediate MIR slot cannot be omitted merely because it is unnamed.
    if join.live_outs().iter().any(|live| match live.source {
        ImageLiveOutSource::PreserveEntry => false,
        ImageLiveOutSource::LastStore(ordinal) => join
            .store_at(ordinal)
            .is_none_or(|entry| matches!(entry.destination, ImageStoreDestination::Intermediate)),
    }) {
        return Ok(ImageStageOutcome::Declined);
    }
    if join.ordered_stores().iter().any(|entry| {
        matches!(entry.destination, ImageStoreDestination::Checkpoint(_))
            && selected_stage_pc(selected.graph, entry).is_none()
    }) {
        return Ok(ImageStageOutcome::Declined);
    }
    let Some(inputs) = installation.capture_inputs(selected.region, frame.vars, frame.context)
    else {
        return Ok(ImageStageOutcome::Declined);
    };
    let Some(publications) = join.source_publications() else {
        return Ok(ImageStageOutcome::Declined);
    };
    let prepared = ImageVmStages::prepare(ImageVmPreparation {
        graph: selected.graph,
        plan: &selected.region.plan,
        source: std::sync::Arc::clone(&selected.region.source),
        inputs: &inputs,
        publications: &publications,
        runtime: &frame.context.runtime,
        placement: None,
        decline: Some(&selected.decline),
    })
    .await;
    let mut stage = match prepared {
        Ok(Some(stage)) => stage,
        Ok(None) => return Ok(ImageStageOutcome::Declined),
        Err(error) if error.is_unsupported() => return Ok(ImageStageOutcome::Declined),
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
    Ok(ImageStageOutcome::Executed {
        next_pc: join.pcs().end,
    })
}
