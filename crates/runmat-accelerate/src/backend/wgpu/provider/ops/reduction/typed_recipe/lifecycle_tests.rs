//! Real checked two-stage queue/lifetime witnesses; requires an actual adapter.
use super::*;
use crate::backend::wgpu::dispatch::observation::{
    with_retirement_for_test, SubmittedAllocationRetirement,
};
use crate::fusion::{
    reduction_recipe::{build, ReductionRecipe, ReductionRecipeConfig},
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use crate::fusion_exec::{resolve_reduction_program, FusionExecutionRequest};
use crate::graph::{InstrSpan, PrimitiveOp, ShapeInfo};
use anyhow::Result;
use futures::executor::block_on;
use futures::FutureExt;
use runmat_accelerate_api::observation::WgpuObservationProgress;
use runmat_accelerate_api::{
    allocation::AllocationReleaseOutcome,
    reduction::{ReductionSelection, ReductionStrategy, ReductionSubmission},
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorView, ProviderLayout,
    ProviderPrecision, ProviderTypedReductionWorkload, TransposeInfo, WgpuBufferOwner,
};
use runmat_value::{Tensor, Value};
use std::{collections::HashMap, sync::Weak};
#[path = "lifecycle_tests/retention.rs"]
mod retention;
#[path = "lifecycle_tests/managed_host.rs"]
mod managed_host;

fn owner() -> &'static WgpuProvider {
    let provider = Box::leak(Box::new(
        block_on(WgpuProvider::new_async(
            crate::backend::wgpu::provider::WgpuProviderOptions {
                precision: crate::backend::wgpu::provider::WgpuPrecisionPolicy::RequireF32,
                ..Default::default()
            },
        ))
        .expect("actual F32 WGPU adapter required; unavailable is not a pass"),
    ));
    // SAFETY: this unique actual device facade lives for the process and is
    // never mutated after publishing its exact allocation-owner identity.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

#[derive(Default)]
struct Trace {
    output: Option<Weak<wgpu::Buffer>>,
    partial: Option<Weak<wgpu::Buffer>>,
    stages: Vec<(Stage, u32)>,
    fail_after_first: bool,
    retirements: [Option<SubmittedAllocationRetirement>; 2],
    progress: Option<std::sync::Arc<dyn WgpuObservationProgress>>,
}
impl Trace {
    fn wait_retirements(&mut self) {
        for slot in &mut self.retirements {
            let Some(mut retirement) = slot.take() else {
                continue;
            };
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                if let Some(result) = std::pin::Pin::new(&mut retirement).now_or_never() {
                    result.expect("actual exact-submission retirement");
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "exact typed-reduction resource owner did not retire"
                );
                std::thread::yield_now();
            }
        }
    }
}
impl ReductionObserver for Trace {
    fn prepare_pending(
        &self,
        pending: &mut crate::backend::wgpu::dispatch::observation::PendingSubmissions,
    ) {
        if let Some(progress) = &self.progress {
            pending.set_progress_port_for_test(progress.clone());
        }
    }
    fn retaining(
        &mut self,
        resources: SubmittedReductionResources,
    ) -> Result<crate::backend::wgpu::dispatch::observation::SubmittedAllocationRetention> {
        let (retention, retirement) = with_retirement_for_test(resources.into_retention())?;
        let slot = self
            .retirements
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or_else(|| anyhow::anyhow!("typed recipe emitted more than its two stages"))?;
        *slot = Some(retirement);
        Ok(retention)
    }
    fn output_acquired(&mut self, output: &runmat_accelerate_api::WgpuBufferOwner) -> Result<()> {
        self.output = Some(WgpuBufferOwner::downgrade(output));
        Ok(())
    }

    fn partial_acquired(&mut self, partial: &runmat_accelerate_api::WgpuBufferOwner) -> Result<()> {
        self.partial = Some(WgpuBufferOwner::downgrade(partial));
        Ok(())
    }

    fn submitted(&mut self, stage: Stage, id: u32) -> Result<()> {
        self.stages.push((stage, id));
        if self.fail_after_first && stage == Stage::First {
            anyhow::bail!("injected failure after authentic pass1 submission");
        }
        Ok(())
    }
}

fn checked_recipe_with_config(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    config: ReductionRecipeConfig,
) -> ReductionRecipe {
    // Same direct canonical resolution and checked lowering path used by the
    // public recipe fixture. The controls are min([],1,"includenan").
    let operations = vec![
        FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(100),
        },
        FusionOp::Builtin {
            name: "min".into(),
            origin: Default::default(),
            inputs: vec![100, 2, 3, 4],
            output: Some(101),
        },
    ];
    let plan = FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::Reduction,
            nodes: vec![],
            shape: ShapeInfo::Tensor(vec![Some(2), Some(3)]),
            span: InstrSpan {
                start: 0,
                end: operations.len(),
            },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0, 1, 2, 3, 4],
        stack_pattern: vec![0],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(101),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(100),
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    };
    let controls = vec![
        Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
        Value::Num(1.0),
        Value::String("includenan".into()),
    ];
    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(a.clone()), Value::GpuTensor(b.clone())]
            .into_iter()
            .chain(controls)
            .collect(),
        placement: None,
        runtime: None,
    };
    let program = block_on(resolve_reduction_program(&request, ProviderPrecision::F32))
        .unwrap()
        .expect("canonical checked folded reduction");
    build(&program, config, None).expect("physical checked recipe")
}

fn checked_recipe(a: &GpuTensorHandle, b: &GpuTensorHandle) -> ReductionRecipe {
    let config = ReductionRecipeConfig {
        workgroup_size: 8,
        chunk_elements: 1,
        max_workgroup_size: 256,
        max_workgroup_storage_bytes: 32_768,
        max_dispatch_dimension: 65_535,
        max_storage_binding_bytes: 1 << 30,
        max_rank: 8,
    };
    checked_recipe_with_config(a, b, config)
}

#[test]
#[ignore = "requires actual WGPU adapter; unavailable is failure"]
fn authentic_checked_candidate_probe_matches_enabled_limits_without_effects() {
    let provider = owner();
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let upload = || {
        provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &[2, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
    };
    let a = upload();
    let b = upload();
    let seed = checked_recipe(&a, &b);
    let geometry = seed.geometry();
    let records = provider.allocations.record_count().unwrap();
    let pipelines = provider.fused_pipeline_cache.lock().unwrap().len();
    let metrics = provider.metrics.counters();
    let config = provider
        .typed_reduction_config(geometry.reduce_elements, geometry.slices)
        .expect("enabled typed reduction configuration");
    assert_eq!(
        config.max_storage_binding_bytes,
        provider.adapter_limits.max_buffer_size.min(u64::from(
            provider.adapter_limits.max_storage_buffer_binding_size
        ))
    );
    let recipe = checked_recipe_with_config(&a, &b, config);
    assert_eq!(
        provider
            .typed_reduction_config(recipe.geometry().reduce_elements, recipe.geometry().slices,)
            .unwrap(),
        recipe.config()
    );
    let mut wrong_config = config;
    wrong_config.chunk_elements = if config.chunk_elements == 1 { 2 } else { 1 };
    let wrong_recipe = checked_recipe_with_config(&a, &b, wrong_config);
    assert!(provider
        .typed_reduction_candidates(wrong_recipe.physical(), None)
        .unwrap_err()
        .is_unsupported());
    let (preferred, feasible) = provider
        .typed_reduction_feasible_candidates(recipe.physical(), None)
        .unwrap();
    let select = |mode| {
        super::super::typed_selection::select_candidates(mode, preferred, &feasible).unwrap()
    };
    assert_eq!(
        select(runmat_accelerate_api::ReductionTwoPassMode::ForceOff),
        [Some(ReductionStrategy::SinglePass), None],
    );
    assert_eq!(
        select(runmat_accelerate_api::ReductionTwoPassMode::ForceOn),
        [Some(ReductionStrategy::TwoPass), None],
    );
    let auto = select(runmat_accelerate_api::ReductionTwoPassMode::Auto);
    assert_eq!(auto.iter().flatten().count(), 2);
    assert!(auto.contains(&Some(ReductionStrategy::SinglePass)));
    assert!(auto.contains(&Some(ReductionStrategy::TwoPass)));
    assert_eq!(provider.allocations.record_count().unwrap(), records);
    assert_eq!(
        provider.fused_pipeline_cache.lock().unwrap().len(),
        pipelines
    );
    assert_eq!(provider.metrics.counters(), metrics);
}

#[test]
#[ignore = "requires actual WGPU adapter; unavailable is failure"]
fn authentic_two_pass_tracks_final_id_and_drains_after_first_stage_fault() {
    let provider = owner();
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let upload = |values: &[f32]| {
        provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(values),
                shape: &[2, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap()
    };
    let a = upload(&left);
    let b = upload(&right);
    let recipe = checked_recipe(&a, &b);
    assert!(recipe.geometry().chunks > 1);
    let inputs = [a.clone(), b.clone()];
    let before = provider.allocations.record_count().unwrap();
    assert!(before >= 2, "both uploaded inputs must have owner records");
    let read = |handle: &GpuTensorHandle| match block_on(provider.download_numeric(handle))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => values,
        other => panic!("expected native F32: {other:?}"),
    };

    {
        let selected =
            ReductionSelection::new(recipe.physical(), ReductionStrategy::TwoPass).unwrap();
        let submitted =
            ReductionSubmission::new(selected, &inputs, provider.device_id(), None, None).unwrap();
        let mut success = Trace::default();
        let output = provider
            .submit_reduction_recipe_observed(&submitted, &mut success)
            .expect("authentic two-pass dispatch");
        assert_eq!(success.stages.len(), 2);
        assert_eq!(success.stages[0].0, Stage::First);
        assert_eq!(success.stages[1].0, Stage::Final);
        assert_ne!(success.stages[0].1, success.stages[1].1);
        assert_eq!(
            provider.get_entry_raw(&output).unwrap().last_submission_id,
            Some(success.stages[1].1),
        );
        assert_eq!(provider.allocations.record_count().unwrap(), before + 1);
        assert_eq!(read(&output), vec![11.0, 33.0, 55.0]);
        success.wait_retirements();
        assert!(success.partial.as_ref().unwrap().upgrade().is_none());
        assert_eq!(output.release().unwrap(), AllocationReleaseOutcome::Retired);
        assert_eq!(provider.allocations.record_count().unwrap(), before);
    }

    {
        let selected =
            ReductionSelection::new(recipe.physical(), ReductionStrategy::TwoPass).unwrap();
        let submitted =
            ReductionSubmission::new(selected, &inputs, provider.device_id(), None, None).unwrap();
        let mut fault = Trace {
            fail_after_first: true,
            ..Default::default()
        };
        let error = provider
            .submit_reduction_recipe_observed(&submitted, &mut fault)
            .expect_err("injected post-pass1 failure must not publish output");
        assert!(error.is_after_effect());
        assert!(!error.is_unsupported());
        assert_eq!(fault.stages.len(), 1);
        assert_eq!(fault.stages[0].0, Stage::First);
        // Queue-ordered readback drains the already-submitted first stage.
        assert_eq!(read(&a), left);
        assert_eq!(read(&b), right);
        fault.wait_retirements();
        assert!(fault.output.as_ref().unwrap().upgrade().is_none());
        assert!(fault.partial.as_ref().unwrap().upgrade().is_none());
        assert_eq!(provider.allocations.record_count().unwrap(), before);
    }
    drop(inputs);
    assert_eq!(a.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert_eq!(b.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert_eq!(provider.allocations.record_count().unwrap(), before - 2);
    retention::prove_exact_owners(provider);
}

#[test]
#[ignore = "requires actual WGPU adapter; unavailable is failure"]
fn same_class_transpose_binds_the_source_buffer_and_uses_checked_strides() {
    let provider = owner();
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let upload = |values: &[f32], shape: &[usize]| {
        provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(values),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
    };
    let physical = upload(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], &[3, 2]);
    let transposed = physical
        .with_view(physical.view().clone().with_shape(vec![2, 3]).with_layout(
            GpuTensorLayout::Transpose(TransposeInfo {
                base_rows: 3,
                base_cols: 2,
            }),
        ))
        .unwrap();
    drop(physical);
    let contiguous = upload(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    let recipe = checked_recipe(&transposed, &contiguous);
    assert_eq!(recipe.inputs()[0].source_precision, ProviderPrecision::F32);
    assert_eq!(recipe.inputs()[0].bound_precision, ProviderPrecision::F32);
    assert!(matches!(
        recipe.physical().descriptor().inputs[0].layout,
        GpuTensorLayout::Transpose(_)
    ));
    let selection = ReductionSelection::new(recipe.physical(), ReductionStrategy::SinglePass)
        .expect("checked same-class transpose selection");
    let workload = ProviderTypedReductionWorkload::from_selection(&selection);
    assert_eq!(workload.inputs[0].layout, ProviderLayout::Strided);
    let inputs = [transposed.clone(), contiguous.clone()];
    let submission = ReductionSubmission::new(selection, &inputs, provider.device_id(), None, None)
        .expect("checked transpose submission");

    let source = provider.get_entry_raw(&transposed).unwrap().buffer;
    let admitted = provider
        .admit_reduction_recipe(&submission)
        .expect("provider authenticates transpose owner");
    assert!(WgpuBufferOwner::ptr_eq(&source, &admitted[0].buffer));
    drop(admitted);
    let before = provider.allocations.record_count().unwrap();
    let mut trace = Trace::default();
    let output = provider
        .submit_reduction_recipe_observed(&submission, &mut trace)
        .expect("same-class transpose reduction");
    assert_eq!(provider.allocations.record_count().unwrap(), before + 1);
    let values = match block_on(provider.download_numeric(&output)).unwrap().data {
        HostNumericDataOwned::F32(values) => values,
        other => panic!("expected native F32: {other:?}"),
    };
    assert_eq!(values, vec![11.0, 33.0, 55.0]);
    trace.wait_retirements();
    assert_eq!(output.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    drop(source);
    drop(inputs);
    assert_eq!(
        transposed.release().unwrap(),
        AllocationReleaseOutcome::Retired
    );
    assert_eq!(
        contiguous.release().unwrap(),
        AllocationReleaseOutcome::Retired
    );
}
