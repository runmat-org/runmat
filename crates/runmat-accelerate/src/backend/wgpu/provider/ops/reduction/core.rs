use super::global_pass::{GlobalPassKind, GlobalReductionPassPolicy};
use super::phase_context::{DimSumMeanSelection, SelectedEmission, StdFormulaPhase};
use super::*;

#[path = "core/power_probe.rs"]
mod power_probe;
pub(crate) use power_probe::{AnyStageObserved, FrozenAnyPipeline, FrozenAnySnapshot};
#[path = "core/logical_global.rs"]
mod logical_global;

struct DimEmissionObserved {
    output_buffer_bytes: u64,
    snapshot_copy_bytes: Option<u64>,
    uniform_buffer_bytes: Option<u64>,
    workgroups_x: Option<u32>,
    snapshot_copy_submitted: bool,
    dispatch_submitted: bool,
    snapshot_fence_requested_bytes: Option<u64>,
    dispatch_fence_requested_bytes: Option<u64>,
    snapshot_ticket: Option<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    dispatch_ticket: Option<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    output_resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
}

impl DimEmissionObserved {
    fn new(
        output_buffer_bytes: u64,
        snapshot_copy_bytes: Option<u64>,
        snapshot_copy_submitted: bool,
        snapshot_fence_requested_bytes: Option<u64>,
    ) -> Self {
        Self {
            output_buffer_bytes,
            snapshot_copy_bytes,
            uniform_buffer_bytes: None,
            workgroups_x: None,
            snapshot_copy_submitted,
            dispatch_submitted: false,
            snapshot_fence_requested_bytes,
            dispatch_fence_requested_bytes: None,
            snapshot_ticket: None,
            dispatch_ticket: None,
            output_resource_ticket: None,
        }
    }
}

impl WgpuProvider {
    pub(crate) fn prepare_power_any_pipeline_exec(&self) -> Result<FrozenAnyPipeline> {
        FrozenAnyPipeline::prepare(
            self.device_id(),
            self.precision,
            &self.pipelines.reduce_dim_sum_mean,
        )
    }

    pub(crate) fn reduce_any_exec(
        &self,
        a: &GpuTensorHandle,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        self.reduce_logical_global_exec(a, runmat_builtins::LogicalReductionKind::Any, omit_nan)
    }

    /// The POWER probe uses both ordinary stages with one snapshot decision.
    pub(crate) fn reduce_power_any_omit_observed_exec(
        &self,
        a: &GpuTensorHandle,
        snapshot: FrozenAnySnapshot,
        pipeline: &FrozenAnyPipeline,
    ) -> Result<(GpuTensorHandle, [AnyStageObserved; 2])> {
        use crate::backend::wgpu::types::DimReduceOp;
        pipeline.require(
            self.device_id(),
            self.precision,
            &self.pipelines.reduce_dim_sum_mean,
        )?;
        let (first, first_observed) = self.reduce_dim_sum_mean_with_snapshot_exec(
            a,
            0,
            DimReduceOp::AnyOmit,
            snapshot,
            Some(pipeline),
        )?;
        first_observed.require_output(&first)?;
        if let Some(scope) = runmat_accelerate_api::observation::current_provider_resource_scope() {
            super::super::power::PowerProbeResourcePhase::AnyColumns.begin(&scope)?;
        }
        let (second, second_observed) = self.reduce_dim_sum_mean_with_snapshot_exec(
            &first,
            1,
            DimReduceOp::AnyOmit,
            snapshot,
            Some(pipeline),
        )?;
        second_observed.require_output(&second)?;
        drop(first);
        Ok((second, [first_observed, second_observed]))
    }

    pub(crate) fn reduce_all_exec(
        &self,
        a: &GpuTensorHandle,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        self.reduce_logical_global_exec(a, runmat_builtins::LogicalReductionKind::All, omit_nan)
    }

    pub(crate) fn reduce_mean_global_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let sum_handle =
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let total_elems: usize = self.get_entry(a)?.len.max(1);
        let scalar = 1.0 / (total_elems as f64);
        let out = self.scalar_op_exec(
            crate::backend::wgpu::types::ScalarOpCode::Mul,
            &sum_handle,
            scalar,
        )?;
        drop(sum_handle);
        Ok(out)
    }

    pub(crate) fn reduce_any_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AnyOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AnyInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    pub(crate) fn reduce_all_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AllOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AllInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_global_exec(
        &self,
        a: &GpuTensorHandle,
        op: crate::backend::wgpu::types::GlobalReduceOp,
    ) -> Result<GpuTensorHandle> {
        self.reduce_global_with_selected_pipeline_exec(a, op, SelectedEmission::legacy())
    }

    pub(super) fn reduce_global_with_selected_pipeline_exec(
        &self,
        a: &GpuTensorHandle,
        op: crate::backend::wgpu::types::GlobalReduceOp,
        emission: SelectedEmission<'_>,
    ) -> Result<GpuTensorHandle> {
        self.reduce_global_recorded_exec(a, op, emission, None)
    }

    pub(super) fn reduce_global_recorded_exec(
        &self,
        a: &GpuTensorHandle,
        op: crate::backend::wgpu::types::GlobalReduceOp,
        emission: SelectedEmission<'_>,
        observed: Option<&std::cell::RefCell<runmat_accelerate_api::ProviderGlobalExtremaObserved>>,
    ) -> Result<GpuTensorHandle> {
        let SelectedEmission {
            selected: validated,
            effect: allocated,
            after_allocation,
        } = emission;
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            log::debug!(
                "[reduce-global] in ptr={:p} len={} op={}",
                entry.buffer.as_ref(),
                entry.len,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_GLOBAL").is_ok() {
            return Err(anyhow!(
                "reduce_global disabled via RUNMAT_DISABLE_REDUCE_GLOBAL"
            ));
        }
        if entry.len == 0 {
            ensure!(
                observed.is_none(),
                "selected global probe requires a nonempty source"
            );
            let default = match op {
                crate::backend::wgpu::types::GlobalReduceOp::Sum => 0.0,
                crate::backend::wgpu::types::GlobalReduceOp::Prod => 1.0,
                crate::backend::wgpu::types::GlobalReduceOp::Min => f64::INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::Max => f64::NEG_INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::CountNonZero => 0.0,
            };
            let data = [default];
            let shape = [1usize, 1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            return self.upload_exec(&view);
        }
        self.require_ieee_words(entry.precision)?;
        let pass_policy = GlobalReductionPassPolicy::for_operation(op);
        let launch = self.global_reduce_launch;
        let max_input_per_pass = launch
            .max_input_per_pass()
            .ok_or_else(|| anyhow!("reduce_global: frozen launch capacity overflows"))?;
        let current = self.with_emission(|emission, pending| -> Result<_> {
        // A selected recipe pins the no-snapshot physical route. Legacy hooks
        // retain their environment-selected diagnostic copy behavior.
        let mut current = if observed.is_none() && std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-global-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if let Some(allocated) = allocated {
                allocated.set(true);
            }
            let mut enc = pending.profiled_encoder(emission, &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-global-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(emission, pending, enc)?;
            snap
        } else {
            entry.buffer.clone()
        };
        let mut current_len = entry.len;
        let mut pass_kind = GlobalPassKind::SourceMapping;
        while pass_policy.should_dispatch(pass_kind, current_len) {
            let output_len_total = launch.output_len(current_len).max(1);
            // Metal-safe (opt-in): snapshot input buffer for this pass to avoid any SR/SRW conflicts
            // Enabled only if RUNMAT_FORCE_REDUCE_SNAPSHOT is set to avoid perf impact by default.
            let mut input_for_pass = current.clone();
            if observed.is_none() && self.adapter_info.backend == wgpu::Backend::Metal
                && std::env::var("RUNMAT_FORCE_REDUCE_SNAPSHOT").is_ok()
            {
                let size_bytes = (current_len * self.element_size) as u64;
                let snap = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-input-snapshot"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                if let Some(allocated) = allocated {
                    allocated.set(true);
                }
                let mut enc = pending.profiled_encoder(emission, &wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-reduce-pass-input-snapshot-copy"),
                        });
                enc.copy_buffer_to_buffer(current.as_ref(), 0, snap.as_ref(), 0, size_bytes);
                self.submit(emission, pending, enc)?;
                input_for_pass = snap;
            }
            let mut out_buffer =
                self.create_storage_buffer_checked(emission, output_len_total, "runmat-reduce-pass")?;
            if let Some(allocated) = allocated {
                allocated.set(true);
            }
            if let Some(observed) = observed {
                let mut value = observed.borrow_mut();
                value.allocated_buffer_bytes = value.allocated_buffer_bytes.checked_add(out_buffer.size())
                    .ok_or_else(|| anyhow!("global extrema allocation accounting overflow"))?;
            }
            if let Some(after_allocation) = after_allocation {
                after_allocation(&out_buffer)?;
            }
            // Prevent aliasing: output buffer must not be the same as input buffer
            if std::ptr::eq(out_buffer.as_ref(), input_for_pass.as_ref()) {
                ensure!(observed.is_none(), "selected global allocator aliased a borrowed source; an unadmitted replacement is forbidden");
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: alias detected; current_ptr={:p} out_ptr={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                    eprintln!(
                        "[reduction] alias current={:p} out={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                }
                let size_bytes = (output_len_total * self.element_size) as u64;
                if size_bytes > self.adapter_limits.max_buffer_size {
                    return Err(gpu_per_buffer_limit_error(
                        "runmat-reduce-pass-unique",
                        size_bytes,
                        self.adapter_limits.max_buffer_size,
                    ));
                }
                out_buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-unique"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                if let Some(observed) = observed {
                    let mut value = observed.borrow_mut();
                    value.allocated_buffer_bytes = value.allocated_buffer_bytes.checked_add(out_buffer.size())
                        .ok_or_else(|| anyhow!("global extrema allocation accounting overflow"))?;
                }
            }

            let mut in_offset_elems = 0usize;
            let mut _out_offset_elems = 0usize;
            while in_offset_elems < current_len {
                let remain = current_len - in_offset_elems;
                let chunk_in = remain.min(max_input_per_pass);
                let chunk_out = launch.output_len(chunk_in).max(1);

                let params = crate::backend::wgpu::params::ReduceGlobalParams {
                    len: chunk_in as u32,
                    op: pass_policy.operation(pass_kind) as u32,
                    offset: in_offset_elems as u32,
                    total: current_len as u32,
                };
                let params_buffer = self.uniform_buffer(emission, &params, "runmat-reduce-global-params")?;
                if let Some(observed) = observed {
                    let mut value = observed.borrow_mut();
                    value.uniform_allocations = value.uniform_allocations.checked_add(1)
                        .ok_or_else(|| anyhow!("global extrema uniform count overflow"))?;
                    value.uniform_buffer_bytes = value.uniform_buffer_bytes.checked_add(params_buffer.size())
                        .ok_or_else(|| anyhow!("global extrema uniform accounting overflow"))?;
                }

                let bind_group = emission.device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduce-global-bind"),
                        layout: match validated {
                            Some(pipeline) => pipeline.layout.as_ref(),
                            None => &self.pipelines.reduce_global.get(emission)?.layout,
                        },
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_for_pass.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let groups = launch
                    .dispatch_groups(chunk_in)
                    .ok_or_else(|| anyhow!("reduce_global: dispatch exceeds frozen launch"))?;
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: dispatch groups={} current_ptr={:p} out_ptr={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                    eprintln!(
                        "[reduction] dispatch groups={} in={:p} out={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                }
                crate::backend::wgpu::dispatch::reduction::run_single_pass(
                    emission,
                    pending,
                    match validated {
                        Some(pipeline) => pipeline.pipeline.as_ref(),
                        None => &self.pipelines.reduce_global.get(emission)?.pipeline,
                    },
                    &bind_group,
                    groups,
                )?;
                if let Some(observed) = observed {
                    let mut value = observed.borrow_mut();
                    value.dispatches = value.dispatches.checked_add(1)
                        .ok_or_else(|| anyhow!("global extrema dispatch accounting overflow"))?;
                    value.queue_submissions = value.queue_submissions.checked_add(1)
                        .ok_or_else(|| anyhow!("global extrema queue accounting overflow"))?;
                }
                in_offset_elems += chunk_in;
                _out_offset_elems += chunk_out;
            }

            current = out_buffer;
            current_len = output_len_total;
            pass_kind = GlobalPassKind::PartialCombine;
        }
        Ok(current)
        })??;
        if runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&current, &entry.buffer) {
            return Ok(a.with_view(a.view().clone().with_shape(vec![1, 1]).with_logical(false))?);
        }
        self.register_existing_buffer(current, vec![1, 1], 1)
    }
    pub(crate) fn reduce_dim_sum_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceOp,
    ) -> Result<GpuTensorHandle> {
        Ok(self
            .reduce_dim_sum_mean_with_snapshot_exec(a, dim, op, FrozenAnySnapshot::capture(), None)?
            .0)
    }

    fn reduce_dim_sum_mean_with_snapshot_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceOp,
        snapshot: FrozenAnySnapshot,
        pipeline: Option<&FrozenAnyPipeline>,
    ) -> Result<(GpuTensorHandle, AnyStageObserved)> {
        self.reduce_dim_sum_mean_selected_exec(
            a,
            dim,
            op,
            DimSumMeanSelection {
                snapshot,
                power_pipeline: pipeline,
                emission: SelectedEmission::legacy(),
            },
        )
    }

    pub(super) fn reduce_std_dim_sum_selected_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        validated: &super::super::validated_fusion::ValidatedFusionPipeline,
        effect: &std::cell::Cell<bool>,
        after_allocation: Option<&super::phase_context::AllocationFault<'_>>,
    ) -> Result<GpuTensorHandle> {
        Ok(self
            .reduce_dim_sum_mean_selected_exec(
                a,
                dim,
                crate::backend::wgpu::types::DimReduceOp::Sum,
                DimSumMeanSelection {
                    snapshot: FrozenAnySnapshot::capture(),
                    power_pipeline: None,
                    emission: SelectedEmission {
                        selected: Some(validated),
                        effect: Some(effect),
                        after_allocation,
                    },
                },
            )?
            .0)
    }

    fn reduce_dim_sum_mean_selected_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceOp,
        selection: DimSumMeanSelection<'_>,
    ) -> Result<(GpuTensorHandle, AnyStageObserved)> {
        let DimSumMeanSelection {
            snapshot,
            power_pipeline: pipeline,
            emission:
                SelectedEmission {
                    selected: validated,
                    effect,
                    after_allocation,
                },
        } = selection;
        let entry = self.get_entry(a)?;
        if let Some(pipeline) = pipeline {
            pipeline.require(
                self.device_id(),
                entry.precision,
                &self.pipelines.reduce_dim_sum_mean,
            )?;
        }
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_DIM").is_ok() {
            return Err(anyhow!("reduce_dim disabled via RUNMAT_DISABLE_REDUCE_DIM"));
        }
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        let (_, shader_workgroup) = self
            .pipelines
            .reduce_dim_sum_mean
            .frozen_reduction_source()
            .ok_or_else(|| anyhow!("dimensional reduction shader workgroup is not frozen"))?;
        let output_words = u32::try_from(out_len)
            .map_err(|_| anyhow!("dimensional reduction output exceeds u32 dispatch"))?;
        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(output_words, shader_workgroup);
        ensure!(
            shader_workgroup > 0
                && groups <= self.adapter_limits.max_compute_workgroups_per_dimension,
            "dimensional reduction dispatch exceeds enabled limits"
        );
        self.require_ieee_words(entry.precision)?;
        let snapshot_size_bytes = snapshot
            .enabled
            .then(|| {
                entry
                    .len
                    .checked_mul(self.element_size)
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| anyhow!("reduce-dim snapshot extent overflow"))
            })
            .transpose()?;
        if let Some(bytes) = snapshot_size_bytes {
            ensure!(
                bytes <= self.adapter_limits.max_buffer_size
                    && bytes <= u64::from(self.adapter_limits.max_storage_buffer_binding_size),
                "reduce-dim snapshot exceeds buffer/binding limit"
            );
        }
        let (out_buffer, mut emitted) = self.with_emission(|emission, pending| -> Result<_> {
        // Optional snapshot of input
        let mut snapshot_copy_bytes = None;
        let mut snapshot_copy_submitted = false;
        let mut snapshot_fence_requested_bytes = None;
        let mut snapshot_ticket = None;
        let mut snapshot_resource_ticket = None;
        let in_buf = if let Some(size_bytes) = snapshot_size_bytes {
            let snap = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if let Some(effect) = effect { effect.set(true); }
            if pipeline.is_some() {
                snapshot_resource_ticket =
                    self.issue_power_probe_resource_ticket(snap.size())?;
            }
            let mut enc = pending.profiled_encoder(emission, &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-dim-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            let ticket = pipeline.map(|_| pending.arm_probe_completion());
            self.submit(emission, pending, enc)?;
            snapshot_copy_bytes = Some(size_bytes);
            snapshot_copy_submitted = true;
            snapshot_fence_requested_bytes = pending.last_fence_requested_bytes();
            snapshot_ticket = ticket;
            snap
        } else {
            entry.buffer.clone()
        };
        let mut out_buffer =
            self.create_storage_buffer_checked(emission, out_len, "runmat-reduce-dim-out")?;
        if let Some(effect) = effect { effect.set(true); }
        if let Some(after_allocation) = after_allocation { after_allocation(&out_buffer)?; }
        // Prevent aliasing: output must not be identical to input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                log::debug!(
                    "reduce_dim_sum_mean_exec: alias detected; in_ptr={:p} out_ptr={:p} rows={} cols={} out_len={}",
                    entry.buffer.as_ref(),
                    out_buffer.as_ref(),
                    rows,
                    cols,
                    out_len
                );
            }
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-out-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            out_buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let output_resource_ticket = if pipeline.is_some() {
            self.issue_power_probe_resource_ticket(out_buffer.size())?
        } else {
            None
        };
        if out_len == 0 {
            let bytes = out_buffer.size();
            let mut observed = DimEmissionObserved::new(bytes, snapshot_copy_bytes, snapshot_copy_submitted, snapshot_fence_requested_bytes);
            observed.snapshot_ticket = snapshot_ticket;
            observed.output_resource_ticket = output_resource_ticket;
            return Ok((out_buffer, observed));
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(emission, &params, "runmat-reduce-dim-params")?;
        let uniform_buffer_bytes = params_buffer.size();
        let uniform_resource_ticket = if pipeline.is_some() {
            self.issue_power_probe_resource_ticket(uniform_buffer_bytes)?
        } else {
            None
        };
        let bind_group = emission.device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-dim-bind"),
                layout: match validated {
                    Some(selected) => selected.layout.as_ref(),
                    None => &self.pipelines.reduce_dim_sum_mean.get(emission)?.layout,
                },
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean] in ptr={:p} out ptr={:p} rows={} cols={} dim={} groups={}",
                entry.buffer.as_ref(),
                out_buffer.as_ref(),
                rows,
                cols,
                reduce_dim,
                groups
            );
        }
        let dispatch_ticket = pipeline.map(|_| pending.arm_probe_completion());
        for ticket in [snapshot_resource_ticket, uniform_resource_ticket]
            .into_iter()
            .flatten()
        {
            pending.retain_probe_resource_until_next_submit(ticket);
        }
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            emission,
            pending,
            match validated {
                Some(selected) => selected.pipeline.as_ref(),
                None => &self.pipelines.reduce_dim_sum_mean.get(emission)?.pipeline,
            },
            &bind_group,
            groups,
        )?;
        let mut observed = DimEmissionObserved::new(out_buffer.size(), snapshot_copy_bytes, snapshot_copy_submitted, snapshot_fence_requested_bytes);
        observed.uniform_buffer_bytes = Some(uniform_buffer_bytes);
        observed.workgroups_x = Some(groups);
        observed.dispatch_submitted = true;
        observed.dispatch_fence_requested_bytes = pending.last_fence_requested_bytes();
        observed.snapshot_ticket = snapshot_ticket;
        observed.dispatch_ticket = dispatch_ticket;
        observed.output_resource_ticket = output_resource_ticket;
        Ok((out_buffer, observed))
        })??;
        let output = if matches!(
            op,
            crate::backend::wgpu::types::DimReduceOp::AnyInclude
                | crate::backend::wgpu::types::DimReduceOp::AnyOmit
                | crate::backend::wgpu::types::DimReduceOp::AllInclude
                | crate::backend::wgpu::types::DimReduceOp::AllOmit
        ) {
            self.register_logical_buffer(out_buffer, out_shape, out_len, BufferUsageClass::Generic)?
        } else {
            self.register_existing_buffer(out_buffer, out_shape, out_len)?
        };
        if let Some(ticket) = emitted.output_resource_ticket.take() {
            self.allocations
                .attach_probe_resource_ticket(&output, ticket)?;
        }
        let observed = AnyStageObserved {
            input_view: a.view_identity(),
            output_allocation: output.allocation_identity(),
            output_view: output.view_identity(),
            output_buffer_bytes: emitted.output_buffer_bytes,
            snapshot_copy_bytes: emitted.snapshot_copy_bytes,
            uniform_buffer_bytes: emitted.uniform_buffer_bytes,
            workgroups_x: emitted.workgroups_x,
            snapshot_copy_submitted: emitted.snapshot_copy_submitted,
            dispatch_submitted: emitted.dispatch_submitted,
            snapshot_fence_requested_bytes: emitted.snapshot_fence_requested_bytes,
            dispatch_fence_requested_bytes: emitted.dispatch_fence_requested_bytes,
            snapshot_ticket: emitted.snapshot_ticket,
            dispatch_ticket: emitted.dispatch_ticket,
        };
        Ok((output, observed))
    }

    pub(super) fn reduce_dim_minmax_with_selected_pipeline_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceExtrema,
        emission: SelectedEmission<'_>,
    ) -> Result<ReduceDimResult> {
        let SelectedEmission {
            selected: validated,
            effect: allocated,
            after_allocation,
        } = emission;
        let original = self.get_entry_raw(a)?;
        if original.storage != GpuTensorStorage::Real || a.view().is_logical() {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "dimensional extrema requires a real numeric input",
            ));
        }
        let geometry = super::extrema_geometry::geometry(
            &original.shape,
            dim,
            self.precision,
            self.workgroup_config.scalar,
            self.adapter_limits.max_compute_workgroups_per_dimension,
        )?;
        if original.integer_type().is_none() {
            self.get_entry(a)?; // Authenticate physical precision before materialization.
            self.require_ieee_words(self.precision)?;
        }
        // The shader consumes contiguous logical order; preserve the owning
        // adapter until all submissions referencing its physical buffer exist.
        let contiguous = self.reshape_exec(a, a.shape())?;
        if let Some(allocated) = allocated {
            if !runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                &self.get_entry_raw(&contiguous)?.buffer,
                &original.buffer,
            ) {
                allocated.set(true);
            }
        }
        let a = &contiguous;
        if original.integer_type().is_some() {
            return self.integer_reduce_extrema_dim_with_selected_pipeline_exec(
                matches!(op, crate::backend::wgpu::types::DimReduceExtrema::Min),
                dim,
                match op {
                    crate::backend::wgpu::types::DimReduceExtrema::Min => "reduce_min_dim",
                    crate::backend::wgpu::types::DimReduceExtrema::Max => "reduce_max_dim",
                },
                a,
                emission,
            );
        }
        let entry = self.get_entry(a)?;
        let rows = geometry.rows;
        let cols = geometry.cols;
        let reduce_dim = dim + 1;
        let out_len = geometry.length;
        let out_shape = geometry.shape;
        let (values_buffer, indices_buffer) =
            self.with_emission(|emission, pending| -> Result<_> {
                let mut values_buffer = self.create_storage_buffer_checked(
                    emission,
                    out_len,
                    "runmat-reduce-dim-ext-values",
                )?;
                if let Some(allocated) = allocated {
                    allocated.set(true);
                }
                if let Some(after_allocation) = after_allocation {
                    after_allocation(&values_buffer)?;
                }
                let mut indices_buffer = self.create_storage_buffer_checked(
                    emission,
                    out_len,
                    "runmat-reduce-dim-ext-indices",
                )?;
                // Prevent aliasing either output with input buffer
                if std::ptr::eq(values_buffer.as_ref(), entry.buffer.as_ref()) {
                    let size_bytes = (out_len * self.element_size) as u64;
                    if size_bytes > self.adapter_limits.max_buffer_size {
                        return Err(gpu_per_buffer_limit_error(
                            "runmat-reduce-dim-ext-values-unique",
                            size_bytes,
                            self.adapter_limits.max_buffer_size,
                        ));
                    }
                    values_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                        emission.device().create_buffer(&wgpu::BufferDescriptor {
                            label: Some("runmat-reduce-dim-ext-values-unique"),
                            size: size_bytes,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    );
                }
                if std::ptr::eq(indices_buffer.as_ref(), entry.buffer.as_ref()) {
                    let size_bytes = (out_len * self.element_size) as u64;
                    if size_bytes > self.adapter_limits.max_buffer_size {
                        return Err(gpu_per_buffer_limit_error(
                            "runmat-reduce-dim-ext-indices-unique",
                            size_bytes,
                            self.adapter_limits.max_buffer_size,
                        ));
                    }
                    indices_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                        emission.device().create_buffer(&wgpu::BufferDescriptor {
                            label: Some("runmat-reduce-dim-ext-indices-unique"),
                            size: size_bytes,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    );
                }
                if out_len == 0 {
                    return Ok((values_buffer, indices_buffer));
                }
                let params = crate::backend::wgpu::params::ReduceDimParams {
                    rows,
                    cols,
                    dim: reduce_dim as u32,
                    op: op as u32,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-reduce-dim-ext-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduce-dim-ext-bind"),
                        layout: match validated {
                            Some(pipeline) => pipeline.layout.as_ref(),
                            None => &self.pipelines.reduce_dim_minmax.get(emission)?.layout,
                        },
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: values_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: indices_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                crate::backend::wgpu::dispatch::reduction::run_single_pass(
                    emission,
                    pending,
                    match validated {
                        Some(pipeline) => pipeline.pipeline.as_ref(),
                        None => &self.pipelines.reduce_dim_minmax.get(emission)?.pipeline,
                    },
                    &bind_group,
                    geometry.groups,
                )?;
                Ok((values_buffer, indices_buffer))
            })??;
        let values_handle =
            self.register_existing_buffer(values_buffer, out_shape.clone(), out_len)?;
        let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len)?;
        Ok(ReduceDimResult {
            values: values_handle,
            indices: indices_handle,
        })
    }

    pub(super) fn finish_std_from_sums_selected_exec(
        &self,
        sum_handle: GpuTensorHandle,
        sum_sq_handle: GpuTensorHandle,
        sample_count: usize,
        normalization: ProviderStdNormalization,
        nan_shape: &[usize],
        phase: StdFormulaPhase<'_>,
    ) -> Result<GpuTensorHandle> {
        let StdFormulaPhase { selected, effect } = phase;
        let inv_len = 1.0 / (sample_count as f64);

        let sum_squared = self.std_binary_primitive_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_handle,
            &sum_handle,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(sum_handle, selected);

        let sum_shape = self.get_entry(&sum_squared)?.shape.clone();
        let scale_tensor = self.std_fill_primitive_exec(&sum_shape, inv_len, selected, effect)?;
        let sum_squared_scaled = self.std_binary_primitive_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_squared,
            &scale_tensor,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(scale_tensor, selected);
        let variance_numer = self.std_binary_primitive_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &sum_sq_handle,
            &sum_squared_scaled,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(sum_sq_handle, selected);
        super::typed_std::finish_std_temporary(sum_squared, selected);
        super::typed_std::finish_std_temporary(sum_squared_scaled, selected);

        let denom = match normalization {
            ProviderStdNormalization::Sample => {
                if sample_count > 1 {
                    (sample_count - 1) as f64
                } else {
                    1.0
                }
            }
            ProviderStdNormalization::Population => sample_count as f64,
        };
        if denom == 0.0 {
            super::typed_std::finish_std_temporary(variance_numer, selected);
            return self.std_fill_primitive_exec(nan_shape, f64::NAN, selected, effect);
        }

        let variance_shape = self.get_entry(&variance_numer)?.shape.clone();
        let denom_tensor =
            self.std_fill_primitive_exec(&variance_shape, denom, selected, effect)?;
        let variance = self.std_binary_primitive_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &variance_numer,
            &denom_tensor,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(denom_tensor, selected);
        super::typed_std::finish_std_temporary(variance_numer, selected);

        let abs_variance = self.std_unary_primitive_exec(
            crate::backend::wgpu::types::UnaryOpCode::Abs,
            &variance,
            selected,
            effect,
        )?;
        if let Some(selected) = selected {
            selected.track(&abs_variance);
        }
        let variance_plus_abs = self.std_binary_primitive_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(abs_variance, selected);
        super::typed_std::finish_std_temporary(variance, selected);

        let half_shape = self.get_entry(&variance_plus_abs)?.shape.clone();
        let half_tensor = self.std_fill_primitive_exec(&half_shape, 0.5, selected, effect)?;
        let clamped = self.std_binary_primitive_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(half_tensor, selected);
        super::typed_std::finish_std_temporary(variance_plus_abs, selected);

        let std_handle = self.std_unary_primitive_exec(
            crate::backend::wgpu::types::UnaryOpCode::Sqrt,
            &clamped,
            selected,
            effect,
        )?;
        super::typed_std::finish_std_temporary(clamped, selected);
        Ok(std_handle)
    }
}
