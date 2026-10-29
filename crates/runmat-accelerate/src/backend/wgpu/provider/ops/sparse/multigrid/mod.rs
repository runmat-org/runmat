//! Device-resident smoothed-aggregation multigrid (single precision) for preconditioning.
//!
//! `prepare` uploads a hierarchy once and records every dispatch of a V-cycle and of one
//! inner conjugate-gradient step as prebuilt commands (pipeline, bind group, grid). An
//! application uploads one residual, replays the commands in one compute pass (no host
//! synchronization, no per-call allocation besides the readback staging buffer) and downloads
//! one correction. Workgroup sizes come from the provider's configured, adapter-clamped
//! linear and reduction sizes; the lanes per CSR row follow each operator's mean row length.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProviderFuture, MultigridCsr, SparseMultigridApply, SparseMultigridHandle,
    SparseMultigridRequest, SparseMultigridSolveRequest, SparseMultigridSolveResult,
    WgpuBufferOwner, WgpuEmission,
};

use super::super::WgpuProvider;

mod kernels;
#[cfg(test)]
mod tests;
use kernels::SpmvMode;

/// Dot-product stage-1 workgroups (a fixed reduction tree for every vector length).
const DOT_GROUPS: u32 = 256;

#[derive(Default)]
pub(in crate::backend::wgpu::provider) struct MultigridState {
    hierarchies: Mutex<HashMap<u64, Arc<Hierarchy>>>,
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
}

struct Command {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    groups: (u32, u32),
}

struct DeviceCsr {
    rows: u32,
    row_offsets: WgpuBufferOwner,
    column_indices: WgpuBufferOwner,
    values: WgpuBufferOwner,
    lanes: u32,
}

struct Hierarchy {
    rows: u32,
    /// Level-0 right-hand side (the uploaded residual; the CG residual during an inner solve).
    input: WgpuBufferOwner,
    /// Level-0 V-cycle output.
    cycle_output: WgpuBufferOwner,
    /// Inner-CG iterate.
    solution: WgpuBufferOwner,
    vcycle: Vec<Command>,
    pcg_init: Vec<Command>,
    pcg_step: Vec<Command>,
    operator: Command,
    /// Buffers referenced only by commands.
    _retained: Vec<WgpuBufferOwner>,
}

/// Builds commands against one device.
struct Builder<'p, 'device> {
    provider: &'p WgpuProvider,
    emission: &'p WgpuEmission<'device>,
    linear: u32,
    reduction: u32,
    retained: Vec<WgpuBufferOwner>,
}

/// Largest power of two not above `value` (at least 1).
fn power_of_two_floor(value: u32) -> u32 {
    if value == 0 {
        1
    } else {
        1 << (31 - value.leading_zeros())
    }
}

impl Builder<'_, '_> {
    fn pipeline(&self, source: String) -> anyhow::Result<Arc<wgpu::ComputePipeline>> {
        let state = &self.provider.sparse.multigrid;
        let mut cache = state.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pipeline) = cache.get(&source) {
            return Ok(pipeline.clone());
        }
        let module = crate::backend::wgpu::shader_compilation::create_shader_module(
            self.emission,
            "runmat-multigrid-kernel",
            &source,
            None,
        )?;
        let pipeline = Arc::new(self.emission.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("runmat-multigrid-pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        cache.insert(source, pipeline.clone());
        Ok(pipeline)
    }

    fn grid(&self, workgroups: u32) -> (u32, u32) {
        let limit = self
            .provider
            .adapter_limits
            .max_compute_workgroups_per_dimension
            .max(1);
        if workgroups <= limit {
            (workgroups.max(1), 1)
        } else {
            (limit, workgroups.div_ceil(limit))
        }
    }

    fn command(
        &self,
        source: String,
        buffers: &[&wgpu::Buffer],
        workgroups: u32,
    ) -> anyhow::Result<Command> {
        let pipeline = self.pipeline(source)?;
        let layout = pipeline.get_bind_group_layout(0);
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self
            .emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-multigrid-bind-group"),
                layout: &layout,
                entries: &entries,
            });
        Ok(Command {
            pipeline,
            bind_group,
            groups: self.grid(workgroups),
        })
    }

    fn read_only(&mut self, label: &str, bytes: &[u8]) -> anyhow::Result<WgpuBufferOwner> {
        super::upload_operand(self.provider, self.emission, label, bytes, 4)
    }

    fn vector(&self, label: &str, len: u32) -> WgpuBufferOwner {
        WgpuBufferOwner::new(
            self.emission
                .device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: u64::from(len.max(1)) * 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
        )
    }

    fn params(&mut self, values: &[f32]) -> anyhow::Result<WgpuBufferOwner> {
        let buffer = self.read_only("runmat-multigrid-params", bytemuck::cast_slice(values))?;
        self.retained.push(buffer.clone());
        Ok(buffer)
    }

    fn csr(&mut self, matrix: &MultigridCsr<'_>) -> anyhow::Result<DeviceCsr> {
        let mean = matrix.values.len() as u32 / matrix.rows.max(1);
        let lanes = power_of_two_floor((mean / 2).max(1))
            .min(32)
            .min(self.reduction);
        Ok(DeviceCsr {
            rows: matrix.rows,
            row_offsets: self.read_only(
                "runmat-multigrid-rows",
                bytemuck::cast_slice(matrix.row_offsets),
            )?,
            column_indices: self.read_only(
                "runmat-multigrid-columns",
                bytemuck::cast_slice(matrix.column_indices),
            )?,
            values: self.read_only(
                "runmat-multigrid-values",
                bytemuck::cast_slice(matrix.values),
            )?,
            lanes,
        })
    }

    fn spmv(
        &self,
        m: &DeviceCsr,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        b: Option<&wgpu::Buffer>,
        mode: SpmvMode,
    ) -> anyhow::Result<Command> {
        let rows_per_group = self.reduction / m.lanes;
        let workgroups = m.rows.div_ceil(rows_per_group);
        let mut buffers = vec![
            m.row_offsets.as_ref(),
            m.column_indices.as_ref(),
            m.values.as_ref(),
            x,
            y,
        ];
        if let Some(b) = b {
            buffers.push(b);
        }
        self.command(
            kernels::spmv(self.reduction, m.lanes, mode),
            &buffers,
            workgroups,
        )
    }

    fn elementwise(
        &self,
        source: String,
        buffers: &[&wgpu::Buffer],
        len: u32,
    ) -> anyhow::Result<Command> {
        self.command(source, buffers, len.div_ceil(self.linear))
    }
}

struct DeviceLevel {
    matrix: DeviceCsr,
    block: u32,
    inverse: WgpuBufferOwner,
    lambda_max: f32,
    prolongation: Option<DeviceCsr>,
    restriction: Option<DeviceCsr>,
    x: WgpuBufferOwner,
    b: WgpuBufferOwner,
    r: WgpuBufferOwner,
    d: WgpuBufferOwner,
    z: WgpuBufferOwner,
}

/// Chebyshev smoothing of `x` toward `A x = b` (the CPU smoother's recurrence).
fn smooth(
    builder: &mut Builder<'_, '_>,
    l: &DeviceLevel,
    degree: u32,
    ratio: f32,
    out: &mut Vec<Command>,
) -> anyhow::Result<()> {
    let n = l.matrix.rows;
    let upper = l.lambda_max;
    let lower = upper / ratio;
    let (theta, delta) = (0.5 * (upper + lower), 0.5 * (upper - lower));
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    out.push(builder.spmv(&l.matrix, &l.x, &l.r, Some(&l.b), SpmvMode::Residual)?);
    let scale = builder.params(&[1.0 / theta])?;
    out.push(builder.elementwise(
        kernels::block_inverse(builder.linear, l.block),
        &[&l.inverse, &l.r, &l.d, &scale],
        n,
    )?);
    let one = builder.params(&[1.0, 1.0])?;
    let unit = builder.params(&[1.0])?;
    for k in 0..degree.max(1) {
        out.push(builder.elementwise(kernels::axpby(builder.linear), &[&l.d, &l.x, &one], n)?);
        if k + 1 == degree.max(1) {
            break;
        }
        out.push(builder.spmv(&l.matrix, &l.d, &l.r, None, SpmvMode::Subtract)?);
        out.push(builder.elementwise(
            kernels::block_inverse(builder.linear, l.block),
            &[&l.inverse, &l.r, &l.z, &unit],
            n,
        )?);
        let rho_next = 1.0 / (2.0 * sigma - rho);
        let coefficients = builder.params(&[2.0 * rho_next / delta, rho_next * rho])?;
        out.push(builder.elementwise(
            kernels::axpby(builder.linear),
            &[&l.z, &l.d, &coefficients],
            n,
        )?);
        rho = rho_next;
    }
    Ok(())
}

fn vcycle(
    builder: &mut Builder<'_, '_>,
    levels: &[DeviceLevel],
    coarse: &WgpuBufferOwner,
    degree: u32,
    ratio: f32,
    out: &mut Vec<Command>,
) -> anyhow::Result<()> {
    let l = &levels[0];
    if levels.len() == 1 {
        let workgroups = l.matrix.rows;
        out.push(builder.command(
            kernels::dense_matvec(builder.reduction),
            &[coarse, &l.b, &l.x],
            workgroups,
        )?);
        return Ok(());
    }
    let n = l.matrix.rows;
    out.push(builder.elementwise(kernels::zero(builder.linear), &[&l.x], n)?);
    smooth(builder, l, degree, ratio, out)?;
    out.push(builder.spmv(&l.matrix, &l.x, &l.r, Some(&l.b), SpmvMode::Residual)?);
    let next = &levels[1];
    out.push(builder.spmv(
        l.restriction.as_ref().unwrap(),
        &l.r,
        &next.b,
        None,
        SpmvMode::Assign,
    )?);
    vcycle(builder, &levels[1..], coarse, degree, ratio, out)?;
    out.push(builder.spmv(
        l.prolongation.as_ref().unwrap(),
        &next.x,
        &l.x,
        None,
        SpmvMode::Add,
    )?);
    smooth(builder, l, degree, ratio, out)
}

fn dot(
    builder: &Builder<'_, '_>,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    partials: &wgpu::Buffer,
    scalars: &wgpu::Buffer,
    slot: u32,
    out: &mut Vec<Command>,
) -> anyhow::Result<()> {
    out.push(builder.command(
        kernels::dot_partials(builder.reduction, DOT_GROUPS),
        &[a, b, partials],
        DOT_GROUPS,
    )?);
    out.push(builder.command(
        kernels::dot_finish(builder.reduction, slot),
        &[partials, scalars],
        1,
    )?);
    Ok(())
}

pub(in crate::backend::wgpu::provider) fn prepare(
    provider: &WgpuProvider,
    request: &SparseMultigridRequest<'_>,
) -> anyhow::Result<SparseMultigridHandle> {
    request.validate()?;
    ensure!(
        matches!(provider.precision, super::super::NumericPrecision::F32),
        "device multigrid runs in single precision"
    );
    let hierarchy =
        provider.with_emission(|emission, _pending| -> anyhow::Result<Hierarchy> {
            emission.ensure_device(&provider.access)?;
            let mut builder = Builder {
                provider,
                emission,
                linear: power_of_two_floor(provider.workgroup_config.scalar),
                reduction: power_of_two_floor(provider.workgroup_config.reduction_default),
                retained: Vec::new(),
            };
            let mut levels = Vec::with_capacity(request.levels.len());
            for level in &request.levels {
                let n = level.matrix.rows;
                levels.push(DeviceLevel {
                    matrix: builder.csr(&level.matrix)?,
                    block: level.block,
                    inverse: builder.read_only(
                        "runmat-multigrid-inverse",
                        bytemuck::cast_slice(level.inverse_blocks),
                    )?,
                    lambda_max: level.lambda_max,
                    prolongation: level
                        .prolongation
                        .as_ref()
                        .map(|p| builder.csr(p))
                        .transpose()?,
                    restriction: level
                        .restriction
                        .as_ref()
                        .map(|r| builder.csr(r))
                        .transpose()?,
                    x: builder.vector("runmat-multigrid-x", n),
                    b: builder.vector("runmat-multigrid-b", n),
                    r: builder.vector("runmat-multigrid-r", n),
                    d: builder.vector("runmat-multigrid-d", n),
                    z: builder.vector("runmat-multigrid-z", n),
                });
            }
            let coarse = builder.read_only(
                "runmat-multigrid-coarse",
                bytemuck::cast_slice(request.coarse_inverse),
            )?;
            let mut cycle = Vec::new();
            vcycle(
                &mut builder,
                &levels,
                &coarse,
                request.smoother_degree,
                request.chebyshev_ratio,
                &mut cycle,
            )?;
            // Inner CG on level 0, preconditioned by the cycle: r lives in level-0 `b`, z in `x`.
            let fine = &levels[0];
            let n = fine.matrix.rows;
            let solution = builder.vector("runmat-multigrid-solution", n);
            let direction = builder.vector("runmat-multigrid-p", n);
            let product = builder.vector("runmat-multigrid-q", n);
            let partials = builder.vector("runmat-multigrid-partials", DOT_GROUPS);
            let scalars = builder.vector("runmat-multigrid-scalars", 8);
            let mut init =
                vec![builder.elementwise(kernels::zero(builder.linear), &[&solution], n)?];
            let cycle_again = |builder: &mut Builder<'_, '_>, out: &mut Vec<Command>| {
                vcycle(
                    builder,
                    &levels,
                    &coarse,
                    request.smoother_degree,
                    request.chebyshev_ratio,
                    out,
                )
            };
            cycle_again(&mut builder, &mut init)?;
            init.push(builder.elementwise(
                kernels::copy(builder.linear),
                &[&fine.x, &direction],
                n,
            )?);
            dot(
                &builder, &fine.b, &fine.x, &partials, &scalars, 0, &mut init,
            )?;
            let mut step =
                vec![builder.spmv(&fine.matrix, &direction, &product, None, SpmvMode::Assign)?];
            dot(
                &builder, &direction, &product, &partials, &scalars, 1, &mut step,
            )?;
            step.push(builder.command(kernels::cg_scalars(0), &[&scalars], 1)?);
            step.push(builder.elementwise(
                kernels::cg_update(builder.linear),
                &[&direction, &product, &solution, &fine.b, &scalars],
                n,
            )?);
            cycle_again(&mut builder, &mut step)?;
            dot(
                &builder, &fine.b, &fine.x, &partials, &scalars, 2, &mut step,
            )?;
            step.push(builder.command(kernels::cg_scalars(1), &[&scalars], 1)?);
            step.push(builder.elementwise(
                kernels::cg_direction(builder.linear),
                &[&fine.x, &direction, &scalars],
                n,
            )?);
            let operator = builder.spmv(&fine.matrix, &fine.b, &fine.x, None, SpmvMode::Assign)?;
            let mut retained = std::mem::take(&mut builder.retained);
            retained.extend([coarse, direction, product, partials, scalars]);
            let (input, cycle_output) = (fine.b.clone(), fine.x.clone());
            for level in levels {
                retained.extend([level.inverse, level.x, level.b, level.r, level.d, level.z]);
                for m in std::iter::once(level.matrix)
                    .chain(level.prolongation)
                    .chain(level.restriction)
                {
                    retained.extend([m.row_offsets, m.column_indices, m.values]);
                }
            }
            Ok(Hierarchy {
                rows: n,
                input,
                cycle_output,
                solution,
                vcycle: cycle,
                pcg_init: init,
                pcg_step: step,
                operator,
                _retained: retained,
            })
        })??;
    let id = provider.sparse.next_id();
    let rows = hierarchy.rows;
    provider
        .sparse
        .multigrid
        .hierarchies
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, Arc::new(hierarchy));
    Ok(SparseMultigridHandle {
        device_id: provider.runtime_device_id,
        hierarchy_id: id,
        rows,
    })
}

fn hierarchy(
    provider: &WgpuProvider,
    handle: &SparseMultigridHandle,
) -> anyhow::Result<Arc<Hierarchy>> {
    ensure!(
        handle.device_id == provider.runtime_device_id,
        "multigrid handle belongs to another device"
    );
    provider
        .sparse
        .multigrid
        .hierarchies
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle.hierarchy_id)
        .cloned()
        .ok_or_else(|| anyhow!("unknown multigrid hierarchy"))
}

pub(in crate::backend::wgpu::provider) fn solve<'a>(
    provider: &'a WgpuProvider,
    request: &'a SparseMultigridSolveRequest<'a>,
) -> AccelProviderFuture<'a, SparseMultigridSolveResult> {
    Box::pin(async move {
        let h = hierarchy(provider, request.hierarchy)?;
        ensure!(
            request.rhs.len() == h.rows as usize,
            "multigrid right-hand side length"
        );
        let (staging, dispatches) =
            provider.with_emission(|emission, pending| -> anyhow::Result<_> {
                emission.ensure_device(&provider.access)?;
                emission.write_buffer(&h.input, 0, bytemuck::cast_slice(request.rhs));
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-multigrid-apply"),
                    },
                );
                let mut dispatches = 0u32;
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("runmat-multigrid-pass"),
                        timestamp_writes: None,
                    });
                    let mut run = |commands: &[Command]| {
                        for c in commands {
                            pass.set_pipeline(&c.pipeline);
                            pass.set_bind_group(0, &c.bind_group, &[]);
                            pass.dispatch_workgroups(c.groups.0, c.groups.1, 1);
                            dispatches += 1;
                        }
                    };
                    match request.apply {
                        SparseMultigridApply::VCycle => run(&h.vcycle),
                        SparseMultigridApply::Pcg { iterations } => {
                            run(&h.pcg_init);
                            for _ in 0..iterations {
                                run(&h.pcg_step);
                            }
                        }
                        SparseMultigridApply::Operator { repetitions } => {
                            for _ in 0..repetitions.max(1) {
                                run(std::slice::from_ref(&h.operator));
                            }
                        }
                    }
                }
                let result = match request.apply {
                    SparseMultigridApply::VCycle | SparseMultigridApply::Operator { .. } => {
                        &h.cycle_output
                    }
                    SparseMultigridApply::Pcg { .. } => &h.solution,
                };
                let bytes = u64::from(h.rows) * 4;
                let staging = super::stage_readback(
                    emission,
                    &mut encoder,
                    result,
                    bytes,
                    "runmat-multigrid-readback",
                );
                provider.submit(emission, pending, encoder)?;
                Ok((staging, dispatches))
            })??;
        let solution = provider
            .map_readback_with(staging, "sparse multigrid", |bytes| {
                Ok(bytemuck::pod_collect_to_vec::<u8, f32>(bytes))
            })
            .await?;
        Ok(SparseMultigridSolveResult {
            solution,
            dispatches,
        })
    })
}

pub(in crate::backend::wgpu::provider) fn free(
    provider: &WgpuProvider,
    handle: &SparseMultigridHandle,
) -> anyhow::Result<()> {
    ensure!(
        handle.device_id == provider.runtime_device_id,
        "multigrid handle belongs to another device"
    );
    provider
        .sparse
        .multigrid
        .hierarchies
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&handle.hierarchy_id)
        .map(|_| ())
        .ok_or_else(|| anyhow!("unknown multigrid hierarchy"))
}
