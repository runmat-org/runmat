use super::*;

impl WgpuProvider {
    fn matrix_dims_for_solve(shape: &[usize]) -> Result<(usize, usize)> {
        match shape.len() {
            0 => Ok((1, 1)),
            1 => Ok((shape[0], 1)),
            2 => Ok((shape[0], shape[1])),
            _ => Err(anyhow!("solve: input must be 2-D")),
        }
    }

    fn triangular_solve_bind_group_layout(&self) -> Arc<wgpu::BindGroupLayout> {
        self.cached_bind_group_layout("runmat-triangular-solve-layout", |device| {
            let entries = [
                crate::backend::wgpu::bindings::storage_read_entry(0),
                crate::backend::wgpu::bindings::storage_read_entry(1),
                crate::backend::wgpu::bindings::storage_read_entry(2),
                crate::backend::wgpu::bindings::storage_read_write_entry(3),
                crate::backend::wgpu::bindings::uniform_entry(4),
            ];
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-triangular-solve-bgl"),
                entries: &entries,
            })
        })
    }

    fn triangular_solve_pipeline(
        &self,
        transpose: bool,
        lower: bool,
    ) -> Arc<wgpu::ComputePipeline> {
        let shader = crate::backend::wgpu::shaders::solve::triangular_linsolve_shader(
            self.precision,
            transpose,
            lower,
        );
        let bind_group_layout = self.triangular_solve_bind_group_layout();
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-triangular-solve-pipeline-layout",
            bind_group_layout.as_ref(),
        );
        let layout_tag = "runmat-triangular-solve-layout";
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            layout_tag,
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-triangular-solve-shader",
            &shader,
        );
        self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-triangular-solve-pipeline",
            Some(shader.as_bytes()),
            Some(layout_tag),
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        )
    }

    fn triangular_solve_bind_group(
        &self,
        bind_group_layout: &wgpu::BindGroupLayout,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        previous: &GpuTensorHandle,
        output: &GpuTensorHandle,
        params_buffer: &wgpu::Buffer,
    ) -> Result<Arc<wgpu::BindGroup>> {
        let lhs_entry = self.get_entry(lhs)?;
        let rhs_entry = self.get_entry(rhs)?;
        let prev_entry = self.get_entry(previous)?;
        let out_entry = self.get_entry(output)?;
        let bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: prev_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        Ok(self
            .bind_group_cache
            .get_or_create(bind_group_layout, &bind_entries, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-triangular-solve-bind-group"),
                            layout: bind_group_layout,
                            entries: &bind_entries,
                        }),
                )
            }))
    }

    pub(super) fn try_triangular_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<Option<ProviderLinsolveResult>> {
        if options.rectangular || options.conjugate || options.symmetric || options.posdef {
            return Ok(None);
        }
        if options.lower == options.upper {
            return Ok(None);
        }

        let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(&lhs.shape)?;
        let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(&rhs.shape)?;
        if lhs_rows != lhs_cols || rhs_rows != lhs_rows {
            return Ok(None);
        }
        if lhs_rows == 0 || rhs_cols == 0 {
            let solution = self.zeros(&rhs.shape)?;
            return Ok(Some(ProviderLinsolveResult {
                solution,
                reciprocal_condition: f64::NAN,
            }));
        }

        let effective_lower = if options.transposed {
            options.upper
        } else {
            options.lower
        };
        let effective_upper = if options.transposed {
            options.lower
        } else {
            options.upper
        };
        if effective_lower == effective_upper {
            return Ok(None);
        }

        let len = rhs_rows
            .checked_mul(rhs_cols)
            .ok_or_else(|| anyhow!("linsolve: rhs dimensions overflow"))?;
        let start = Instant::now();
        let mut current = self.zeros(&rhs.shape)?;
        let mut scratch = self.zeros(&rhs.shape)?;
        let bind_group_layout = self.triangular_solve_bind_group_layout();
        let pipeline = self.triangular_solve_pipeline(options.transposed, effective_lower);
        crate::backend::wgpu::dispatch::solve::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        let mut params = crate::backend::wgpu::params::TriangularSolveParams {
            len: len as u32,
            offset: 0,
            total: len as u32,
            rows: lhs_rows as u32,
            rhs_cols: rhs_cols as u32,
            target_row: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-triangular-solve-params");
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        for step in 0..lhs_rows {
            params.target_row = if effective_lower {
                step
            } else {
                lhs_rows - 1 - step
            } as u32;
            self.queue_ref()
                .write_buffer(&params_buffer, 0, bytes_of(&params));
            let bind_group = self.triangular_solve_bind_group(
                bind_group_layout.as_ref(),
                lhs,
                rhs,
                &current,
                &scratch,
                &params_buffer,
            )?;
            crate::backend::wgpu::dispatch::solve::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                bind_group.as_ref(),
                workgroups,
            );
            std::mem::swap(&mut current, &mut scratch);
        }
        let _ = self.free(&scratch);
        self.telemetry.record_linsolve_duration(start.elapsed());
        let shape = [
            ("rows", lhs_rows as u64),
            ("rhs_cols", rhs_cols as u64),
            ("steps", lhs_rows as u64),
        ];
        let tuning = [
            ("transpose", if options.transposed { 1 } else { 0 }),
            ("lower", if effective_lower { 1 } else { 0 }),
        ];
        self.record_kernel_launch_basic("linsolve_triangular", &shape, &tuning);
        Ok(Some(ProviderLinsolveResult {
            solution: current,
            reciprocal_condition: f64::NAN,
        }))
    }
}
