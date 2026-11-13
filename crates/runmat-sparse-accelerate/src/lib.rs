//! Device preconditioners for the shared sparse solvers.
//!
//! The solver crate stays value-free and device-free; the accelerator provider (runmat-accelerate's
//! wgpu provider, the single owner of the device, pipelines and residency) implements the
//! value-free multigrid contract of `runmat-accelerate-api`. This adapter joins the two: it exports
//! a host smoothed-aggregation hierarchy in single precision, uploads it once, and applies it as a
//! [`Preconditioner<f64>`] — one upload of the residual and one download of the correction per
//! application, everything else resident on the device.
//!
//! Precision: the device works in `f32`. The outer Krylov loop stays `f64` on the host and certifies
//! convergence on its own `f64` true residual, so accuracy guarantees are unchanged; a device
//! inner solve ([`DeviceApply::InnerCg`]) is a variable preconditioner and the drivers select
//! flexible CG / flexible GMRES for it.
//!
//! Browser (wasm) builds have no device path: callers solve on the host there.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

use runmat_accelerate_api::{
    AccelProvider, MultigridCsr, MultigridLevel, SparseMultigridApply, SparseMultigridHandle,
    SparseMultigridRequest, SparseMultigridSolveRequest,
};
use runmat_sparse_solvers::precond::{Amg, AmgExport};
use runmat_sparse_solvers::{CsrMatrix, Preconditioner, SolverError, SolverResult};

/// What one application runs on the device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceApply {
    /// One V-cycle (a fixed linear map up to `f32` rounding).
    VCycle,
    /// `iterations` steps of device CG preconditioned by the V-cycle (variable).
    InnerCg { iterations: u32 },
}

/// A smoothed-aggregation hierarchy resident on an accelerator.
pub struct DeviceAmg<'p> {
    provider: &'p dyn AccelProvider,
    handle: SparseMultigridHandle,
    apply: DeviceApply,
}

struct SinglePrecisionCsr {
    rows: u32,
    columns: u32,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
    values: Vec<f32>,
}

impl SinglePrecisionCsr {
    fn of(matrix: &CsrMatrix<f64>) -> SolverResult<Self> {
        let narrow = |value: usize| {
            u32::try_from(value)
                .map_err(|_| SolverError::Unsupported("device index exceeds 32 bits".into()))
        };
        let a = matrix.as_ref();
        Ok(Self {
            rows: narrow(a.nrows())?,
            columns: narrow(a.ncols())?,
            row_offsets: a
                .row_ptr()
                .iter()
                .map(|&p| narrow(p))
                .collect::<SolverResult<_>>()?,
            column_indices: a
                .col_idx()
                .iter()
                .map(|&c| narrow(c))
                .collect::<SolverResult<_>>()?,
            values: a.values().iter().map(|&v| v as f32).collect(),
        })
    }
    fn view(&self) -> MultigridCsr<'_> {
        MultigridCsr {
            rows: self.rows,
            columns: self.columns,
            row_offsets: &self.row_offsets,
            column_indices: &self.column_indices,
            values: &self.values,
        }
    }
}

fn device_error(error: impl std::fmt::Display) -> SolverError {
    SolverError::Operator(format!("device multigrid: {error}"))
}

impl<'p> DeviceAmg<'p> {
    /// Uploads `amg` (exported in single precision) to `provider`.
    pub fn new(
        provider: &'p dyn AccelProvider,
        amg: &Amg,
        apply: DeviceApply,
    ) -> SolverResult<Self> {
        let export = amg.export()?;
        Self::from_export(provider, &export, apply)
    }

    pub fn from_export(
        provider: &'p dyn AccelProvider,
        export: &AmgExport,
        apply: DeviceApply,
    ) -> SolverResult<Self> {
        let coarse_inverse: Vec<f32> = export
            .coarse_inverse
            .as_ref()
            .ok_or_else(|| {
                SolverError::Unsupported("device multigrid needs an exact coarse solve".into())
            })?
            .iter()
            .map(|&v| v as f32)
            .collect();
        let matrices = export
            .levels
            .iter()
            .map(|level| {
                Ok((
                    SinglePrecisionCsr::of(&level.matrix)?,
                    level
                        .prolongation
                        .as_ref()
                        .map(SinglePrecisionCsr::of)
                        .transpose()?,
                    level
                        .restriction
                        .as_ref()
                        .map(SinglePrecisionCsr::of)
                        .transpose()?,
                    level
                        .inverse_blocks
                        .iter()
                        .map(|&v| v as f32)
                        .collect::<Vec<f32>>(),
                ))
            })
            .collect::<SolverResult<Vec<_>>>()?;
        let levels = export
            .levels
            .iter()
            .zip(&matrices)
            .map(|(level, (a, p, r, inverse))| MultigridLevel {
                matrix: a.view(),
                block: level.block as u32,
                inverse_blocks: inverse,
                lambda_max: level.lambda_max as f32,
                prolongation: p.as_ref().map(SinglePrecisionCsr::view),
                restriction: r.as_ref().map(SinglePrecisionCsr::view),
            })
            .collect();
        let request = SparseMultigridRequest {
            levels,
            coarse_inverse: &coarse_inverse,
            smoother_degree: export.smoother_degree as u32,
            chebyshev_ratio: export.chebyshev_ratio as f32,
        };
        let handle = provider
            .prepare_sparse_multigrid(&request)
            .map_err(device_error)?;
        Ok(Self {
            provider,
            handle,
            apply,
        })
    }

    /// One device application in single precision (also the benchmark entry point).
    pub fn apply_f32(&self, rhs: &[f32], apply: SparseMultigridApply) -> SolverResult<Vec<f32>> {
        let request = SparseMultigridSolveRequest {
            hierarchy: &self.handle,
            rhs,
            apply,
        };
        let result = pollster::block_on(self.provider.sparse_multigrid_solve(&request))
            .map_err(device_error)?;
        Ok(result.solution)
    }

    fn device_apply(&self) -> SparseMultigridApply {
        match self.apply {
            DeviceApply::VCycle => SparseMultigridApply::VCycle,
            DeviceApply::InnerCg { iterations } => SparseMultigridApply::Pcg { iterations },
        }
    }
}

// Browser providers are not `Sync`; device preconditioning is native-only (wasm uses the CPU path).
#[cfg(not(target_arch = "wasm32"))]
impl Preconditioner<f64> for DeviceAmg<'_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        if r.len() != self.handle.rows as usize || z.len() != r.len() {
            return Err(SolverError::Dimension(
                "device multigrid vector length".into(),
            ));
        }
        // Scale into f32 range by the residual's largest magnitude; the map is linear (V-cycle)
        // or scale-equivariant (CG), so the correction is scaled back exactly.
        let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if scale == 0.0 || !scale.is_finite() {
            z.iter_mut().for_each(|v| *v = 0.0);
            return if scale == 0.0 {
                Ok(())
            } else {
                Err(SolverError::NonFinite { entry: 0 })
            };
        }
        let rhs: Vec<f32> = r.iter().map(|&v| (v / scale) as f32).collect();
        let solution = self.apply_f32(&rhs, self.device_apply())?;
        for (z, &v) in z.iter_mut().zip(&solution) {
            *z = f64::from(v) * scale;
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Err(SolverError::Breakdown(
                "device multigrid produced a nonfinite correction".into(),
            ));
        }
        Ok(())
    }
    fn label(&self) -> &'static str {
        match self.apply {
            DeviceApply::VCycle => "device-amg-vcycle",
            DeviceApply::InnerCg { .. } => "device-amg-inner-cg",
        }
    }
    fn is_variable(&self) -> bool {
        matches!(self.apply, DeviceApply::InnerCg { .. })
    }
}

impl Drop for DeviceAmg<'_> {
    fn drop(&mut self) {
        let _ = self.provider.free_sparse_multigrid(&self.handle);
    }
}
