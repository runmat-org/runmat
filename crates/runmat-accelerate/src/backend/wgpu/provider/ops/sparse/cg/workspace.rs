use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{WgpuBufferOwner, WgpuEmission};

use super::super::super::{NumericPrecision, WgpuProvider};

pub(super) struct CgWorkspace {
    pub(super) solution: WgpuBufferOwner,
    pub(super) residual: WgpuBufferOwner,
    pub(super) preconditioned: WgpuBufferOwner,
    pub(super) direction: WgpuBufferOwner,
    pub(super) product: WgpuBufferOwner,
    pub(super) residual_squared: WgpuBufferOwner,
    pub(super) rho: WgpuBufferOwner,
    pub(super) denominator: WgpuBufferOwner,
    pub(super) dot_partials: WgpuBufferOwner,
    pub(super) state: WgpuBufferOwner,
    pub(super) state_size: u64,
}

impl CgWorkspace {
    pub(super) fn allocate(
        provider: &WgpuProvider,
        emission: &WgpuEmission<'_>,
        dimension: usize,
        reduction_groups: u32,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        max_iterations: u32,
    ) -> anyhow::Result<Self> {
        let vector = |label| provider.create_storage_buffer_checked(emission, dimension, label);
        let scalar = |label| provider.create_storage_buffer_checked(emission, 1, label);
        let state_bytes = state_bytes(
            provider.precision,
            relative_tolerance,
            absolute_tolerance,
            max_iterations,
        );
        let state_size = state_bytes.len() as u64;
        let state = super::super::readback_source(
            provider,
            emission,
            "runmat-sparse-cg-state",
            &state_bytes,
        )?;
        Ok(Self {
            solution: vector("runmat-sparse-cg-solution")?,
            residual: vector("runmat-sparse-cg-residual")?,
            preconditioned: vector("runmat-sparse-cg-preconditioned")?,
            direction: vector("runmat-sparse-cg-direction")?,
            product: vector("runmat-sparse-cg-product")?,
            residual_squared: scalar("runmat-sparse-cg-residual-squared")?,
            rho: scalar("runmat-sparse-cg-rho")?,
            denominator: scalar("runmat-sparse-cg-denominator")?,
            dot_partials: provider.create_storage_buffer_checked(
                emission,
                reduction_groups as usize,
                "runmat-sparse-cg-dot-partials",
            )?,
            state,
            state_size,
        })
    }

    pub(super) fn scratch_bytes(&self) -> u64 {
        self.residual
            .size()
            .saturating_add(self.preconditioned.size())
            .saturating_add(self.direction.size())
            .saturating_add(self.product.size())
            .saturating_add(self.residual_squared.size())
            .saturating_add(self.rho.size())
            .saturating_add(self.denominator.size())
            .saturating_add(self.dot_partials.size())
            .saturating_add(self.state.size())
    }

    pub(super) fn allocated_bytes(&self) -> u64 {
        self.solution.size().saturating_add(self.scratch_bytes())
    }
}

pub(super) struct CgTerminalState {
    pub(super) initial_residual_norm: f64,
    pub(super) final_residual_norm: f64,
    pub(super) iterations: u32,
    pub(super) termination: Option<runmat_accelerate_api::SparseKrylovTermination>,
}

pub(super) fn decode_state(
    precision: NumericPrecision,
    bytes: &[u8],
) -> anyhow::Result<CgTerminalState> {
    let (initial_squared, final_squared, integer_offset) = match precision {
        NumericPrecision::F32 => (
            f64::from(read_f32(bytes, 0)?),
            f64::from(read_f32(bytes, 20)?),
            32,
        ),
        NumericPrecision::F64 => (read_f64(bytes, 0)?, read_f64(bytes, 40)?, 64),
    };
    ensure!(
        initial_squared.is_finite()
            && initial_squared >= 0.0
            && final_squared.is_finite()
            && final_squared >= 0.0,
        "sparse CG returned invalid residual state"
    );
    let iterations = read_u32(bytes, integer_offset)?;
    let termination = match read_u32(bytes, integer_offset + 4)? {
        0 => None,
        1 => Some(runmat_accelerate_api::SparseKrylovTermination::Converged),
        2 => Some(runmat_accelerate_api::SparseKrylovTermination::MaximumIterations),
        3 => Some(runmat_accelerate_api::SparseKrylovTermination::NumericalBreakdown),
        4 => Some(runmat_accelerate_api::SparseKrylovTermination::Cancelled),
        code => {
            return Err(anyhow!(
                "sparse CG returned unknown termination code {code}"
            ))
        }
    };
    Ok(CgTerminalState {
        initial_residual_norm: initial_squared.sqrt(),
        final_residual_norm: final_squared.sqrt(),
        iterations,
        termination,
    })
}

fn state_bytes(
    precision: NumericPrecision,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_iterations: u32,
) -> Vec<u8> {
    match precision {
        NumericPrecision::F32 => {
            let mut bytes = vec![0_u8; 44];
            bytes[24..28].copy_from_slice(&(relative_tolerance as f32).powi(2).to_le_bytes());
            bytes[28..32].copy_from_slice(&(absolute_tolerance as f32).powi(2).to_le_bytes());
            bytes[40..44].copy_from_slice(&max_iterations.to_le_bytes());
            bytes
        }
        NumericPrecision::F64 => {
            let mut bytes = vec![0_u8; 80];
            bytes[48..56].copy_from_slice(&relative_tolerance.powi(2).to_le_bytes());
            bytes[56..64].copy_from_slice(&absolute_tolerance.powi(2).to_le_bytes());
            bytes[72..76].copy_from_slice(&max_iterations.to_le_bytes());
            bytes
        }
    }
}

fn read_f32(bytes: &[u8], offset: usize) -> anyhow::Result<f32> {
    Ok(f32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_f64(bytes: &[u8], offset: usize) -> anyhow::Result<f64> {
    Ok(f64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> anyhow::Result<u32> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> anyhow::Result<[u8; N]> {
    bytes
        .get(offset..offset + N)
        .ok_or_else(|| anyhow!("sparse CG state is truncated"))?
        .try_into()
        .map_err(|_| anyhow!("sparse CG state has an invalid field width"))
}

#[cfg(test)]
mod tests {
    use super::{decode_state, state_bytes};
    use crate::backend::wgpu::types::NumericPrecision;

    #[test]
    fn state_layout_round_trips_terminal_fields() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let mut bytes = state_bytes(precision, 1.0e-4, 1.0e-9, 17);
            let (scalar_size, integer_offset) = match precision {
                NumericPrecision::F32 => (4, 32),
                NumericPrecision::F64 => (8, 64),
            };
            if scalar_size == 4 {
                bytes[0..4].copy_from_slice(&4.0_f32.to_le_bytes());
                bytes[20..24].copy_from_slice(&1.0_f32.to_le_bytes());
            } else {
                bytes[0..8].copy_from_slice(&4.0_f64.to_le_bytes());
                bytes[40..48].copy_from_slice(&1.0_f64.to_le_bytes());
            }
            bytes[integer_offset..integer_offset + 4].copy_from_slice(&7_u32.to_le_bytes());
            bytes[integer_offset + 4..integer_offset + 8].copy_from_slice(&1_u32.to_le_bytes());
            let decoded = decode_state(precision, &bytes).expect("decode state");
            assert_eq!(decoded.initial_residual_norm, 2.0);
            assert_eq!(decoded.final_residual_norm, 1.0);
            assert_eq!(decoded.iterations, 7);
            assert_eq!(
                decoded.termination,
                Some(runmat_accelerate_api::SparseKrylovTermination::Converged)
            );
        }
    }
}
