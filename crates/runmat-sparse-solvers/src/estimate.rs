//! Memory estimates for admission before any solver work starts.
//!
//! Direct factorizations quote exact sizes from their symbolic analysis
//! ([`crate::direct::SymbolicFactorization::estimate`]); this module provides closed-form figures
//! for iterative workspaces and multigrid hierarchies, and a quick pre-analysis direct estimate
//! (calibrated on 3D finite element meshes) used by the automatic policy to decide whether a
//! symbolic analysis is worth computing.

use crate::krylov::{
    bicgstab_workspace_len, gmres_workspace_len, minres_workspace_len, pcg_workspace_len,
    GmresOptions, KrylovOptions,
};

/// Krylov method selector for estimates and policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KrylovMethod {
    Pcg,
    Minres,
    Gmres { restart: usize },
    Bicgstab,
}

/// Workspace bytes of a Krylov solve (vectors only; operator and preconditioner excluded).
pub fn krylov_workspace_bytes(method: KrylovMethod, n: usize, scalar_bytes: usize) -> u64 {
    let options = KrylovOptions::default();
    let entries = match method {
        KrylovMethod::Pcg => pcg_workspace_len(n, &options),
        KrylovMethod::Minres => minres_workspace_len(n, &options),
        KrylovMethod::Gmres { restart } => gmres_workspace_len(
            n,
            &options,
            &GmresOptions {
                restart,
                ..Default::default()
            },
        ),
        KrylovMethod::Bicgstab => bicgstab_workspace_len(n, &options),
    };
    // Solution and right-hand side vectors held by the caller are not counted.
    (entries as u64) * scalar_bytes as u64
}

/// Upper estimate of a smoothed-aggregation hierarchy: operator complexity ≤ 1.6 for scalar
/// problems and ≤ 2.2 with rigid-body modes, plus prolongators of comparable size and level
/// vectors. Includes the copy of the fine operator kept by the hierarchy.
pub fn amg_bytes(n: usize, nnz: usize, block: usize) -> u64 {
    let complexity = if block > 1 { 2.2 } else { 1.6 };
    let csr_entry = 16.0; // f64 value + usize index
    let operators = complexity * nnz as f64 * csr_entry;
    let prolongators = 0.8 * nnz as f64 * csr_entry;
    let vectors = 12.0 * n as f64 * 8.0;
    (operators + prolongators + vectors) as u64
}

/// ILU(0) storage: a copy of the matrix plus the diagonal map.
pub fn ilu0_bytes(n: usize, nnz: usize, scalar_bytes: usize) -> u64 {
    (nnz * (scalar_bytes + 8) + (n + 1) * 16) as u64
}

/// Quick factor-size estimate before ordering, from `n`, `nnz` and the spatial dimension of the
/// discretization. Nested dissection gives `O(n log n)` fill in 2D and `O(n^{4/3})` in 3D; the
/// constants were calibrated on the structured 3D Poisson and elasticity benchmarks (factor
/// entries ≈ 1.5 · (nnz/n)^{1/2} · n^{4/3}). Returns `(factor_entries, flops)`.
pub fn quick_direct_estimate(n: usize, nnz: usize, dimension: u8) -> (f64, f64) {
    let n_f = n.max(1) as f64;
    let density = (nnz as f64 / n_f).max(1.0);
    match dimension {
        0..=2 => {
            let entries = 3.0 * density.sqrt() * n_f * n_f.log2().max(1.0);
            (entries, entries * n_f.sqrt() * 2.0)
        }
        _ => {
            let entries = 1.5 * density.sqrt() * n_f.powf(4.0 / 3.0);
            (entries, 1.0 * density * n_f * n_f)
        }
    }
}

/// Physical memory of the host in bytes (`None` when it cannot be determined).
pub fn host_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        return String::from_utf8(output.stdout).ok()?.trim().parse().ok();
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kib * 1024);
    }
    #[allow(unreachable_code)]
    None
}

/// Planning reservation for a direct factorization known only by counts (no pattern yet):
/// twice the nested-dissection fill model for a 3D discretization, factor values and indices,
/// plus workspace. `general` doubles the factor for LU (`L` and `U`). After assembly the exact
/// symbolic estimate replaces this figure.
pub fn direct_reservation_bytes(n: usize, nnz: usize, general: bool) -> u64 {
    let (entries, _) = quick_direct_estimate(n, nnz, 3);
    let factor = if general { 2.0 * entries } else { entries };
    let bytes = 2.0 * factor * 16.0 + (n as f64) * 128.0 + (nnz as f64) * 24.0;
    if bytes.is_finite() {
        bytes.min(u64::MAX as f64) as u64
    } else {
        u64::MAX
    }
}
