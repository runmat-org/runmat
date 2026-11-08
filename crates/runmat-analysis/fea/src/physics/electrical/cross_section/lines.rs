//! Line parameters from the per-length capacitance matrices.
//!
//! - L = μ0 ε0 C0⁻¹ (non-magnetic conductors and dielectrics).
//! - Zc = C^(-1/2) (C^(1/2) L C^(1/2))^(1/2) C^(-1/2), the characteristic impedance matrix of
//!   the lossless multiconductor line; for one conductor Zc = sqrt(L/C).
//! - Per signal conductor (the others held at 0 V): Z0 = Zc_ii, εeff = C_ii/C0_ii and
//!   delay = sqrt(εeff)/c per metre.
//! - A pair, driven differentially (v = [1, −1]) and in common (v = [1, 1]):
//!   Zdiff = vᵀ Zc v, Zcommon = vᵀ Zc v / 4, Zodd = Zdiff/2, Zeven = 2 Zcommon, and modal
//!   εeff = vᵀ C v / vᵀ C0 v. Coupling uses the mutual capacitance Cm = −C12 and inductance
//!   Lm = L12 against the self terms Cs = sqrt(C11 C22), Ls = sqrt(L11 L22):
//!   near-end K_b = (Lm/Ls + Cm/Cs)/4 and far-end K_f = (Cm/Cs − Lm/Ls)/2. A quiet line of
//!   length ℓ sees V_near = K_b ΔV (once the line is longer than half the rise time) and
//!   V_far = K_f (ℓ/v) dV/dt, with v the line's propagation velocity.
use super::{mesh::CrossSectionMeshSummary, solve::Matrix, CrossSectionError};
use runmat_types::physical::{VACUUM_ELECTRIC_PERMITTIVITY, VACUUM_MAGNETIC_PERMEABILITY};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineParameters {
    /// Signal conductors, in the order of the matrix rows.
    pub conductors: Vec<String>,
    /// Maxwell capacitance per length with the dielectrics, F/m.
    pub capacitance_f_per_m: Vec<Vec<f64>>,
    /// Maxwell capacitance per length in vacuum, F/m.
    pub vacuum_capacitance_f_per_m: Vec<Vec<f64>>,
    /// Inductance per length, H/m.
    pub inductance_h_per_m: Vec<Vec<f64>>,
    /// Characteristic impedance matrix, Ω.
    pub characteristic_impedance_ohm: Vec<Vec<f64>>,
    /// Z0 of each signal conductor with the others at 0 V, Ω.
    pub impedance_ohm: Vec<f64>,
    /// εeff of each signal conductor.
    pub effective_permittivity: Vec<f64>,
    /// Propagation delay per length of each signal conductor, s/m.
    pub delay_s_per_m: Vec<f64>,
    /// Present when there are exactly two signal conductors.
    pub pair: Option<DifferentialPairParameters>,
    pub mesh: CrossSectionMeshSummary,
    /// Largest relative residual of the potential solves.
    pub relative_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DifferentialPairParameters {
    pub differential_impedance_ohm: f64,
    pub common_mode_impedance_ohm: f64,
    pub odd_mode_impedance_ohm: f64,
    pub even_mode_impedance_ohm: f64,
    pub odd_mode_effective_permittivity: f64,
    pub even_mode_effective_permittivity: f64,
    pub odd_mode_delay_s_per_m: f64,
    pub even_mode_delay_s_per_m: f64,
    /// Backward (near-end) coupling coefficient K_b.
    pub near_end_coupling: f64,
    /// Forward (far-end) coupling coefficient K_f.
    pub far_end_coupling: f64,
}

pub(super) fn speed_of_light() -> f64 {
    1.0 / (VACUUM_ELECTRIC_PERMITTIVITY.canonical_value()
        * VACUUM_MAGNETIC_PERMEABILITY.canonical_value())
    .sqrt()
}

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}

fn quadratic(a: &Matrix, v: &[f64]) -> f64 {
    let n = a.len();
    (0..n)
        .map(|i| (0..n).map(|j| v[i] * a[i][j] * v[j]).sum::<f64>())
        .sum()
}

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix by cyclic Jacobi rotations.
fn symmetric_eigen(a: &Matrix) -> (Vec<f64>, Matrix) {
    let n = a.len();
    let mut a = a.clone();
    let mut v: Matrix = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let scale: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// f(A) for a symmetric positive definite A.
fn spd_function(a: &Matrix, f: impl Fn(f64) -> f64) -> Result<Matrix, CrossSectionError> {
    let (values, vectors) = symmetric_eigen(a);
    if values.iter().any(|value| !(*value > 0.0)) {
        return Err(CrossSectionError::Operator(
            "a per-length matrix is not positive definite".into(),
        ));
    }
    let n = a.len();
    Ok((0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (0..n)
                        .map(|k| vectors[i][k] * f(values[k]) * vectors[j][k])
                        .sum()
                })
                .collect()
        })
        .collect())
}

impl LineParameters {
    pub(super) fn from_capacitance(
        conductors: Vec<String>,
        capacitance: Vec<Vec<f64>>,
        vacuum: Vec<Vec<f64>>,
        mesh: CrossSectionMeshSummary,
        relative_residual: f64,
    ) -> Result<Self, CrossSectionError> {
        let n = conductors.len();
        let mu_epsilon = VACUUM_ELECTRIC_PERMITTIVITY.canonical_value()
            * VACUUM_MAGNETIC_PERMEABILITY.canonical_value();
        let inverse = spd_function(&vacuum, |x| 1.0 / x)?;
        let inductance: Matrix = inverse
            .iter()
            .map(|row| row.iter().map(|v| mu_epsilon * v).collect())
            .collect();
        let root = spd_function(&capacitance, f64::sqrt)?;
        let inverse_root = spd_function(&capacitance, |x| 1.0 / x.sqrt())?;
        let inner = multiply(&multiply(&root, &inductance), &root);
        // Symmetrize before the eigen-decomposition; the product is symmetric in exact arithmetic.
        let inner: Matrix = (0..n)
            .map(|i| (0..n).map(|j| 0.5 * (inner[i][j] + inner[j][i])).collect())
            .collect();
        let impedance = multiply(
            &multiply(&inverse_root, &spd_function(&inner, f64::sqrt)?),
            &inverse_root,
        );
        let c = speed_of_light();
        let effective: Vec<f64> = (0..n).map(|i| capacitance[i][i] / vacuum[i][i]).collect();
        let pair = (n == 2).then(|| {
            let odd = [1.0, -1.0];
            let even = [1.0, 1.0];
            let differential = quadratic(&impedance, &odd);
            let common = quadratic(&impedance, &even) / 4.0;
            let odd_eff = quadratic(&capacitance, &odd) / quadratic(&vacuum, &odd);
            let even_eff = quadratic(&capacitance, &even) / quadratic(&vacuum, &even);
            let cs = (capacitance[0][0] * capacitance[1][1]).sqrt();
            let ls = (inductance[0][0] * inductance[1][1]).sqrt();
            let cm = -capacitance[0][1];
            let lm = inductance[0][1];
            DifferentialPairParameters {
                differential_impedance_ohm: differential,
                common_mode_impedance_ohm: common,
                odd_mode_impedance_ohm: differential / 2.0,
                even_mode_impedance_ohm: 2.0 * common,
                odd_mode_effective_permittivity: odd_eff,
                even_mode_effective_permittivity: even_eff,
                odd_mode_delay_s_per_m: odd_eff.sqrt() / c,
                even_mode_delay_s_per_m: even_eff.sqrt() / c,
                near_end_coupling: 0.25 * (lm / ls + cm / cs),
                far_end_coupling: 0.5 * (cm / cs - lm / ls),
            }
        });
        Ok(Self {
            conductors,
            impedance_ohm: (0..n).map(|i| impedance[i][i]).collect(),
            delay_s_per_m: effective.iter().map(|e| e.sqrt() / c).collect(),
            effective_permittivity: effective,
            capacitance_f_per_m: capacitance,
            vacuum_capacitance_f_per_m: vacuum,
            inductance_h_per_m: inductance,
            characteristic_impedance_ohm: impedance,
            pair,
            mesh,
            relative_residual,
        })
    }
}
