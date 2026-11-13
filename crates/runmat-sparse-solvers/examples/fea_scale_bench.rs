//! FEA-scale solver benchmark: `fea_scale_bench <section> <size> <threads>`.
//!
//! Sections (one JSON object per measured phase on stdout):
//! - `kernels <grid>`: CSR and 3×3 BSR SpMV of the elasticity operator, dot, axpy and a
//!   STREAM-style triad, with achieved memory bandwidth (GB/s);
//! - `elasticity <grid>`: Q1-hexahedron elasticity (3 dof/node, ~81 nnz/row): AMG setup, AMG-PCG
//!   solve with per-iteration time, V-cycle apply time;
//! - `direct <grid>`: the same operator through supernodal Cholesky (analyze, factor, solve);
//! - `cfd <grid>`: 3D convection–diffusion (momentum-like, nonsymmetric) with GMRES(30)+ILU0 and
//!   GMRES(30)+AMG, and the pressure Poisson operator with AMG-PCG;
//! - `modal <grid>`: 10 lowest modes of the elasticity operator with a lumped mass (shift-invert
//!   block Lanczos).
//!
//! Work runs on an admitted pool of `<threads>` workers. Run through `gate bench` for timings.
//! Peak RSS is process-lifetime evidence; unavailable platform metrics are JSON null.

use std::time::Instant;

#[path = "support/memory_report.rs"]
mod memory_report;

use memory_report::peak_rss_gib;

use runmat_sparse_solvers::control::SolveControl;
use runmat_sparse_solvers::direct::{
    DirectOptions, DirectSolver, FactorKind, SymbolicFactorization,
};
use runmat_sparse_solvers::eigen::{modal, EigenOptions};
use runmat_sparse_solvers::krylov::{gmres, pcg, GmresOptions, KrylovOptions};
use runmat_sparse_solvers::ordering::Ordering;
use runmat_sparse_solvers::par;
use runmat_sparse_solvers::precond::{rigid_body_modes, Amg, AmgOptions, Ilu0};
use runmat_sparse_solvers::testutil::{elasticity::elasticity3d, poisson3d};
use runmat_sparse_solvers::{BsrMatrix, CsrMatrix, Preconditioner, TripletBuilder};

/// Median wall time of `repeats` runs of `work`.
fn median_seconds(repeats: usize, mut work: impl FnMut()) -> f64 {
    let mut samples: Vec<f64> = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Upwinded 7-point convection–diffusion on an `m³` grid (Péclet ≈ 1 per cell).
fn convection3d(m: usize) -> CsrMatrix<f64> {
    let n = m * m * m;
    let mut t = TripletBuilder::new(n, n);
    let id = |i: usize, j: usize, k: usize| (i * m + j) * m + k;
    let wind = [0.8, 0.4, 0.2];
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                let row = id(i, j, k);
                t.push(row, row, 6.0 + wind.iter().sum::<f64>());
                let mut neighbor = |ok: bool, col: usize, value: f64| {
                    if ok {
                        t.push(row, col, value);
                    }
                };
                neighbor(i > 0, id(i.wrapping_sub(1), j, k), -1.0 - wind[0]);
                neighbor(i + 1 < m, id(i + 1, j, k), -1.0);
                neighbor(j > 0, id(i, j.wrapping_sub(1), k), -1.0 - wind[1]);
                neighbor(j + 1 < m, id(i, j + 1, k), -1.0);
                neighbor(k > 0, id(i, j, k.wrapping_sub(1)), -1.0 - wind[2]);
                neighbor(k + 1 < m, id(i, j, k + 1), -1.0);
            }
        }
    }
    t.build()
}

fn relative_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.as_ref().spmv(x, &mut ax);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
    par::norm2(&r) / par::norm2(b)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let section = args.get(1).map(String::as_str).unwrap_or("kernels");
    let grid: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(30);
    let threads: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let gb = |bytes: f64, seconds: f64| bytes / seconds / 1e9;
    par::with_threads(threads, || match section {
        "kernels" => {
            let p = elasticity3d(grid, grid, grid);
            let a = p.matrix;
            let (n, nnz) = (a.nrows(), a.nnz());
            let x = par::seeded_vector::<f64>(n, 1);
            let mut y = vec![0.0; n];
            let repeats = 15;
            let csr = median_seconds(repeats, || a.as_ref().spmv(&x, &mut y));
            // values + column indices + row pointers + x (read once, cache permitting) + y.
            let csr_bytes = (nnz * 16 + (n + 1) * 8 + 2 * n * 8) as f64;
            println!("{{\"section\":\"kernels\",\"kernel\":\"csr_spmv\",\"n\":{n},\"nnz\":{nnz},\"threads\":{threads},\"seconds\":{csr:.6},\"gb_per_s\":{:.1}}}", gb(csr_bytes, csr));
            let bsr = BsrMatrix::from_csr(a.as_ref(), 3).unwrap();
            let blocks = nnz / 9;
            let bsr_t = median_seconds(repeats, || bsr.spmv(&x, &mut y));
            let bsr_bytes = (nnz * 8 + blocks * 8 + (n / 3 + 1) * 8 + 2 * n * 8) as f64;
            println!("{{\"section\":\"kernels\",\"kernel\":\"bsr3_spmv\",\"n\":{n},\"nnz\":{nnz},\"threads\":{threads},\"seconds\":{bsr_t:.6},\"gb_per_s\":{:.1}}}", gb(bsr_bytes, bsr_t));
            let big = 1usize << 25; // 32 Mi doubles = 256 MiB per vector, beyond every cache
            let u = par::seeded_vector::<f64>(big, 2);
            let mut v = par::seeded_vector::<f64>(big, 3);
            let mut w = vec![0.0; big];
            let mut sink = 0.0;
            let dot = median_seconds(repeats, || sink += par::dot(&u, &v));
            println!("{{\"section\":\"kernels\",\"kernel\":\"dot\",\"n\":{big},\"threads\":{threads},\"seconds\":{dot:.6},\"gb_per_s\":{:.1}}}", gb(16.0 * big as f64, dot));
            let axpy = median_seconds(repeats, || par::axpy(1e-9, &u, &mut v));
            println!("{{\"section\":\"kernels\",\"kernel\":\"axpy\",\"n\":{big},\"threads\":{threads},\"seconds\":{axpy:.6},\"gb_per_s\":{:.1}}}", gb(24.0 * big as f64, axpy));
            let triad = median_seconds(repeats, || {
                par::for_each_chunk_mut(&mut w, |start, chunk| {
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        *value = u[start + offset] + 0.5 * v[start + offset];
                    }
                })
            });
            println!("{{\"section\":\"kernels\",\"kernel\":\"triad\",\"n\":{big},\"threads\":{threads},\"seconds\":{triad:.6},\"gb_per_s\":{:.1},\"sink\":{}}}", gb(24.0 * big as f64, triad), sink.is_finite());
        }
        "elasticity" => {
            let t = Instant::now();
            let p = elasticity3d(grid, grid, grid);
            let assemble = t.elapsed().as_secs_f64();
            let (a, b) = (p.matrix, p.load);
            let n = a.nrows();
            let t = Instant::now();
            let amg = Amg::new(
                a.as_ref(),
                AmgOptions {
                    block_size: 3,
                    near_nullspace: Some(rigid_body_modes(&p.coordinates, 3)),
                    strength_threshold: 0.02,
                    ..Default::default()
                },
                SolveControl::NONE,
            )
            .unwrap();
            let setup = t.elapsed().as_secs_f64();
            let r = par::seeded_vector::<f64>(n, 5);
            let mut z = vec![0.0; n];
            let vcycle = median_seconds(5, || amg.apply(&r, &mut z).unwrap());
            let mut y = vec![0.0; n];
            let spmv = median_seconds(10, || a.as_ref().spmv(&r, &mut y));
            let t = Instant::now();
            let mut x = vec![0.0; n];
            let options = KrylovOptions {
                rtol: 1e-8,
                max_iterations: 2000,
                ..Default::default()
            };
            let report = pcg(&a, Some(&amg), &b, &mut x, &options, SolveControl::NONE).unwrap();
            let solve = t.elapsed().as_secs_f64();
            let iterations = report.iterations.max(1) as f64;
            println!(
                "{{\"section\":\"elasticity\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"assemble_s\":{assemble:.3},\"amg_setup_s\":{setup:.3},\"pcg_solve_s\":{solve:.3},\"iterations\":{},\"per_iteration_s\":{:.5},\"vcycle_s\":{vcycle:.5},\"spmv_s\":{spmv:.5},\"levels\":{:?},\"operator_complexity\":{:.2},\"relative_residual\":{:.2e},\"peak_rss_gib\":{:.2}}}",
                a.nnz(),
                report.iterations,
                solve / iterations,
                amg.statistics().level_rows,
                amg.statistics().operator_complexity,
                relative_residual(&a, &x, &b),
                peak_rss_gib()
            );
        }
        "direct" => {
            let p = elasticity3d(grid, grid, grid);
            let (a, b) = (p.matrix, p.load);
            let n = a.nrows();
            let options = DirectOptions {
                ordering: Ordering::NestedDissection,
                ..DirectOptions::new(FactorKind::Cholesky)
            };
            let t = Instant::now();
            let symbolic = SymbolicFactorization::analyze::<f64>(
                a.as_ref().into(),
                &options,
                SolveControl::NONE,
            )
            .unwrap();
            let analyze = t.elapsed().as_secs_f64();
            let estimate = symbolic.estimate();
            let t = Instant::now();
            let solver = DirectSolver::factorize(
                symbolic,
                a.as_ref().into(),
                options,
                None,
                SolveControl::NONE,
            )
            .unwrap();
            let factor = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut x = b.clone();
            solver.solve_in_place(&mut x, 1).unwrap();
            let solve = t.elapsed().as_secs_f64();
            println!(
                "{{\"section\":\"direct\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"analyze_s\":{analyze:.3},\"factor_s\":{factor:.3},\"solve_s\":{solve:.3},\"factor_gib\":{:.2},\"gflop\":{:.1},\"gflops_rate\":{:.1},\"relative_residual\":{:.2e},\"peak_rss_gib\":{:.2}}}",
                a.nnz(),
                estimate.factor_bytes as f64 / (1u64 << 30) as f64,
                estimate.flops / 1e9,
                estimate.flops / 1e9 / factor,
                relative_residual(&a, &x, &b),
                peak_rss_gib()
            );
        }
        "cfd" => {
            let a = convection3d(grid);
            let n = a.nrows();
            let b = par::seeded_vector::<f64>(n, 9);
            let options = KrylovOptions {
                rtol: 1e-8,
                max_iterations: 3000,
                ..Default::default()
            };
            let gmres_options = GmresOptions {
                restart: 30,
                ..Default::default()
            };
            let t = Instant::now();
            let ilu = Ilu0::new(a.as_ref(), false).unwrap();
            let ilu_setup = t.elapsed().as_secs_f64();
            let r = par::seeded_vector::<f64>(n, 4);
            let mut z = vec![0.0; n];
            let ilu_apply = median_seconds(5, || ilu.apply(&r, &mut z).unwrap());
            let t = Instant::now();
            let mut x = vec![0.0; n];
            let report = gmres(
                &a,
                Some(&ilu),
                &b,
                &mut x,
                &options,
                &gmres_options,
                SolveControl::NONE,
            )
            .unwrap();
            let solve = t.elapsed().as_secs_f64();
            println!(
                "{{\"section\":\"cfd_gmres_ilu0\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"setup_s\":{ilu_setup:.3},\"apply_s\":{ilu_apply:.5},\"solve_s\":{solve:.3},\"iterations\":{},\"per_iteration_s\":{:.5},\"relative_residual\":{:.2e}}}",
                a.nnz(), report.iterations, solve / report.iterations.max(1) as f64, relative_residual(&a, &x, &b)
            );
            let t = Instant::now();
            let amg = Amg::new(a.as_ref(), AmgOptions::default(), SolveControl::NONE).unwrap();
            let amg_setup = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut x = vec![0.0; n];
            let report = gmres(
                &a,
                Some(&amg),
                &b,
                &mut x,
                &options,
                &gmres_options,
                SolveControl::NONE,
            )
            .unwrap();
            let solve = t.elapsed().as_secs_f64();
            println!(
                "{{\"section\":\"cfd_gmres_amg\",\"n\":{n},\"threads\":{threads},\"setup_s\":{amg_setup:.3},\"solve_s\":{solve:.3},\"iterations\":{},\"per_iteration_s\":{:.5},\"relative_residual\":{:.2e}}}",
                report.iterations, solve / report.iterations.max(1) as f64, relative_residual(&a, &x, &b)
            );
            let pressure = poisson3d::<f64>(grid, grid, grid);
            let t = Instant::now();
            let amg =
                Amg::new(pressure.as_ref(), AmgOptions::default(), SolveControl::NONE).unwrap();
            let setup = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut x = vec![0.0; n];
            let report = pcg(
                &pressure,
                Some(&amg),
                &b,
                &mut x,
                &options,
                SolveControl::NONE,
            )
            .unwrap();
            let solve = t.elapsed().as_secs_f64();
            println!(
                "{{\"section\":\"cfd_pressure_amg_pcg\",\"n\":{n},\"threads\":{threads},\"setup_s\":{setup:.3},\"solve_s\":{solve:.3},\"iterations\":{},\"relative_residual\":{:.2e},\"peak_rss_gib\":{:.2}}}",
                report.iterations, relative_residual(&pressure, &x, &b), peak_rss_gib()
            );
        }
        "modal" => {
            let p = elasticity3d(grid, grid, grid);
            let k = p.matrix;
            let n = k.nrows();
            let mut mass = TripletBuilder::new(n, n);
            for row in 0..n {
                mass.push(row, row, 1.0);
            }
            let m: CsrMatrix<f64> = mass.build();
            let t = Instant::now();
            let result = modal(
                k.as_ref(),
                m.as_ref(),
                &EigenOptions {
                    count: 10,
                    ..Default::default()
                },
                SolveControl::NONE,
            )
            .unwrap();
            let total = t.elapsed().as_secs_f64();
            println!(
                "{{\"section\":\"modal\",\"n\":{n},\"threads\":{threads},\"modes\":{},\"total_s\":{total:.3},\"restarts\":{},\"operator_applications\":{},\"max_residual\":{:.2e},\"converged\":{},\"peak_rss_gib\":{:.2}}}",
                result.values.len(),
                result.restarts,
                result.operator_applications,
                result.residuals.iter().cloned().fold(0.0, f64::max),
                result.converged,
                peak_rss_gib()
            );
        }
        other => eprintln!("unknown section {other}"),
    });
}
