//! Scaling benchmark: `sparse_bench <poisson|elasticity> <grid> <threads> <direct|amg|both> [nd|amd]`.
//! Prints one JSON object per measured phase. Run through `gate bench` for clean timings.
//! Peak RSS is process-lifetime evidence; unavailable platform metrics are JSON null.

use std::time::Instant;

#[path = "support/memory_report.rs"]
mod memory_report;

use memory_report::peak_rss_gib;

use runmat_sparse_solvers::control::SolveControl;
use runmat_sparse_solvers::direct::{
    DirectOptions, DirectSolver, FactorKind, SymbolicFactorization,
};
use runmat_sparse_solvers::krylov::{pcg, KrylovOptions};
use runmat_sparse_solvers::ordering::Ordering;
use runmat_sparse_solvers::par;
use runmat_sparse_solvers::precond::{rigid_body_modes, Amg, AmgOptions};
use runmat_sparse_solvers::testutil::{elasticity::elasticity3d, poisson3d};
use runmat_sparse_solvers::CsrMatrix;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let problem = args.get(1).map(String::as_str).unwrap_or("poisson");
    let grid: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50);
    let threads: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mode = args.get(4).map(String::as_str).unwrap_or("both");
    let ordering = match args.get(5).map(String::as_str) {
        Some("amd") => Ordering::Amd,
        _ => Ordering::NestedDissection,
    };
    let t = Instant::now();
    let (a, b, near_nullspace, block): (CsrMatrix<f64>, Vec<f64>, _, usize) =
        par::with_threads(threads, || match problem {
            "elasticity" => {
                let p = elasticity3d(grid, grid, grid);
                let ns = rigid_body_modes(&p.coordinates, 3);
                (p.matrix, p.load, Some(ns), 3)
            }
            _ => {
                let a = poisson3d::<f64>(grid, grid, grid);
                let n = a.nrows();
                (a, vec![1.0; n], None, 1)
            }
        });
    let n = a.nrows();
    println!(
        "{{\"phase\":\"assemble\",\"problem\":\"{problem}\",\"grid\":{grid},\"n\":{n},\"nnz\":{},\"threads\":{threads},\"seconds\":{:.3}}}",
        a.nnz(),
        t.elapsed().as_secs_f64()
    );
    let relative = |x: &[f64]| {
        let mut ax = vec![0.0; n];
        a.as_ref().spmv(x, &mut ax);
        let r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
        par::norm2(&r) / par::norm2(&b)
    };
    par::with_threads(threads, || {
        if mode == "estimate" {
            let options = DirectOptions {
                ordering: ordering.clone(),
                ..DirectOptions::new(FactorKind::Cholesky)
            };
            let t = Instant::now();
            let symbolic = SymbolicFactorization::analyze::<f64>(
                a.as_ref().into(),
                &options,
                SolveControl::NONE,
            )
            .unwrap();
            let e = symbolic.estimate();
            println!(
                "{{\"phase\":\"estimate\",\"problem\":\"{problem}\",\"n\":{n},\"ordering\":\"{ordering:?}\",\"analyze_s\":{:.3},\"factor_entries\":{},\"factor_gib\":{:.3},\"workspace_gib\":{:.3},\"gflop\":{:.1}}}",
                t.elapsed().as_secs_f64(),
                e.factor_entries,
                e.factor_bytes as f64 / (1u64 << 30) as f64,
                e.workspace_bytes as f64 / (1u64 << 30) as f64,
                e.flops / 1e9
            );
        }
        if mode == "direct" || mode == "both" {
            let options = DirectOptions {
                ordering: ordering.clone(),
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
            let e = symbolic.estimate();
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
                "{{\"phase\":\"direct\",\"problem\":\"{problem}\",\"n\":{n},\"threads\":{threads},\"ordering\":\"{ordering:?}\",\"analyze_s\":{analyze:.3},\"factor_s\":{factor:.3},\"solve_s\":{solve:.3},\"factor_entries\":{},\"factor_gib\":{:.3},\"gflop\":{:.1},\"gflops_rate\":{:.1},\"relative_residual\":{:.2e},\"peak_rss_gib\":{:.2}}}",
                e.factor_entries,
                e.factor_bytes as f64 / (1u64 << 30) as f64,
                e.flops / 1e9,
                e.flops / 1e9 / factor,
                relative(&x),
                peak_rss_gib()
            );
        }
        if mode == "amg" || mode == "both" {
            let t = Instant::now();
            let amg = Amg::new(
                a.as_ref(),
                AmgOptions {
                    block_size: block,
                    near_nullspace: near_nullspace.clone(),
                    strength_threshold: if block > 1 { 0.02 } else { 0.0 },
                    ..Default::default()
                },
                SolveControl::NONE,
            )
            .unwrap();
            let setup = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut x = vec![0.0; n];
            let report = pcg(
                &a,
                Some(&amg),
                &b,
                &mut x,
                &KrylovOptions {
                    rtol: 1e-8,
                    max_iterations: 1000,
                    ..Default::default()
                },
                SolveControl::NONE,
            )
            .unwrap();
            let solve = t.elapsed().as_secs_f64();
            println!(
                "{{\"phase\":\"amg_pcg\",\"problem\":\"{problem}\",\"n\":{n},\"threads\":{threads},\"setup_s\":{setup:.3},\"solve_s\":{solve:.3},\"iterations\":{},\"levels\":{:?},\"operator_complexity\":{:.2},\"relative_residual\":{:.2e},\"peak_rss_gib\":{:.2}}}",
                report.iterations,
                amg.statistics().level_rows,
                amg.statistics().operator_complexity,
                relative(&x),
                peak_rss_gib()
            );
        }
    });
}
