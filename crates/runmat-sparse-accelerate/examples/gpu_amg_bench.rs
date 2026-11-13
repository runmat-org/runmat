//! Device multigrid benchmark: `gpu_amg_bench <grid> <threads>`.
//!
//! Q1 elasticity (3 dof/node): the host AMG-PCG in f64 against the same hierarchy on the
//! accelerator (f32): a fixed V-cycle preconditioning f64 PCG, and an inner device CG
//! preconditioning f64 flexible CG. Also times the device fine-level SpMV (repetitions, transfers
//! excluded) and reports its achieved bandwidth. One JSON object per measurement.

use std::time::Instant;

use runmat_accelerate::backend::wgpu::provider::{
    WgpuPrecisionPolicy, WgpuProvider, WgpuProviderOptions,
};
use runmat_accelerate_api::SparseMultigridApply;
use runmat_sparse_accelerate::{DeviceAmg, DeviceApply};
use runmat_sparse_solvers::control::SolveControl;
use runmat_sparse_solvers::krylov::{pcg, KrylovOptions};
use runmat_sparse_solvers::precond::{rigid_body_modes, Amg, AmgOptions};
use runmat_sparse_solvers::testutil::elasticity::elasticity3d;
use runmat_sparse_solvers::{par, CsrMatrix, Preconditioner};

fn relative_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.as_ref().spmv(x, &mut ax);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
    par::norm2(&r) / par::norm2(b)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(32);
    let threads: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let provider = WgpuProvider::new(WgpuProviderOptions {
        precision: WgpuPrecisionPolicy::RequireF32,
        ..WgpuProviderOptions::default()
    })
    .expect("wgpu provider");
    par::with_threads(threads, || {
        let p = elasticity3d(grid, grid, grid);
        let (a, b) = (&p.matrix, &p.load);
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
        let options = KrylovOptions {
            rtol: 1e-8,
            max_iterations: 500,
            ..Default::default()
        };
        let mut x = vec![0.0; n];
        let t = Instant::now();
        let report = pcg(a, Some(&amg), b, &mut x, &options, SolveControl::NONE).unwrap();
        let cpu_solve = t.elapsed().as_secs_f64();
        println!(
            "{{\"case\":\"cpu_amg_pcg_f64\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"setup_s\":{setup:.3},\"solve_s\":{cpu_solve:.3},\"iterations\":{},\"relres\":{:.2e}}}",
            a.nnz(), report.iterations, relative_residual(a, &x, b)
        );
        let t = Instant::now();
        let device = DeviceAmg::new(&provider, &amg, DeviceApply::VCycle).unwrap();
        let upload = t.elapsed().as_secs_f64();
        // Device SpMV: time(R) − time(1) over R − 1 repetitions isolates the kernel.
        let rhs: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let timed = |reps: u32| {
            let t = Instant::now();
            device
                .apply_f32(&rhs, SparseMultigridApply::Operator { repetitions: reps })
                .unwrap();
            t.elapsed().as_secs_f64()
        };
        timed(1);
        let (one, many) = (timed(1), timed(51));
        let per_spmv = (many - one) / 50.0;
        let bytes = (a.nnz() * 8 + n * 4 * 3) as f64; // values + indices, x (cached gather), y, row pointers
        println!(
            "{{\"case\":\"gpu_spmv_f32\",\"n\":{n},\"nnz\":{},\"upload_s\":{upload:.3},\"transfer_roundtrip_s\":{one:.5},\"spmv_s\":{per_spmv:.6},\"gb_per_s\":{:.1}}}",
            a.nnz(), bytes / per_spmv / 1e9
        );
        let t = Instant::now();
        let mut z = vec![0.0; n];
        for _ in 0..5 {
            device.apply(b, &mut z).unwrap();
        }
        let vcycle = t.elapsed().as_secs_f64() / 5.0;
        let mut x = vec![0.0; n];
        let t = Instant::now();
        let report = pcg(a, Some(&device), b, &mut x, &options, SolveControl::NONE).unwrap();
        let solve = t.elapsed().as_secs_f64();
        println!(
            "{{\"case\":\"gpu_vcycle_pcg\",\"n\":{n},\"threads\":{threads},\"vcycle_apply_s\":{vcycle:.5},\"solve_s\":{solve:.3},\"iterations\":{},\"relres\":{:.2e}}}",
            report.iterations, relative_residual(a, &x, b)
        );
        drop(device);
        for inner in [5u32, 10] {
            let device =
                DeviceAmg::new(&provider, &amg, DeviceApply::InnerCg { iterations: inner })
                    .unwrap();
            assert!(device.is_variable());
            let mut x = vec![0.0; n];
            let t = Instant::now();
            let report = pcg(a, Some(&device), b, &mut x, &options, SolveControl::NONE).unwrap();
            let solve = t.elapsed().as_secs_f64();
            println!(
                "{{\"case\":\"gpu_inner_cg_fcg\",\"inner\":{inner},\"n\":{n},\"threads\":{threads},\"solve_s\":{solve:.3},\"outer_iterations\":{},\"relres\":{:.2e}}}",
                report.iterations, relative_residual(a, &x, b)
            );
        }
    });
}
