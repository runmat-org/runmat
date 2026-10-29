//! Krylov driver benchmark: `krylov_bench [grid] [threads] [repeats]`.
//!
//! Times the synchronous PCG (3D Poisson, Jacobi) and right-preconditioned GMRES(30)
//! (2D convection–diffusion, Jacobi) drivers on fixed systems and prints one JSON line per
//! solve with iterations, operator applications and wall time. The same source builds against
//! earlier revisions of the crate, so driver overhead can be compared across versions.

use std::time::Instant;

use runmat_sparse_solvers::control::SolveControl;
use runmat_sparse_solvers::krylov::{gmres, pcg, GmresOptions, KrylovOptions};
use runmat_sparse_solvers::testutil::poisson3d;
use runmat_sparse_solvers::{par, CsrMatrix, Preconditioner, SolverResult, TripletBuilder};

struct Jacobi(Vec<f64>);
impl Preconditioner<f64> for Jacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        for ((z, r), d) in z.iter_mut().zip(r).zip(&self.0) {
            *z = r / d;
        }
        Ok(())
    }
}

/// Upwinded convection–diffusion on an `m × m` grid (nonsymmetric).
fn convection(m: usize) -> CsrMatrix<f64> {
    let n = m * m;
    let mut t = TripletBuilder::new(n, n);
    let id = |i: usize, j: usize| i * m + j;
    for i in 0..m {
        for j in 0..m {
            let row = id(i, j);
            t.push(row, row, 4.0 + 0.6);
            if i > 0 {
                t.push(row, id(i - 1, j), -1.0 - 0.6);
            }
            if i + 1 < m {
                t.push(row, id(i + 1, j), -1.0);
            }
            if j > 0 {
                t.push(row, id(i, j - 1), -1.0);
            }
            if j + 1 < m {
                t.push(row, id(i, j + 1), -1.0);
            }
        }
    }
    t.build()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(40);
    let threads: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let repeats: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3);
    par::with_threads(threads, || {
        let a = poisson3d::<f64>(grid, grid, grid);
        let b = par::seeded_vector::<f64>(a.nrows(), 7);
        let jacobi = Jacobi(a.as_ref().diagonal());
        let options = KrylovOptions {
            rtol: 1e-10,
            max_iterations: 5000,
            ..Default::default()
        };
        for _ in 0..repeats {
            let mut x = vec![0.0; a.nrows()];
            let start = Instant::now();
            let report = pcg(&a, Some(&jacobi), &b, &mut x, &options, SolveControl::NONE).unwrap();
            println!(
                "{{\"solver\":\"pcg\",\"n\":{},\"threads\":{threads},\"iterations\":{},\"operator_applications\":{},\"relres\":{:e},\"seconds\":{:.6}}}",
                a.nrows(), report.iterations, report.operator_applications, report.relative_residual, start.elapsed().as_secs_f64()
            );
        }
        let m = grid * 6;
        let a = convection(m);
        let b = par::seeded_vector::<f64>(a.nrows(), 11);
        let jacobi = Jacobi(a.as_ref().diagonal());
        let options = KrylovOptions {
            rtol: 1e-8,
            max_iterations: 5000,
            ..Default::default()
        };
        let gmres_options = GmresOptions {
            restart: 30,
            ..Default::default()
        };
        for _ in 0..repeats {
            let mut x = vec![0.0; a.nrows()];
            let start = Instant::now();
            let report = gmres(
                &a,
                Some(&jacobi),
                &b,
                &mut x,
                &options,
                &gmres_options,
                SolveControl::NONE,
            )
            .unwrap();
            println!(
                "{{\"solver\":\"gmres\",\"n\":{},\"threads\":{threads},\"iterations\":{},\"operator_applications\":{},\"relres\":{:e},\"seconds\":{:.6}}}",
                a.nrows(), report.iterations, report.operator_applications, report.relative_residual, start.elapsed().as_secs_f64()
            );
        }
    });
}
