//! Device multigrid against the host hierarchy (skipped when no adapter is available).

use runmat_accelerate::backend::wgpu::provider::{
    WgpuPrecisionPolicy, WgpuProvider, WgpuProviderOptions,
};
use runmat_sparse_accelerate::{DeviceAmg, DeviceApply};
use runmat_sparse_solvers::control::SolveControl;
use runmat_sparse_solvers::krylov::{pcg, KrylovOptions};
use runmat_sparse_solvers::precond::{rigid_body_modes, Amg, AmgOptions};
use runmat_sparse_solvers::testutil::elasticity::elasticity3d;
use runmat_sparse_solvers::{par, Preconditioner};

fn provider() -> Option<WgpuProvider> {
    WgpuProvider::new(WgpuProviderOptions {
        precision: WgpuPrecisionPolicy::RequireF32,
        ..Default::default()
    })
    .ok()
}

#[test]
fn device_vcycle_matches_the_host_cycle_and_inner_cg_certifies_in_f64() {
    let Some(provider) = provider() else {
        eprintln!("no f32 wgpu adapter; skipping");
        return;
    };
    let p = elasticity3d(10, 9, 8);
    let a = &p.matrix;
    let amg = Amg::new(
        a.as_ref(),
        AmgOptions {
            block_size: 3,
            near_nullspace: Some(rigid_body_modes(&p.coordinates, 3)),
            coarse_size: 300,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    let r = par::seeded_vector::<f64>(a.nrows(), 3);
    let (mut host, mut device) = (vec![0.0; r.len()], vec![0.0; r.len()]);
    amg.apply(&r, &mut host).unwrap();
    let cycle = DeviceAmg::new(&provider, &amg, DeviceApply::VCycle).unwrap();
    cycle.apply(&r, &mut device).unwrap();
    let scale = host.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = host
        .iter()
        .zip(&device)
        .fold(0.0f64, |m, (h, d)| m.max((h - d).abs()));
    assert!(
        worst <= 1e-4 * scale,
        "device cycle differs by {worst} (scale {scale})"
    );

    let inner = DeviceAmg::new(&provider, &amg, DeviceApply::InnerCg { iterations: 8 }).unwrap();
    assert!(inner.is_variable());
    let b = &p.load;
    let mut x = vec![0.0; b.len()];
    let options = KrylovOptions {
        rtol: 1e-10,
        max_iterations: 100,
        ..Default::default()
    };
    let report = pcg(a, Some(&inner), b, &mut x, &options, SolveControl::NONE).unwrap();
    assert!(report.converged(), "{report:?}");
    let mut ax = vec![0.0; b.len()];
    a.as_ref().spmv(&x, &mut ax);
    let residual: Vec<f64> = b.iter().zip(&ax).map(|(b, v)| b - v).collect();
    assert!(par::norm2(&residual) <= 1e-10 * par::norm2(b));
    assert!(report.iterations <= 10, "{report:?}");
}
