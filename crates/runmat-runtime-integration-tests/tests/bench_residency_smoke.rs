use runmat_value::{Tensor, Value};

#[test]
fn residency_broadcast_2d_chain() {
    runmat_accelerate::simple_provider::register_inprocess_provider();

    // Shapes: (p×C) vs (1×C)
    let p = 4usize;
    let c = 3usize;
    let a_data: Vec<f64> = (0..(p * c)).map(|i| i as f64).collect();
    let b_data: Vec<f64> = (0..c).map(|i| (i as f64) + 1.0).collect();
    let a = Value::Tensor(Tensor::new(a_data, vec![p, c]).unwrap());
    let b = Value::Tensor(Tensor::new(b_data, vec![1, c]).unwrap());

    let ga = runmat_runtime::call_builtin("gpuArray", &[a]).unwrap();
    let gb = runmat_runtime::call_builtin("gpuArray", &[b]).unwrap();

    // Chain: (ga - gb) ./ (gb + 1) .* 0.5 + 2 then .^ 0.9
    let sub =
        runmat_runtime::call_builtin("minus", &[ga.value().clone(), gb.value().clone()]).unwrap();
    assert!(matches!(sub.value(), Value::GpuTensor(_)));
    let denom =
        runmat_runtime::call_builtin("plus", &[gb.value().clone(), Value::Num(1.0)]).unwrap();
    assert!(matches!(denom.value(), Value::GpuTensor(_)));
    let div =
        runmat_runtime::call_builtin("rdivide", &[sub.value().clone(), denom.value().clone()])
            .unwrap();
    assert!(matches!(div.value(), Value::GpuTensor(_)));
    let half =
        runmat_runtime::call_builtin("times", &[div.value().clone(), Value::Num(0.5)]).unwrap();
    assert!(matches!(half.value(), Value::GpuTensor(_)));
    let add2 =
        runmat_runtime::call_builtin("plus", &[half.value().clone(), Value::Num(2.0)]).unwrap();
    assert!(matches!(add2.value(), Value::GpuTensor(_)));
    let pow =
        runmat_runtime::call_builtin("power", &[add2.value().clone(), Value::Num(0.9)]).unwrap();
    assert!(matches!(pow.value(), Value::GpuTensor(_)));

    // Gather and check shape
    let out = runmat_runtime::call_builtin("gather", &[pow.value().clone()]).unwrap();
    if let Value::Tensor(t) = out.value() {
        assert_eq!(t.shape(), &[p, c]);
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn residency_broadcast_3d_chain_and_mean_vecdim() {
    runmat_accelerate::simple_provider::register_inprocess_provider();

    // Shapes: (B,H,W) vs (B,1,1)
    let (b, h, w) = (2usize, 3usize, 4usize);
    let total = b * h * w;
    let a_data: Vec<f64> = (0..total).map(|i| (i % 7) as f64).collect();
    let bias_data: Vec<f64> = (0..b).map(|i| (i as f64) * 0.1).collect();
    let a = Value::Tensor(Tensor::new(a_data, vec![b, h, w]).unwrap());
    let bias = Value::Tensor(Tensor::new(bias_data, vec![b, 1, 1]).unwrap());

    let ga = runmat_runtime::call_builtin("gpuArray", &[a]).unwrap();
    let gbias = runmat_runtime::call_builtin("gpuArray", &[bias]).unwrap();

    // (ga - gbias) .* 0.25 + 1.0 then gamma-like power
    let sub = runmat_runtime::call_builtin("minus", &[ga.value().clone(), gbias.value().clone()])
        .unwrap();
    assert!(matches!(sub.value(), Value::GpuTensor(_)));
    let scaled =
        runmat_runtime::call_builtin("times", &[sub.value().clone(), Value::Num(0.25)]).unwrap();
    assert!(matches!(scaled.value(), Value::GpuTensor(_)));
    let shifted =
        runmat_runtime::call_builtin("plus", &[scaled.value().clone(), Value::Num(1.0)]).unwrap();
    assert!(matches!(shifted.value(), Value::GpuTensor(_)));

    // mean over [2 3] (dims vector)
    let mean = runmat_runtime::call_builtin(
        "mean",
        &[
            shifted.value().clone(),
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        ],
    )
    .unwrap();
    assert!(matches!(mean.value(), Value::GpuTensor(_)));

    // Gather and check shape [B,1,1] -> effectively [B,1,1] or [B,1]
    let out = runmat_runtime::call_builtin("gather", &[mean.value().clone()]).unwrap();
    if let Value::Tensor(t) = out.value() {
        assert_eq!(t.shape().first().copied(), Some(b));
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn residency_thermal_camera_chain() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let (b, h, w) = (2usize, 3usize, 4usize);
    let total = b * h * w;
    // imgs ~ random-ish but deterministic
    let imgs: Vec<f64> = (0..total).map(|i| ((i % 11) as f64) * 0.1).collect();
    let offset_hw: Vec<f64> = (0..(h * w)).map(|i| (i % 7) as f64 * 0.01).collect();
    let g_imgs = runmat_runtime::call_builtin(
        "gpuArray",
        &[Value::Tensor(Tensor::new(imgs, vec![b, h, w]).unwrap())],
    )
    .unwrap();
    let g_off = runmat_runtime::call_builtin(
        "gpuArray",
        &[Value::Tensor(
            Tensor::new(offset_hw, vec![1, h, w]).unwrap(),
        )],
    )
    .unwrap();

    // subtraction with implicit expansion: (B,H,W) - (1,H,W)
    let sub =
        runmat_runtime::call_builtin("minus", &[g_imgs.value().clone(), g_off.value().clone()])
            .unwrap();
    assert!(matches!(sub.value(), Value::GpuTensor(_)));
    // times (.*)
    let scaled =
        runmat_runtime::call_builtin("times", &[sub.value().clone(), Value::Num(0.75)]).unwrap();
    assert!(matches!(scaled.value(), Value::GpuTensor(_)));
    // add (+)
    let shifted =
        runmat_runtime::call_builtin("plus", &[scaled.value().clone(), Value::Num(0.1)]).unwrap();
    assert!(matches!(shifted.value(), Value::GpuTensor(_)));
    // clamp via mask: A .* (A >= 0)
    let mask =
        runmat_runtime::call_builtin("ge", &[shifted.value().clone(), Value::Num(0.0)]).unwrap();
    assert!(matches!(mask.value(), Value::GpuTensor(_)));
    // Arithmetic consumes numerical device views. Convert the predicate via
    // the public class-conversion path before multiplying the image.
    let numeric_mask = runmat_runtime::call_builtin("double", &[mask.value().clone()]).unwrap();
    assert!(matches!(numeric_mask.value(), Value::GpuTensor(_)));
    let clamped = runmat_runtime::call_builtin(
        "times",
        &[shifted.value().clone(), numeric_mask.value().clone()],
    )
    .unwrap();
    assert!(matches!(clamped.value(), Value::GpuTensor(_)));
    // log1p via log(1 + x)
    let one_plus =
        runmat_runtime::call_builtin("plus", &[clamped.value().clone(), Value::Num(1.0)]).unwrap();
    assert!(matches!(one_plus.value(), Value::GpuTensor(_)));
    let loged = runmat_runtime::call_builtin("log", &[one_plus.value().clone()]).unwrap();
    assert!(matches!(loged.value(), Value::GpuTensor(_)));
    // mean(A,'all') -> scalar
    let mean_all =
        runmat_runtime::call_builtin("mean", &[loged.value().clone(), Value::from("all")]).unwrap();
    assert!(matches!(mean_all.value(), Value::GpuTensor(_)));
}

#[test]
fn residency_pca_center_and_normalize() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    // shapes: (n,d) vs (1,d)
    let (n, d) = (5usize, 7usize);
    let total = n * d;
    let x: Vec<f64> = (0..total).map(|i| (i % 13) as f64 * 0.01).collect();
    let mu: Vec<f64> = (0..d).map(|i| (i as f64) * 0.001).collect();
    let gx = runmat_runtime::call_builtin(
        "gpuArray",
        &[Value::Tensor(Tensor::new(x, vec![n, d]).unwrap())],
    )
    .unwrap();
    let gmu = runmat_runtime::call_builtin(
        "gpuArray",
        &[Value::Tensor(Tensor::new(mu, vec![1, d]).unwrap())],
    )
    .unwrap();

    // center: X - mu (broadcast)
    let centered =
        runmat_runtime::call_builtin("minus", &[gx.value().clone(), gmu.value().clone()]).unwrap();
    assert!(matches!(centered.value(), Value::GpuTensor(_)));
    // variance estimate: mean(centered.^2, 1)
    let sq = runmat_runtime::call_builtin("power", &[centered.value().clone(), Value::Num(2.0)])
        .unwrap();
    assert!(matches!(sq.value(), Value::GpuTensor(_)));
    let var = runmat_runtime::call_builtin("mean", &[sq.value().clone(), Value::Num(1.0)]).unwrap();
    assert!(matches!(var.value(), Value::GpuTensor(_)));
    // cast to single then back to double to check residency of casts
    let single = runmat_runtime::call_builtin("single", &[var.value().clone()]).unwrap();
    assert!(matches!(single.value(), Value::GpuTensor(_)));
    let dbl = runmat_runtime::call_builtin("double", &[single.value().clone()]).unwrap();
    assert!(matches!(dbl.value(), Value::GpuTensor(_)));
}

#[test]
fn residency_nlms_like_ops() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    // sum(x.*x,1) and sum(x.*w,1) with (p×C) vs (1×C)
    let (p, c) = (8usize, 5usize);
    let total = p * c;
    let x: Vec<f64> = (0..total).map(|i| (i % 9) as f64 * 0.02).collect();
    let wvec: Vec<f64> = (0..c).map(|i| 1.0 + (i as f64) * 0.1).collect();
    let gx = runmat_runtime::call_builtin(
        "gpuArray",
        &[Value::Tensor(Tensor::new(x, vec![p, c]).unwrap())],
    )
    .unwrap();
    let gw = runmat_runtime::call_builtin(
        "gpuArray",
        &[Value::Tensor(Tensor::new(wvec, vec![1, c]).unwrap())],
    )
    .unwrap();

    let xx =
        runmat_runtime::call_builtin("times", &[gx.value().clone(), gx.value().clone()]).unwrap();
    assert!(matches!(xx.value(), Value::GpuTensor(_)));
    let sx = runmat_runtime::call_builtin("sum", &[xx.value().clone(), Value::Num(1.0)]).unwrap();
    assert!(matches!(sx.value(), Value::GpuTensor(_)));

    let xw =
        runmat_runtime::call_builtin("times", &[gx.value().clone(), gw.value().clone()]).unwrap();
    assert!(matches!(xw.value(), Value::GpuTensor(_)));
    let sw = runmat_runtime::call_builtin("sum", &[xw.value().clone(), Value::Num(1.0)]).unwrap();
    assert!(matches!(sw.value(), Value::GpuTensor(_)));
}
