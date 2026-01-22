use std::sync::Once;

use runmat_accelerate::simple_provider::register_inprocess_provider;
use runmat_accelerate::{prepare_builtin_args, promote_binary, BinaryOp};
use runmat_builtins::{Tensor, Value};

static INIT: Once = Once::new();

fn ensure_auto_init() {
    INIT.call_once(|| {
        std::env::set_var("RUNMAT_ACCEL_AUTO_OFFLOAD", "1");
        std::env::set_var("RUNMAT_ACCEL_CALIBRATE", "0");
        std::env::set_var("RUNMAT_ACCEL_THRESHOLD_ELEMWISE", "1");
        std::env::set_var("RUNMAT_ACCEL_THRESHOLD_REDUCTION", "1");
        register_inprocess_provider();
    });
}

fn make_tensor(len: usize) -> Tensor {
    let data: Vec<f64> = (0..len).map(|i| i as f64).collect();
    Tensor::new(data, vec![len, 1]).expect("tensor")
}

#[test]
fn promotes_large_tensors_to_gpu_for_elementwise() {
    ensure_auto_init();
    let tensor = make_tensor(8);
    let value = Value::Tensor(tensor.clone());
    let (a_gpu, b_gpu) = promote_binary(BinaryOp::Elementwise, &value, &value).expect("promote");
    assert!(matches!(a_gpu, Value::GpuTensor(_)));
    assert!(matches!(b_gpu, Value::GpuTensor(_)));
}

#[tokio::test]
async fn gather_occurs_for_sink_builtins() {
    ensure_auto_init();
    let tensor = make_tensor(4);
    let value = Value::Tensor(tensor.clone());
    let (gpu, _) = promote_binary(BinaryOp::Elementwise, &value, &value).expect("promote");
    let prepared = prepare_builtin_args("disp", &[gpu]).await.expect("prepare");
    assert!(matches!(prepared.as_slice(), [Value::Tensor(_)]));
}
