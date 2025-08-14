use runmat_builtins::Value;

#[test]
fn gpuarray_gather_roundtrip() {
	// Register in-process accelerate provider so gpuArray/gather are functional
	runmat_accelerate::simple_provider::register_inprocess_provider();

	// Create a small tensor and upload via gpuArray
	let t = runmat_builtins::Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
	let v = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t.clone())]).unwrap();
	match v {
		Value::GpuTensor(h) => {
			// Gather back and verify contents
			let g = runmat_runtime::call_builtin("gather", &[Value::GpuTensor(h)]).unwrap();
			if let Value::Tensor(tt) = g {
				assert_eq!(tt.rows(), 2);
				assert_eq!(tt.cols(), 2);
				assert_eq!(tt.data, t.data);
			} else { panic!("expected tensor from gather"); }
		}
		other => panic!("expected GpuTensor, got {other:?}"),
	}
}

#[test]
fn elementwise_add_on_gpu_handles() {
	// Ensure provider registered
	runmat_accelerate::simple_provider::register_inprocess_provider();
	let t1 = runmat_builtins::Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
	let t2 = runmat_builtins::Tensor::new_2d(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
	let v1 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t1.clone())]).unwrap();
	let v2 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t2.clone())]).unwrap();
	let sum = runmat_runtime::elementwise::elementwise_add(&v1, &v2).unwrap();
	let gathered = runmat_runtime::call_builtin("gather", &[sum]).unwrap();
	if let Value::Tensor(tt) = gathered {
		assert_eq!(tt.rows(), 2);
		assert_eq!(tt.cols(), 2);
		assert_eq!(tt.data, vec![6.0, 8.0, 10.0, 12.0]);
	} else { panic!("expected tensor from gather"); }
}

#[test]
fn elementwise_mul_on_gpu_handles() {
	runmat_accelerate::simple_provider::register_inprocess_provider();
	let t1 = runmat_builtins::Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
	let t2 = runmat_builtins::Tensor::new_2d(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
	let v1 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t1.clone())]).unwrap();
	let v2 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t2.clone())]).unwrap();
	let prod = runmat_runtime::elementwise::elementwise_mul(&v1, &v2).unwrap();
	let gathered = runmat_runtime::call_builtin("gather", &[prod]).unwrap();
	if let Value::Tensor(tt) = gathered {
		assert_eq!(tt.data, vec![5.0, 12.0, 21.0, 32.0]);
	} else { panic!("expected tensor from gather"); }
}

#[test]
fn elementwise_sub_on_gpu_handles() {
	runmat_accelerate::simple_provider::register_inprocess_provider();
	let t1 = runmat_builtins::Tensor::new_2d(vec![10.0, 20.0, 30.0, 40.0], 2, 2).unwrap();
	let t2 = runmat_builtins::Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
	let v1 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t1.clone())]).unwrap();
	let v2 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t2.clone())]).unwrap();
	let res = runmat_runtime::elementwise::elementwise_sub(&v1, &v2).unwrap();
	let gathered = runmat_runtime::call_builtin("gather", &[res]).unwrap();
	if let Value::Tensor(tt) = gathered { assert_eq!(tt.data, vec![9.0, 18.0, 27.0, 36.0]); } else { panic!("expected tensor"); }
}

#[test]
fn elementwise_div_on_gpu_handles() {
	runmat_accelerate::simple_provider::register_inprocess_provider();
	let t1 = runmat_builtins::Tensor::new_2d(vec![10.0, 20.0, 30.0, 40.0], 2, 2).unwrap();
	let t2 = runmat_builtins::Tensor::new_2d(vec![2.0, 4.0, 5.0, 8.0], 2, 2).unwrap();
	let v1 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t1.clone())]).unwrap();
	let v2 = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(t2.clone())]).unwrap();
	let res = runmat_runtime::elementwise::elementwise_div(&v1, &v2).unwrap();
	let gathered = runmat_runtime::call_builtin("gather", &[res]).unwrap();
	if let Value::Tensor(tt) = gathered { assert_eq!(tt.data, vec![5.0, 5.0, 6.0, 5.0]); } else { panic!("expected tensor"); }
}

#[test]
fn matmul_on_gpu_handles_or_fallback() {
	runmat_accelerate::simple_provider::register_inprocess_provider();
	let a = runmat_builtins::Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
	let b = runmat_builtins::Tensor::new_2d(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
	let ga = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(a.clone())]).unwrap();
	let gb = runmat_runtime::call_builtin("gpuArray", &[Value::Tensor(b.clone())]).unwrap();
	let res = runmat_runtime::call_builtin("mtimes", &[ga, gb]).unwrap();
	let gathered = runmat_runtime::call_builtin("gather", &[res]).unwrap();
	if let Value::Tensor(tt) = gathered {
		// Column-major for A*B: first column [23,34], second [31,46]
		assert_eq!(tt.data, vec![23.0, 34.0, 31.0, 46.0]);
	} else { panic!("expected tensor"); }
}


