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


