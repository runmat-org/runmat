// Complex numbers end-to-end tests
use runmat_builtins::Value;

#[test]
fn complex_scalar_arithmetic() {
	let a = Value::Complex(3.0, 2.0);
	let b = Value::Complex(1.5, -4.0);
	let s = runmat_runtime::elementwise_add(&a, &b).unwrap();
	if let Value::Complex(re, im) = s { assert!((re - 4.5).abs() < 1e-12); assert!((im + 2.0).abs() < 1e-12); } else { panic!("expected complex"); }
	let m = runmat_runtime::elementwise_mul(&a, &b).unwrap();
	if let Value::Complex(re, im) = m { assert!((re - (3.0*1.5 - 2.0*(-4.0))).abs() < 1e-12); assert!((im - (3.0*(-4.0) + 2.0*1.5)).abs() < 1e-12); } else { panic!("expected complex"); }
	let d = runmat_runtime::elementwise_div(&a, &b).unwrap();
	if let Value::Complex(re, im) = d { assert!(re.is_finite()); assert!(im.is_finite()); } else { panic!("expected complex"); }
}

#[test]
fn complex_scalar_with_real() {
	let a = Value::Complex(2.0, -1.0);
	let s = runmat_runtime::elementwise_add(&a, &Value::Num(3.0)).unwrap();
	if let Value::Complex(re, im) = s { assert!((re - 5.0).abs() < 1e-12); assert!((im + 1.0).abs() < 1e-12); } else { panic!(); }
	let m = runmat_runtime::elementwise_mul(&a, &Value::Num(2.0)).unwrap();
	if let Value::Complex(re, im) = m { assert!((re - 4.0).abs() < 1e-12); assert!((im + 2.0).abs() < 1e-12); } else { panic!(); }
	let d = runmat_runtime::elementwise_div(&Value::Num(5.0), &a).unwrap();
	if let Value::Complex(re, im) = d { assert!(re.is_finite()); assert!(im.is_finite()); } else { panic!("expected complex"); }
}

#[test]
fn complex_array_elementwise_add() {
	use runmat_builtins::ComplexTensor;
	let ct = ComplexTensor::new_2d(vec![(1.0,0.0),(0.0,1.0),(2.0,-3.0),(0.0,0.0)], 2, 2).unwrap();
	let a = Value::ComplexTensor(ct);
	let b = Value::Num(2.0);
	let c = runmat_runtime::elementwise_add(&a, &b).unwrap();
	if let Value::ComplexTensor(t) = c { assert_eq!(t.rows, 2); assert_eq!(t.cols,2); assert_eq!(t.data[0], (3.0,0.0)); assert_eq!(t.data[1], (2.0,1.0)); } else { panic!("expected ComplexTensor"); }
}

#[test]
fn complex_matmul_and_transpose() {
	use runmat_builtins::{ComplexTensor, Tensor};
	let a = ComplexTensor::new_2d(vec![(1.0,1.0),(0.0,-1.0),(2.0,0.0),(1.0,0.5)], 2, 2).unwrap();
	let b = ComplexTensor::new_2d(vec![(-1.0,0.0),(3.0,0.5),(0.0,2.0),(1.0,-1.0)], 2, 2).unwrap();
	let v = runmat_runtime::matrix::value_matmul(&Value::ComplexTensor(a.clone()), &Value::ComplexTensor(b.clone())).unwrap();
	if let Value::ComplexTensor(m) = v { assert_eq!(m.rows, 2); assert_eq!(m.cols, 2); }
	// real * complex
	let r = Tensor::new_2d(vec![1.0, 2.0, 0.0, 1.0], 2, 2).unwrap();
	let v2 = runmat_runtime::matrix::value_matmul(&Value::Tensor(r), &Value::ComplexTensor(b.clone())).unwrap();
	if let Value::ComplexTensor(m) = v2 { assert_eq!(m.rows, 2); assert_eq!(m.cols, 2); }
	// transpose conjugate on complex matrix
	let t = runmat_runtime::transpose(Value::ComplexTensor(a)).unwrap();
	if let Value::ComplexTensor(ct) = t { assert_eq!(ct.rows, 2); assert_eq!(ct.cols, 2); }
}

#[test]
fn complex_string_and_logical() {
	let a = Value::Complex(0.0, -2.5);
	let s = runmat_runtime::call_builtin("string", &[a.clone()]).unwrap();
	if let Value::StringArray(sa) = s { assert_eq!(sa.data[0], "-2.5i"); } else { panic!(); }
	let l = runmat_runtime::call_builtin("logical", &[Value::Complex(0.0, 0.0)]).unwrap(); if let Value::Bool(b) = l { assert!(!b); } else { panic!(); }
	let l2 = runmat_runtime::call_builtin("logical", &[Value::Complex(1.0, 0.0)]).unwrap(); if let Value::Bool(b) = l2 { assert!(b); } else { panic!(); }
	use runmat_builtins::ComplexTensor; let ct = ComplexTensor::new_2d(vec![(0.0,0.0),(1.0,0.0)],1,2).unwrap();
	let mask = runmat_runtime::call_builtin("logical", &[Value::ComplexTensor(ct)]).unwrap(); if let Value::LogicalArray(la) = mask { assert_eq!(la.data, vec![0,1]); } else { panic!(); }
}

#[test]
fn complex_power_basic() {
	// Complex elementwise/scalar power now supported
	let a = Value::Complex(2.0, 0.0);
	let p = runmat_runtime::elementwise_pow(&a, &Value::Num(3.0)).unwrap();
	if let Value::Complex(re, im) = p { assert!((re - 8.0).abs() < 1e-12); assert!(im.abs() < 1e-12); } else { panic!(); }
	let p2 = runmat_runtime::power(&Value::Num(2.0), &Value::Complex(3.0, 0.0)).unwrap();
	if let Value::Complex(re, im) = p2 { assert!((re - 8.0).abs() < 1e-12); assert!(im.abs() < 1e-12); } else { panic!(); }
}

#[test]
fn complex_matrix_power_integer() {
	use runmat_builtins::ComplexTensor;
	// A = [1+i, 0-1i; 2+0i, 1+0.5i]
	let a = ComplexTensor::new_2d(vec![(1.0,1.0),(2.0,0.0),(0.0,-1.0),(1.0,0.5)], 2, 2).unwrap();
	let v = runmat_runtime::power(&Value::ComplexTensor(a), &Value::Int(runmat_builtins::IntValue::I32(2))).unwrap();
	if let Value::ComplexTensor(m2) = v { assert_eq!(m2.rows, 2); assert_eq!(m2.cols, 2); } else { panic!(); }
}

#[test]
fn complex_elementwise_power_tensor() {
	use runmat_builtins::ComplexTensor;
	let ct = ComplexTensor::new_2d(vec![(1.0,1.0),(0.0,2.0)], 1, 2).unwrap();
	let out = runmat_runtime::elementwise_pow(&Value::ComplexTensor(ct), &Value::Num(2.0)).unwrap();
	if let Value::ComplexTensor(m) = out { assert_eq!(m.rows, 1); assert_eq!(m.cols, 2); } else { panic!(); }
}
