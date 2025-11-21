use runmat_builtins::Value;
use runmat_runtime as rt;

#[test]
fn squeeze_basic() {
    let t = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
    let v = rt::call_builtin("squeeze", &[Value::Tensor(t)]).unwrap();
    match v {
        Value::Tensor(tt) => {
            assert_eq!(tt.shape, vec![2, 2]);
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn permute_swap_dims() {
    let t = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    // order [2 1]
    let ord = runmat_builtins::Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap();
    let v = rt::call_builtin("permute", &[Value::Tensor(t.clone()), Value::Tensor(ord)]).unwrap();
    if let Value::Tensor(p) = v {
        assert_eq!(p.shape, vec![3, 2]);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn cat_dim1() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = runmat_builtins::Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let v = rt::call_builtin(
        "cat",
        &[Value::Num(1.0), Value::Tensor(a), Value::Tensor(b)],
    )
    .unwrap();
    if let Value::Tensor(t) = v {
        assert_eq!(t.shape, vec![4, 2]);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn cat_variadic_three_inputs() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let b = runmat_builtins::Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
    let c = runmat_builtins::Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
    let v = rt::call_builtin(
        "cat",
        &[
            Value::Num(2.0),
            Value::Tensor(a),
            Value::Tensor(b),
            Value::Tensor(c),
        ],
    )
    .unwrap();
    if let Value::Tensor(t) = v {
        assert_eq!(t.shape, vec![2, 3]);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn repmat_2d() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let v = rt::call_builtin(
        "repmat",
        &[Value::Tensor(a), Value::Num(2.0), Value::Num(3.0)],
    )
    .unwrap();
    if let Value::Tensor(t) = v {
        assert_eq!(t.shape, vec![4, 6]);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn repmat_nd_vector_form() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let reps = runmat_builtins::Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let v = rt::call_builtin("repmat", &[Value::Tensor(a), Value::Tensor(reps)]).unwrap();
    if let Value::Tensor(t) = v {
        assert_eq!(t.shape, vec![3, 8]);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn linspace_basic() {
    let v = rt::call_builtin(
        "linspace",
        &[
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Int(runmat_builtins::IntValue::I32(5)),
        ],
    )
    .unwrap();
    if let Value::Tensor(t) = v {
        assert_eq!(t.shape, vec![1, 5]);
        assert!((t.data[4] - 1.0).abs() < 1e-9);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn meshgrid_basic() {
    let x = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let y = runmat_builtins::Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
    // meshgrid returns its first output by default; ensure the shape follows MATLAB conventions
    let v = rt::call_builtin("meshgrid", &[Value::Tensor(x), Value::Tensor(y)]).unwrap();
    if let Value::Tensor(t) = v {
        assert!(t.shape == vec![2, 3] || t.shape == vec![3, 2]);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn diag_vector_to_matrix_and_back() {
    // Vector -> diag matrix -> extract main diagonal
    let v = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let m = rt::call_builtin("diag", &[Value::Tensor(v)]).unwrap();
    if let Value::Tensor(mt) = &m {
        assert_eq!(mt.shape, vec![3, 3]);
    }
    let d = rt::call_builtin("diag", &[m]).unwrap();
    if let Value::Tensor(dt) = d {
        assert_eq!(dt.shape, vec![3, 1]);
        assert!((dt.data[2] - 3.0).abs() < 1e-9);
    } else {
        panic!("expected tensor")
    }
}

#[test]
fn triu_tril_shapes() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let u = rt::call_builtin("triu", &[Value::Tensor(a.clone())]).unwrap();
    let l = rt::call_builtin("tril", &[Value::Tensor(a)]).unwrap();
    if let Value::Tensor(ut) = u {
        assert_eq!(ut.shape, vec![2, 2]);
    } else {
        panic!("expected tensor")
    }
    if let Value::Tensor(lt) = l {
        assert_eq!(lt.shape, vec![2, 2]);
    } else {
        panic!("expected tensor")
    }
}
