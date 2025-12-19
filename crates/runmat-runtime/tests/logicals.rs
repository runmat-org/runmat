use runmat_builtins::{LogicalArray, Tensor, Value};

#[test]
fn logical_array_construction_and_display() {
    let la = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
    let v = Value::LogicalArray(la.clone());
    let s = format!("{v}");
    assert!(s.eq("\n  0  1\n  1  0"));
}

#[test]
fn logical_mask_index_read_write() {
    // Create 2x2 tensor and mask
    let t = Tensor::new_2d(vec![10.0, 20.0, 30.0, 40.0], 2, 2).unwrap();
    let m = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    // Read using VM path is covered elsewhere; here validate conversion path
    let mask = Value::LogicalArray(m);
    let arr = Value::Tensor(t);
    // Convert mask to string for sanity
    let _ = runmat_runtime::call_builtin("string", &[mask]).unwrap();
    let _ = runmat_runtime::call_builtin("string", &[arr]).unwrap();
}

#[test]
fn logical_size_numel_ndims() {
    let la = LogicalArray::new(vec![1, 0, 1, 1, 0, 0], vec![3, 2]).unwrap();
    let v = Value::LogicalArray(la);
    let sz = runmat_runtime::call_builtin("size", std::slice::from_ref(&v)).unwrap();
    if let Value::Tensor(t) = sz {
        // Verify row vector and product equals numel
        let dims = t.data.clone();
        assert!(t.rows() == 1 || t.cols() == 1);
        let prod: usize = dims.iter().map(|x| *x as usize).product();
        let ne = runmat_runtime::call_builtin("numel", std::slice::from_ref(&v)).unwrap();
        if let Value::Num(n) = ne {
            assert_eq!(prod as f64, n);
        } else {
            panic!();
        }
    } else {
        panic!();
    }
    let nd = runmat_runtime::call_builtin("ndims", std::slice::from_ref(&v)).unwrap();
    if let Value::Num(n) = nd {
        assert_eq!(n, 2.0);
    } else {
        panic!();
    }
}

#[test]
fn logical_from_numeric_and_stringarray() {
    let t = Tensor::new_2d(vec![0.0, 1.0, -2.0], 3, 1).unwrap();
    let v = Value::Tensor(t);
    let l = runmat_runtime::call_builtin("logical", &[v]).unwrap();
    if let Value::LogicalArray(la) = l {
        assert_eq!(la.data, vec![0, 1, 1]);
    } else {
        panic!();
    }
    let sa = runmat_builtins::StringArray::new(vec!["".to_string(), "a".to_string()], vec![2, 1])
        .unwrap();
    let l2 = runmat_runtime::call_builtin("logical", &[Value::StringArray(sa)]).unwrap();
    if let Value::LogicalArray(la) = l2 {
        assert_eq!(la.data, vec![0, 1]);
    } else {
        panic!();
    }
}
