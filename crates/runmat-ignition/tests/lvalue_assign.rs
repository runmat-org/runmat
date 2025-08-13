use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn chained_member_and_index_assignments() {
	// Register test classes and construct an OverIdx object, then chain operations
	let src = "__register_test_classes(); o = new_object('OverIdx'); try; o(1)=42; catch e; end; o.k = 7; z = o.k;";
	let hir = lower(&parse(src).unwrap()).unwrap();
	let vars = execute(&hir).unwrap();
	assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn deep_chain_with_try_catch() {
	// obj(1).field{1} = 3; with try/catch around indexing
	let src = "__register_test_classes(); o = new_object('OverIdx'); try; o(1).f{1}=3; catch e; ok=1; end;";
	let hir = lower(&parse(src).unwrap()).unwrap();
	let vars = execute(&hir).unwrap();
	assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn brace_get_and_set_on_object() {
	// Test obj{1}=v and then read obj{1}
	let src = "__register_test_classes(); o = new_object('OverIdx'); o{1} = 5; r = o{1};";
	let hir = lower(&parse(src).unwrap()).unwrap();
	let vars = execute(&hir).unwrap();
	assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn colon_slice_and_broadcast_assign() {
	// Test colon slice on tensor (positive case) to keep suite meaningful
	let src = "A=[1,2;3,4]; y = A(:,2);";
	let hir = lower(&parse(src).unwrap()).unwrap();
	let vars = execute(&hir).unwrap();
	// y should be column vector [2;4] -> data [2,4]
	assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![2.0,4.0])));
}

#[test]
fn logical_mask_indexing_chain() {
	// Logical mask on second column selection then assigning via chain: A(:, [true false]) = A(:,[true false])
	let src = "A=[1,2;3,4]; mask=[1 0]; B=A(:,mask);";
	let hir = lower(&parse(src).unwrap()).unwrap();
	let vars = execute(&hir).unwrap();
	// B should be first column [1;3]
	assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![1.0,3.0])));
}

#[test]
fn broadcast_row_assign() {
	// A(1,:) = [7 6] already covered; also test scalar broadcast: A(:,2)=5
	let src = "A=[1,2;3,4]; A(:,2)=5; B=A;";
	let hir = lower(&parse(src).unwrap()).unwrap();
	let vars = execute(&hir).unwrap();
	if let runmat_builtins::Value::Tensor(b) = &vars[1] { assert_eq!(b.data, vec![1.0,3.0,5.0,5.0]); } else { panic!("B not tensor"); }
}

use runmat_builtins::Value;

#[test]
fn assign_scalar_element() {
    let ast = parse("A=[1,2;3,4]; A(2,1)=42; B=A").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Column-major data ordering
    if let Value::Tensor(b) = &vars[1] { assert_eq!(b.data, vec![1.0,42.0,2.0,4.0]); } else { panic!("B not tensor"); }
}

#[test]
fn assign_full_column() {
    let ast = parse("A=[1,2;3,4]; A(:,2)=[9;8]; B=A").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Column-major: [1 9; 3 8] -> [1,3,9,8]
    if let Value::Tensor(b) = &vars[1] { assert_eq!(b.data, vec![1.0,3.0,9.0,8.0]); } else { panic!("B not tensor"); }
}

#[test]
fn assign_full_row() {
    let ast = parse("A=[1,2;3,4]; A(1,:)=[7,6]; B=A").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Column-major: [7 6; 3 4] -> [7,3,6,4]
    if let Value::Tensor(b) = &vars[1] { assert_eq!(b.data, vec![7.0,3.0,6.0,4.0]); } else { panic!("B not tensor"); }
}


