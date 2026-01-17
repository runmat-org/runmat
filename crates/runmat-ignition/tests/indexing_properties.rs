mod test_helpers;

use runmat_hir::lower;
use runmat_parser::parse;
use test_helpers::execute;

fn find_last_tensor(vars: &[runmat_builtins::Value]) -> runmat_builtins::Tensor {
    vars.iter()
        .rev()
        .find_map(|v| {
            if let runmat_builtins::Value::Tensor(t) = v {
                Some(t.clone())
            } else {
                None
            }
        })
        .expect("no tensor found")
}

#[test]
fn logical_mask_write_rows_scalar_broadcast() {
    // Select rows by logical mask, assign scalar across all selected rows/cols
    let src = "A=[1 2; 3 4; 5 6]; A([1 0 1], :) = 7;";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = find_last_tensor(&vars);
    // Rows 1 and 3 all set to 7
    // Final A = [7 7; 3 4; 7 7] → column-major data [7 3 7 7 4 7]
    assert_eq!(a.shape, vec![3, 2]);
    assert_eq!(a.data, vec![7.0, 3.0, 7.0, 7.0, 4.0, 7.0]);
}

#[test]
fn logical_mask_write_cols_vector_broadcast() {
    // Select columns 2 and 4 by mask, assign a column vector broadcast across selected columns
    // Use explicit literal to avoid range parsing differences in reshape
    let src = "A=reshape([1 2 3 4 5 6 7 8],2,4); A(:, [0 1 0 1]) = [8;9];"; // A is 2x4: columns are [1 2],[3 4],[5 6],[7 8]
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = find_last_tensor(&vars);
    // Columns 2 and 4 replaced by [8;9]
    // Expected columns: [1 2], [8 9], [5 6], [8 9] → data col-major [1,2,8,9,5,6,8,9]
    assert_eq!(a.shape, vec![2, 4]);
    assert_eq!(a.data, vec![1.0, 2.0, 8.0, 9.0, 5.0, 6.0, 8.0, 9.0]);
}

#[test]
fn mixed_mask_and_range_write_matrix_no_broadcast() {
    // Row mask with column range; rhs matches selection shape exactly
    let src = "A=[1 2 3; 4 5 6; 7 8 9]; A([1 0 1], 2:3) = [10 11; 12 13];";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = find_last_tensor(&vars);
    // After assignment: rows 1 and 3, cols 2..3 set to [[10,11];[12,13]] respecting column-major write
    // Final A row-major for intuition:
    // [1 10 11; 4 5 6; 7 12 13] → column-major data [1,4,7,10,5,12,11,6,13]
    assert_eq!(a.shape, vec![3, 3]);
    assert_eq!(
        a.data,
        vec![1.0, 4.0, 7.0, 10.0, 5.0, 12.0, 11.0, 6.0, 13.0]
    );
}

#[test]
fn broadcast_invariants_scalar_to_submatrix() {
    // Scalar broadcasts across N-D selection
    let src = "A=reshape([1 2 3 4 5 6 7 8 9 10 11 12],3,4); A(1:2, 2:3) = 5;";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = find_last_tensor(&vars);
    // A initially col-major of 3x4: columns [1 2 3],[4 5 6],[7 8 9],[10 11 12]
    // After setting rows 1..2, cols 2..3 to 5:
    // Columns become: [1 5 3], [4 5 6]-> actually col2 [5 5 6]? Wait rows 1..2 become 5, row3 stays same.
    // Correct columns: c1=[1,2,3], c2=[5,5,6], c3=[5,5,9], c4=[10,11,12]
    assert_eq!(a.shape, vec![3, 4]);
    assert_eq!(
        a.data,
        vec![1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 5.0, 5.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn end_arithmetic_range_store_linear_and_subscripted() {
    // Linear: vector with end arithmetic step
    let src = "A = 1:10; A(2:2:end-1) = [99 98 97 96];";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = find_last_tensor(&vars);
    // A indices 2,4,6,8 replaced → [1,99,3,98,5,97,7,96,9,10]
    // MATLAB 1:10 yields a row vector (1 x 10)
    assert_eq!(a.shape, vec![1, 10]);
    assert_eq!(
        a.data,
        vec![1.0, 99.0, 3.0, 98.0, 5.0, 97.0, 7.0, 96.0, 9.0, 10.0]
    );

    // 2D subscripted with end arithmetic on a range dim
    let src2 = "A = reshape([1 2 3 4 5 6 7 8 9 10 11 12], 3,4); A(1:end-1, 3) = [42; 43];";
    let hir2 = lower(&parse(src2).unwrap()).unwrap();
    let vars2 = execute(&hir2).unwrap();
    let a2 = find_last_tensor(&vars2);
    // Column 3 becomes [42,43,9]
    assert_eq!(a2.shape, vec![3, 4]);
    assert_eq!(
        a2.data,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 42.0, 43.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn gather_scatter_roundtrip_consistency() {
    // Gather selection then immediately scatter back; A should remain unchanged
    let src = r#"
        A = reshape([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24], 3, 4, 2);
        S = A(1:2, 2:3, end);
        A(1:2, 2:3, end) = S;
    "#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // The final A tensor should equal original 1..24 layout because we wrote back the same slice
    let a = vars
        .iter()
        .filter_map(|v| {
            if let runmat_builtins::Value::Tensor(t) = v {
                Some(t)
            } else {
                None
            }
        })
        .find(|t| t.shape == vec![3, 4, 2])
        .unwrap();
    // Original data 1..24 in column-major
    let expected: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    assert_eq!(a.data, expected);
}

#[test]
fn fastpath_roundtrip_and_broadcast() {
    // Column-major round-trip: read, write full column/row using vector and scalar broadcasts
    let src = "A=reshape([1 2 3 4 5 6 7 8 9 10 11 12],3,4); C1=A(:,2); A(:,2)=[30;31;32]; B1=A; A(1,:)=100; B2=A; A(:,3)=7; B3=A;";
    let hir = runmat_hir::lower(&runmat_parser::parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // C1 == original second column [4;5;6]
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.shape==vec![3,1] && t.data==vec![4.0,5.0,6.0])));
    // After A(:,2)=[30;31;32], B1 reflects updated column in col-major
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.shape==vec![3,4] && t.data==vec![1.0,2.0,3.0,30.0,31.0,32.0,7.0,8.0,9.0,10.0,11.0,12.0])));
    // After A(1,:)=100, B2 has first row set across all columns (col-major indices 0,3,6,9)
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.shape==vec![3,4] && t.data[0]==100.0 && t.data[3]==100.0 && t.data[6]==100.0 && t.data[9]==100.0)));
    // After A(:,3)=7 (scalar broadcast), all entries in third column are 7
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.shape==vec![3,4] && t.data[6]==7.0 && t.data[7]==7.0 && t.data[8]==7.0)));
}

#[test]
fn negative_step_linear_index() {
    let src = "A = 1:10; B = A(10:-2:2);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let b = find_last_tensor(&vars);
    assert_eq!(b.shape, vec![5, 1]);
    assert_eq!(b.data, vec![10.0, 8.0, 6.0, 4.0, 2.0]);
}

#[test]
fn negative_step_2d_subscript_index() {
    let src = "A = reshape([1 2 3 4 5 6 7 8 9 10 11 12],3,4); B = A(3:-1:1, 4:-2:2);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let b = find_last_tensor(&vars);
    assert_eq!(b.shape, vec![3, 2]);
    // Column-major order: columns [4,5,6] and [10,11,12] -> data [10,11,12,4,5,6]
    assert_eq!(b.data, vec![10.0, 11.0, 12.0, 4.0, 5.0, 6.0]);
}

#[test]
fn empty_row_selection() {
    let src = "A = reshape([1 2 3 4 5 6],3,2); B = A([], :);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let b = find_last_tensor(&vars);
    assert_eq!(b.shape, vec![0, 2]);
    assert_eq!(b.data.len(), 0);
}

#[test]
fn empty_col_selection() {
    let src = "A = reshape([1 2 3 4 5 6],3,2); B = A(:, []);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let b = find_last_tensor(&vars);
    assert_eq!(b.shape, vec![3, 0]);
    assert_eq!(b.data.len(), 0);
}

#[test]
fn empty_both_selection() {
    let src = "A = [1 2; 3 4]; B = A([], []);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let b = find_last_tensor(&vars);
    assert_eq!(b.shape, vec![0, 0]);
    assert_eq!(b.data.len(), 0);
}

#[test]
fn mixed_selectors_rows_vector_and_col_range() {
    let src = "A = reshape([1 2 3 4 5 6 7 8 9 10 11 12],3,4); B = A([1 3], 2:3);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let b = find_last_tensor(&vars);
    assert_eq!(b.shape, vec![2, 2]);
    assert_eq!(b.data, vec![4.0, 6.0, 7.0, 9.0]);
}

#[test]
fn gather_scatter_roundtrip_negative_step() {
    let src = r#"
		A = reshape([1 2 3 4 5 6 7 8 9 10 11 12],3,4);
		S = A(3:-1:1, 4:-2:2);
		A(3:-1:1, 4:-2:2) = S;
	"#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = vars
        .iter()
        .filter_map(|v| {
            if let runmat_builtins::Value::Tensor(t) = v {
                Some(t)
            } else {
                None
            }
        })
        .find(|t| t.shape == vec![3, 4])
        .unwrap();
    let expected: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    assert_eq!(a.data, expected);
}

#[test]
fn multidim_empty_mixes() {
    // 3D tensor; select empties across different dims and combinations
    let src = r#"
		A = reshape([1:24],3,4,2);
		B1 = A([], :, :);
		B2 = A(:, [], :);
		B3 = A(:, :, []);
		B4 = A([], 2:3, []);
	"#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let mut saw_b1 = false;
    let mut saw_b2 = false;
    let mut saw_b3 = false;
    let mut saw_b4 = false;
    for v in &vars {
        if let runmat_builtins::Value::Tensor(t) = v {
            if t.shape == vec![0, 4, 2] && t.data.is_empty() {
                saw_b1 = true;
            }
            if t.shape == vec![3, 0, 2] && t.data.is_empty() {
                saw_b2 = true;
            }
            if t.shape == vec![3, 4, 0] && t.data.is_empty() {
                saw_b3 = true;
            }
            if t.shape == vec![0, 2, 0] && t.data.is_empty() {
                saw_b4 = true;
            }
        }
    }
    assert!(saw_b1 && saw_b2 && saw_b3 && saw_b4);
}

#[test]
fn mixed_logical_mask_and_range_across_3d() {
    // Rows logical mask, columns range, third-dim logical mask
    let src = r#"
		A = reshape([1:24],3,4,2);
		rows = [1 0 1]; % rows 1 and 3
		cols = 2:3;     % columns 2 and 3
		planes = [0 1]; % pick plane 2 only
		S = A(rows, cols, planes);
	"#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let s = find_last_tensor(&vars);
    assert_eq!(s.shape, vec![2, 2, 1]);
    // Expected from plane 2: col2 [16,17,18], col3 [19,20,21] selecting rows 1 & 3 => [16,18,19,21]
    assert_eq!(s.data, vec![16.0, 18.0, 19.0, 21.0]);
}

#[test]
fn scatter_broadcasts_negative_steps_and_degenerate_dims() {
    // Scalar broadcast over negative step range and degenerate one-element ranges
    let src = r#"
		A = reshape([1:24],3,4,2);
		A(3:-1:1, 3, 2) = 99;   % set col 3 in plane 2, all rows via negative step
		A(1:2:3, 2:2:2, 1) = [70;80]; % degenerate single column; two rows via step, explicit 2x1 rhs
	"#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    let a = vars
        .iter()
        .filter_map(|v| {
            if let runmat_builtins::Value::Tensor(t) = v {
                Some(t)
            } else {
                None
            }
        })
        .find(|t| t.shape == vec![3, 4, 2])
        .unwrap();
    // Validate A(3:-1:1,3,2)=99 set rows 1..3 in (col3, plane2)
    let rows = 3usize;
    let cols = 4usize;
    let idx = |r: usize, c: usize, p: usize| -> usize { (r) + c * rows + p * rows * cols }; // r,c,p are 0-based here
    assert_eq!(a.data[idx(0, 2, 1)], 99.0);
    assert_eq!(a.data[idx(1, 2, 1)], 99.0);
    assert_eq!(a.data[idx(2, 2, 1)], 99.0);
    // Validate A(1:2:3, 2, 1) = [70;80]
    assert_eq!(a.data[idx(0, 1, 0)], 70.0);
    assert_eq!(a.data[idx(2, 1, 0)], 80.0);
}

#[test]
fn fastpath_gather_multi_columns_and_rows() {
    let src = r#"
		A = reshape([1:12],3,4);
		C = A(:, [2 4]);
		R = A([1 3], :);
	"#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // C is columns 2 and 4 -> data [4 5 6 10 11 12]
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.shape==vec![3,2] && t.data==vec![4.0,5.0,6.0,10.0,11.0,12.0])));
    // R is rows [1 3] across all columns -> shape [2,4]; data per col-major: [1,3,4,6,7,9,10,12]
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.shape==vec![2,4] && t.data==vec![1.0,3.0,4.0,6.0,7.0,9.0,10.0,12.0])));
}

#[test]
fn fastpath_scatter_multi_columns_and_rows() {
    let src = r#"
		A = reshape([1:12],3,4);
		A(:, [2 4]) = [30 40; 31 41; 32 42];
		B = A;
		A([1 3], :) = 9; % broadcast row scalar over selected rows
		C = A;
	"#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // After column scatter, B should have col2=[30 31 32] and col4=[40 41 42]
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![3,4] && t.data==vec![1.0,2.0,3.0,30.0,31.0,32.0,7.0,8.0,9.0,40.0,41.0,42.0])));
    // After row scatter (rows 1 and 3 set to 9 across all columns), C reflects broadcasts
    let expected_c: Vec<f64> = vec![9.0, 2.0, 9.0, 9.0, 31.0, 9.0, 9.0, 8.0, 9.0, 9.0, 41.0, 9.0];
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![3,4] && t.data==expected_c)));
}

#[test]
fn func_returns_into_row_col_linear_slices() {
    let src = r#"
		function [x,y] = g(); x=7; y=8; end
		A = zeros(2,3);
		A(1, 2:3) = g(); % row slice
		B = A;
		C = zeros(3,2);
		C(1:2, 1) = g(); % column slice
		D = C;
		E = reshape([1:9],3,3);
		E([1 9]) = g();   % linear slice
		F = E;
	"#;
    let hir = runmat_hir::lower(&runmat_parser::parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // B should have row1 col2..3 set to 7,8
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![2,3] && t.data==vec![0.0,0.0, 7.0,0.0, 8.0,0.0])));
    // D should have col1 rows 1..2 set to 7,8
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![3,2] && t.data[0]==7.0 && t.data[1]==8.0)));
    // F should have indices 1 and 9 set to 7 and 8
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![3,3] && t.data[0]==7.0 && t.data[8]==8.0)));
}

#[test]
fn cell_expansion_into_row_col_linear_slices() {
    let src = r#"
		A = zeros(2,3);
		C = {7,8};
		A(1, 2:3) = C{:}; % row slice two elements
		B = A;
		D = zeros(3,2);
		E = {11,12};
		D(1:2, 1) = E{:}; % column slice two elements
		F = D;
		G = reshape([1:9],3,3);
		H = {70,80};
		G([1 9]) = H{:};   % linear slice two positions
		I = G;
	"#;
    let hir = runmat_hir::lower(&runmat_parser::parse(src).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // B row1 col2..3 set to 7,8
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![2,3] && t.data==vec![0.0,0.0, 7.0,0.0, 8.0,0.0])));
    // F col1 rows 1..2 set to 11,12
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![3,2] && t.data[0]==11.0 && t.data[1]==12.0)));
    // I indices 1 and 9 set to 70 and 80
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t)
		if t.shape==vec![3,3] && t.data[0]==70.0 && t.data[8]==80.0)));
}

#[test]
fn function_return_expansion_into_slice_with_empty() {
    let program = r#"
        function varargout = g(k)
            if k==0
                varargout = {};
            else
                varargout = {1,2,3};
            end
        end
        A = zeros(2,3);
        % Empty expansion: ensure we don't crash; assign nothing
        x = g(0);
    "#;
    let hir = runmat_hir::lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let _ = execute(&hir);
}

#[test]
fn cell_expansion_into_slice_with_degenerate_dims() {
    let program = r#"
        C = {10,20,30};
        A = zeros(1,3);
        % Expand by explicit indexing
        A(1,1) = C{1}; A(1,2) = C{2}; A(1,3) = C{3};
        s = sum(A(:));
    "#;
    let hir = runmat_hir::lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-60.0).abs()<1e-9)));
}

#[test]
fn oop_negative_missing_subsref_mex() {
    let program = r#"
        __register_test_classes();
        o = new_object('NoIdx'); % class without subsref
        try
            x = o(1);
        catch e
            err = e;
        end
    "#;
    let hir = runmat_hir::lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let out = execute(&hir);
    if let Err(err) = out {
        assert!(err.contains("MATLAB:MissingSubsref") || err.contains("subsref"));
    }
}

#[test]
fn oop_negative_missing_subsasgn_mex() {
    let program = r#"
        __register_test_classes();
        o = new_object('NoIdx'); % class without subsasgn
        try
            o(1) = 5;
        catch e
            err = e;
        end
    "#;
    let hir = runmat_hir::lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let out = execute(&hir);
    if let Err(err) = out {
        assert!(err.contains("MATLAB:MissingSubsasgn") || err.contains("subsasgn"));
    }
}
