#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::{compile_semantic_source, execute_semantic_source};

#[test]
fn chained_member_and_index_assignments() {
    // Register test classes and construct an OverIdx object, then chain operations
    let src = "__register_test_classes(); o = new_object('OverIdx'); try; o(1)=42; catch e; end; o.k = 7; z = o.k;";
    let vars = execute_semantic_source(src).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn deep_chain_with_try_catch() {
    // obj(1).field{1} = 3; with try/catch around indexing
    let src = "__register_test_classes(); o = new_object('OverIdx'); try; o(1).f{1}=3; catch e; ok=1; end;";
    let vars = execute_semantic_source(src).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn brace_get_and_set_on_object() {
    // Test obj{1}=v and then read obj{1}
    let src = "__register_test_classes(); o = new_object('OverIdx'); o{1} = 5; r = o{1};";
    let vars = execute_semantic_source(src).unwrap();
    assert!(vars.iter().any(|v| match v {
        runmat_builtins::Value::Num(n) => (*n - 5.0).abs() < 1e-9,
        runmat_builtins::Value::OutputList(values) => values.iter().any(
            |value| matches!(value, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)
        ),
        _ => false,
    }));
}

#[test]
fn colon_slice_and_broadcast_assign() {
    // Test colon slice read and full-column/row writes
    let src = "A=[1,2;3,4]; y = A(:,2); A(:,2)=[9;8]; C=A; A(1,:)=[7,6]; D=A;";
    let vars = execute_semantic_source(src).unwrap();
    // y should be column vector [2;4] -> data [2,4]
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![2.0,4.0])));
    // C should be [1 9; 3 8] -> [1,3,9,8]
    assert!(vars.iter().any(
        |v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![1.0,3.0,9.0,8.0])
    ));
    // D should be [7 6; 3 8] -> [7,3,6,8]
    assert!(vars.iter().any(
        |v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![7.0,3.0,6.0,8.0])
    ));
}

#[test]
fn logical_mask_indexing_chain() {
    // Logical mask on second column selection then assigning via chain: A(:, [true false]) = A(:,[true false])
    let src = "A=[1,2;3,4]; mask=[true false]; B=A(:,mask);";
    let vars = execute_semantic_source(src).unwrap();
    // B should be first column [1;3]
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![1.0,3.0])));
}

#[test]
fn broadcast_row_assign() {
    // A(1,:) = [7 6] already covered; also test scalar broadcast: A(:,2)=5
    let src = "A=[1,2;3,4]; A(:,2)=5; B=A;";
    let vars = execute_semantic_source(src).unwrap();
    assert!(vars.iter().any(
        |v| matches!(v, runmat_builtins::Value::Tensor(t) if t.data == vec![1.0, 3.0, 5.0, 5.0])
    ));
}

#[test]
fn assign_scalar_element() {
    let vars = execute_semantic_source("A=[1,2;3,4]; A(2,1)=42; B=A").unwrap();
    // Column-major data ordering
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Tensor(t) if t.data == vec![1.0, 42.0, 2.0, 4.0])));
}

#[test]
fn assign_full_column() {
    let vars = execute_semantic_source("A=[1,2;3,4]; A(:,2)=[9;8]; B=A").unwrap();
    // Column-major: [1 9; 3 8] -> [1,3,9,8]
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Tensor(t) if t.data == vec![1.0, 3.0, 9.0, 8.0])));
}

#[test]
fn assign_full_row() {
    let vars = execute_semantic_source("A=[1,2;3,4]; A(1,:)=[7,6]; B=A").unwrap();
    // Column-major: [7 6; 3 4] -> [7,3,6,4]
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Tensor(t) if t.data == vec![7.0, 3.0, 6.0, 4.0])));
}

#[test]
fn vector_index_assignment_lowers_to_store_slice() {
    let bytecode = compile_semantic_source("A=[10,20,30,40]; idx=[2,4]; A(idx)=99; B=A;")
        .expect("compile vector index assignment");
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreSlice(..)
                | runmat_vm::Instr::StoreSliceDelete(..)
                | runmat_vm::Instr::StoreSliceExpr { .. }
                | runmat_vm::Instr::StoreSliceExprDelete { .. }
        )),
        "vector index assignment should lower through StoreSlice* instructions"
    );
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "vector index assignment should not lower through StoreIndex* instructions"
    );
    let vars = execute_semantic_source("A=[10,20,30,40]; idx=[2,4]; A(idx)=99; B=A;")
        .expect("execute vector index assignment");
    assert!(
        vars.iter()
            .any(|v| matches!(v, Value::Tensor(t) if t.data == vec![10.0, 99.0, 30.0, 99.0])),
        "vector index assignment should mutate selected linear indices"
    );
}

#[test]
fn logical_mask_assignment_lowers_to_store_slice() {
    let bytecode = compile_semantic_source("A=[1,2,3,4]; mask=logical([1,0,1,0]); A(mask)=0; B=A;")
        .expect("compile logical mask assignment");
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreSlice(..)
                | runmat_vm::Instr::StoreSliceDelete(..)
                | runmat_vm::Instr::StoreSliceExpr { .. }
                | runmat_vm::Instr::StoreSliceExprDelete { .. }
        )),
        "logical mask assignment should lower through StoreSlice* instructions"
    );
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "logical mask assignment should not lower through StoreIndex* instructions"
    );
    let vars = execute_semantic_source("A=[1,2,3,4]; mask=logical([1,0,1,0]); A(mask)=0; B=A;")
        .expect("execute logical mask assignment");
    assert!(
        vars.iter()
            .any(|v| matches!(v, Value::Tensor(t) if t.data == vec![0.0, 2.0, 0.0, 4.0])),
        "logical mask assignment should mutate true mask positions"
    );
}
