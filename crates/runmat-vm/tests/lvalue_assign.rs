#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
use test_helpers::{compile_source, execute_source, interpret};

fn binding_slot(bytecode: &runmat_vm::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(&slot, candidate)| (candidate == name).then_some(slot))
        .expect("named binding")
}

#[test]
fn fixed_target_expansion_uses_prepared_one_output_assignment() {
    for source in [
        "A=[1,2;3,4]; I={1,2}; A(I{:})=9; assert(isequal(A,[1,9;3,4]));",
        "A=[1,2,3,4]; I={2}; A(I{:})=[]; assert(isequal(A,[1,3,4]));",
        "A=[1,2;3,4]; I={2}; A(:,I{:})=7; assert(isequal(A,[1,7;3,7]));",
    ] {
        let bytecode = compile_source(source).expect("compile expanded fixed target");
        assert!(bytecode.instructions.iter().any(|instruction| matches!(
            instruction,
            runmat_vm::Instr::FinishFixedAssignmentTarget { .. }
        )));
        execute_source(source).expect("execute expanded fixed target");
    }
    assert!(compile_source("A=[1,2]; I={1}; A(end,I{:})=9;").is_err());
}

#[test]
fn chained_member_and_index_assignments() {
    // Register test classes and construct an OverIdx object, then chain operations
    let src = "__register_test_classes(); o = new_object('OverIdx'); try; o(1)=42; catch e; end; o.k = 7; z = o.k;";
    let vars = execute_source(src).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn deep_chain_with_try_catch() {
    // obj(1).field{1} = 3; with try/catch around indexing
    let src = "__register_test_classes(); o = new_object('OverIdx'); try; o(1).f{1}=3; catch e; ok=1; end;";
    let vars = execute_source(src).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn brace_get_and_set_on_object() {
    // Test obj{1}=v and then read obj{1}
    let src = "__register_test_classes(); o = new_object('OverIdx'); o{1} = 5; r = o{1};";
    let vars = execute_source(src).unwrap();
    assert!(vars.iter().any(|v| match v {
        runmat_value::Value::Num(n) => (*n - 5.0).abs() < 1e-9,
        runmat_value::Value::OutputList(values) => values
            .iter()
            .any(|value| matches!(value, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)),
        _ => false,
    }));
}

#[test]
fn colon_slice_and_broadcast_assign() {
    // Test colon slice read and full-column/row writes
    let src = "A=[1,2;3,4]; y = A(:,2); A(:,2)=[9;8]; C=A; A(1,:)=[7,6]; D=A;";
    let vars = execute_source(src).unwrap();
    // y should be column vector [2;4] -> data [2,4]
    assert!(vars.iter().any(
        |v| matches!(v, runmat_value::Value::Tensor(t) if t.materialize_f64() == vec![2.0,4.0])
    ));
    // C should be [1 9; 3 8] -> [1,3,9,8]
    assert!(vars.iter().any(
        |v| matches!(v, runmat_value::Value::Tensor(t) if t.materialize_f64() == vec![1.0,3.0,9.0,8.0])
    ));
    // D should be [7 6; 3 8] -> [7,3,6,8]
    assert!(vars.iter().any(
        |v| matches!(v, runmat_value::Value::Tensor(t) if t.materialize_f64() == vec![7.0,3.0,6.0,8.0])
    ));
}

#[test]
fn logical_mask_indexing_chain() {
    // Logical mask on second column selection then assigning via chain: A(:, [true false]) = A(:,[true false])
    let src = "A=[1,2;3,4]; mask=[true false]; B=A(:,mask);";
    let vars = execute_source(src).unwrap();
    // B should be first column [1;3]
    assert!(vars.iter().any(
        |v| matches!(v, runmat_value::Value::Tensor(t) if t.materialize_f64() == vec![1.0,3.0])
    ));
}

#[test]
fn broadcast_row_assign() {
    // A(1,:) = [7 6] already covered; also test scalar broadcast: A(:,2)=5
    let src = "A=[1,2;3,4]; A(:,2)=5; B=A;";
    let vars = execute_source(src).unwrap();
    assert!(vars.iter().any(
        |v| matches!(v, runmat_value::Value::Tensor(t) if t.materialize_f64() == vec![1.0, 3.0, 5.0, 5.0])
    ));
}

#[test]
fn assign_scalar_element() {
    let vars = execute_source("A=[1,2;3,4]; A(2,1)=42; B=A").unwrap();
    // Column-major data ordering
    assert!(vars.iter().any(
        |v| matches!(v, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 42.0, 2.0, 4.0])
    ));
}

#[test]
fn assign_full_column() {
    let vars = execute_source("A=[1,2;3,4]; A(:,2)=[9;8]; B=A").unwrap();
    // Column-major: [1 9; 3 8] -> [1,3,9,8]
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 3.0, 9.0, 8.0])));
}

#[test]
fn assign_full_row() {
    let vars = execute_source("A=[1,2;3,4]; A(1,:)=[7,6]; B=A").unwrap();
    // Column-major: [7 6; 3 4] -> [7,3,6,4]
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Tensor(t) if t.materialize_f64() == vec![7.0, 3.0, 6.0, 4.0])));
}

#[test]
fn vector_index_assignment_lowers_to_store_slice() {
    let bytecode = compile_source("A=[10,20,30,40]; idx=[2,4]; A(idx)=99; B=A;")
        .expect("compile vector index assignment");
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreSlice(..) | runmat_vm::Instr::StoreSliceDelete(..)
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
    let vars = execute_source("A=[10,20,30,40]; idx=[2,4]; A(idx)=99; B=A;")
        .expect("execute vector index assignment");
    assert!(
        vars.iter().any(
            |v| matches!(v, Value::Tensor(t) if t.materialize_f64() == vec![10.0, 99.0, 30.0, 99.0])
        ),
        "vector index assignment should mutate selected linear indices"
    );
}

#[test]
fn logical_mask_assignment_lowers_to_store_slice() {
    let bytecode = compile_source("A=[1,2,3,4]; mask=logical([1,0,1,0]); A(mask)=0; B=A;")
        .expect("compile logical mask assignment");
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreSlice(..) | runmat_vm::Instr::StoreSliceDelete(..)
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
    let vars = execute_source("A=[1,2,3,4]; mask=logical([1,0,1,0]); A(mask)=0; B=A;")
        .expect("execute logical mask assignment");
    assert!(
        vars.iter().any(
            |v| matches!(v, Value::Tensor(t) if t.materialize_f64() == vec![0.0, 2.0, 0.0, 4.0])
        ),
        "logical mask assignment should mutate true mask positions"
    );
}

#[test]
fn numeric_scalar_base_store_slice_preserves_growth_alias_and_empty_selection() {
    let source = "a=1; old=a; k=[1 3]; a(k)=[4 7]; empty=[]; a(empty)=9; ok=isequal(size(a),[1 3]) && isequal(a,[4 0 7]) && old==1;";
    let bytecode = compile_source(source).expect("compile scalar-base slice store");
    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))));
    let ok_slot = binding_slot(&bytecode, "ok");
    let vars = interpret(&bytecode).expect("execute scalar-base slice store");
    assert!(
        matches!(vars.get(ok_slot), Some(Value::Bool(true))),
        "unexpected result: {:?}",
        vars.get(ok_slot)
    );
}

#[test]
fn computed_bounded_store_detaches_alias_before_read_modify_write() {
    use runmat_value::NumericDType;

    for (constructor, dtype) in [
        ("double", NumericDType::F64),
        ("single", NumericDType::F32),
        ("int32", NumericDType::I32),
    ] {
        let source = format!(
            "A={constructor}(zeros(10)); A(3,3)=1; B=A; d=[2 25]; \
             for j=1:1, \
             B(2*d(j,1)-1,2*d(j,1)-1)=B(2*d(j,1)-1,2*d(j,1)-1)+d(j,2); \
             end;"
        );
        let bytecode = compile_source(&source).expect("compile computed bounded store");
        assert!(bytecode
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, runmat_vm::Instr::StoreSlice(..))));
        let a = binding_slot(&bytecode, "A");
        let b = binding_slot(&bytecode, "B");
        let values = interpret(&bytecode).expect("execute computed bounded store");
        for (slot, expected) in [(a, 1.0), (b, 26.0)] {
            let Some(Value::Tensor(tensor)) = values.get(slot) else {
                panic!("missing {constructor} matrix at {slot}");
            };
            assert_eq!(tensor.numeric_dtype(), dtype);
            assert_eq!(tensor.shape(), &[10, 10]);
            let mut expected_values = vec![0.0; 100];
            expected_values[22] = expected;
            assert_eq!(tensor.materialize_f64(), expected_values, "{constructor}");
        }
    }
}

#[test]
fn integer_scalar_base_store_slice_keeps_exact_native_classes() {
    let cases: &[(&str, IntegerStorage)] = &[
        ("int8(-8)", IntegerStorage::I8(vec![-8, 3])),
        ("int16(-16)", IntegerStorage::I16(vec![-16, 3])),
        ("int32(-32)", IntegerStorage::I32(vec![-32, 3])),
        ("int64(-64)", IntegerStorage::I64(vec![-64, 3])),
        ("uint8(8)", IntegerStorage::U8(vec![8, 3])),
        ("uint16(16)", IntegerStorage::U16(vec![16, 3])),
        ("uint32(32)", IntegerStorage::U32(vec![32, 3])),
        (
            "uint64(9007199254740992)+uint64(1)",
            IntegerStorage::U64(vec![9007199254740993, 3]),
        ),
    ];
    for (scalar, expected) in cases {
        let source = format!("a={scalar}; old=a; k=[2 2]; a(k)=3; ok=isequal(old,{scalar});");
        let bytecode = compile_source(&source).expect("compile integer scalar slice store");
        assert!(
            bytecode
                .instructions
                .iter()
                .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
            "missing StoreSlice for {scalar}"
        );
        let a_slot = binding_slot(&bytecode, "a");
        let ok_slot = binding_slot(&bytecode, "ok");
        let vars = interpret(&bytecode).expect("execute integer scalar slice store");
        let Some(Value::Tensor(actual)) = vars.get(a_slot) else {
            panic!("missing tensor result for {scalar}");
        };
        assert_eq!(actual.shape(), &[1, 2], "shape for {scalar}");
        assert_eq!(
            actual.integer_storage(),
            Some(expected),
            "storage for {scalar}"
        );
        assert!(
            matches!(vars.get(ok_slot), Some(Value::Bool(true))),
            "alias changed for {scalar}"
        );
    }
}

#[test]
fn numeric_scalar_slice_deletion_uses_bounded_plan() {
    let source = "a=5; a(1:1)=[]; ok=isempty(a);";
    let bytecode = compile_source(source).expect("compile scalar slice deletion");
    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, runmat_vm::Instr::StoreSliceDelete(..))));
    let ok_slot = binding_slot(&bytecode, "ok");
    let vars = interpret(&bytecode).expect("execute scalar slice deletion");
    assert!(
        matches!(vars.get(ok_slot), Some(Value::Bool(true))),
        "unexpected result: {:?}",
        vars.get(ok_slot)
    );
}
