#[path = "support/mod.rs"]
mod test_helpers;

use runmat_vm::Instr;
use test_helpers::{compile_source, interpret};

fn execute_with_dynamic_concat(source: &str) -> runmat_runtime::execution::RootedValues {
    let bytecode = compile_source(source).expect("compile aggregate source");
    let has_dynamic = |instructions: &[Instr]| {
        instructions
            .iter()
            .any(|instruction| matches!(instruction, Instr::CreateMatrixFromSequences { .. }))
    };
    assert!(
        has_dynamic(&bytecode.instructions)
            || bytecode
                .bound_functions
                .values()
                .any(|function| has_dynamic(&function.instructions)),
        "a potentially nonprimitive aggregate element must use dynamic concatenation"
    );
    interpret(&bytecode).expect("execute aggregate source")
}

#[test]
fn filter_multioutput_continuation_concatenates_numeric_arrays() {
    execute_with_dynamic_concat(
        "b = [1/3 1/3 1/3]; a = 1; x1 = [1 5 2]; [y1,zf] = filter(b,a,x1); x2 = [0 3]; [y2,zf2] = filter(b,a,x2,zf); joined = [y1 y2]; assert(isequal(size(joined),[1 5])); assert(max(abs(joined - [1/3 2 8/3 7/3 5/3])) < 1e-12);",
    );
}

#[test]
fn alias_and_binary_array_results_use_dynamic_concatenation() {
    for source in [
        "x = [1 2]; alias = x; changed = x + 1; joined = [alias changed]; assert(isequal(joined,[1 2 2 3]));",
        "a = ones(3,1)*2; b = ones(3,1)*2; z = [a,b]; assert(isequal(z,2*ones(3,2)));",
        "a = ones(3,1)*2; b = ones(3,1)*2; z = [a;b]; assert(isequal(z,2*ones(6,1)));",
        "e = [1;2;3]; z = [-e]; assert(isequal(z,[-1;-2;-3]));",
        "x = (1:4)'; z = [x x]; assert(isequal(z,[1 1;2 2;3 3;4 4]));",
    ] {
        execute_with_dynamic_concat(source);
    }
}

#[test]
fn first_scalar_assignment_does_not_prove_reassigned_array_scalar() {
    execute_with_dynamic_concat(
        "x = 1; x = [2 3]; joined = [x 4]; assert(isequal(joined,[2 3 4]));",
    );
}

#[test]
fn parameter_aggregate_element_can_be_an_array() {
    execute_with_dynamic_concat(
        "function y = append(x)\n y = [x 3];\nend\njoined = append([1 2]); assert(isequal(joined,[1 2 3]));",
    );
}

#[test]
fn empty_typed_array_does_not_enter_primitive_constructor() {
    execute_with_dynamic_concat(
        "empty = uint8(zeros(1,0)); joined = [empty uint8(7)]; assert(isa(joined,'uint8')); assert(numel(joined)==1); assert(joined(1)==uint8(7));",
    );
}

#[test]
fn wide_exact_integer_array_survives_dynamic_concatenation() {
    execute_with_dynamic_concat(
        "base = uint64(9007199254740992); a = base + uint64(1); joined = [a a+uint64(2)]; assert(isa(joined,'uint64')); assert(joined(1)==a); assert(joined(2)==a+uint64(2));",
    );
}

#[test]
fn primitive_numeric_and_complex_literals_keep_fixed_constructor() {
    for (source, rows, columns) in [("a = [1 2; 3 4];", 2, 2), ("a = [1+2i 3-4j];", 1, 2)] {
        let bytecode = compile_source(source).expect("compile primitive literal");
        assert!(bytecode.instructions.iter().any(|instruction| {
            matches!(instruction, Instr::CreateMatrix(actual_rows, actual_columns)
                if *actual_rows == rows && *actual_columns == columns)
        }));
        interpret(&bytecode).expect("execute primitive literal");
    }
}

#[test]
fn primitive_scalar_operator_results_keep_fixed_constructor() {
    let source = "a = [1+2 3*4 8/2 2^3 1<2]; assert(isequal(a,[3 12 4 8 1]));";
    let bytecode = compile_source(source).expect("compile primitive scalar operators");
    assert!(bytecode
        .instructions
        .iter()
        .any(|instruction| matches!(instruction, Instr::CreateMatrix(1, 5))));
    interpret(&bytecode).expect("execute primitive scalar operators");
}

#[test]
fn loop_aggregate_has_exact_join_geometry_and_single_operand_arity() {
    let source = "for i = 1:3\n matrix = [i,i+1;i+2,i+3];\nend\nassert(isequal(matrix,[3 4;5 6]));";
    let bytecode = compile_source(source).expect("compile loop aggregate without join-depth drift");
    let aggregate = bytecode.instructions.iter().find(|instruction| matches!(instruction,
        Instr::CreateMatrixFromSequences { rows: 2, row_lengths, elements }
        if row_lengths == &[2, 2] && elements.iter().all(|element| element.stack_operand_count() == 1)
    )).expect("typed aggregate metadata");
    let effect = aggregate.stack_effect().unwrap();
    assert_eq!((effect.pops, effect.pushes), (4, 1));
    assert!(bytecode.stack_geometry.exact_maximum().is_some());
    interpret(&bytecode).expect("ordinary loop preserves matrix values");
}
