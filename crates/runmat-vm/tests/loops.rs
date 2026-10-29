#[path = "support/mod.rs"]
mod test_helpers;

use runmat_vm::Instr;
use std::convert::TryInto;
use test_helpers::compile_source;
use test_helpers::interpret;

fn execute_program(source: &str) -> runmat_runtime::execution::RootedValues {
    let bytecode = compile_source(source).expect("compile source");
    interpret(&bytecode).expect("execute bytecode")
}

#[test]
fn while_loop_decrements() {
    let vars = execute_program("x=3; y=0; while x; y=y+1; x=x-1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 0.0);
    assert_eq!(y, 3.0);
}

#[test]
fn for_loop_sum() {
    let vars = execute_program("s=0; for i=1:4; s=s+i; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 10.0);
}

#[test]
fn for_loop_with_step_and_descending() {
    // Ascending with step
    let vars = execute_program("s=0; for i=1:2:5; s=s+i; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 1.0 + 3.0 + 5.0);

    // Descending with negative step
    let vars = execute_program("s=0; for i=5:-2:0; s=s+i; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 5.0 + 3.0 + 1.0);
}

#[test]
fn for_loop_zero_step_terminates() {
    // Step=0 should not loop
    let vars = execute_program("s=0; for i=1:0:10; s=s+1; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 0.0);
}

#[test]
fn while_false_never_enters() {
    let vars = execute_program("x=0; while 0; x=x+1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 0.0);
}

#[test]
fn for_start_equals_end_executes_once() {
    let vars = execute_program("x=0; for i=5:5; x=x+1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn for_negative_step_no_iterations_when_increasing() {
    let vars = execute_program("x=0; for i=1:-1:5; x=x+1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 0.0);
}

#[test]
fn scalar_for_publishes_only_executed_range_values() {
    for (source, expected) in [
        ("i=77; for i=1:3; end", 3.0),
        ("i=77; for i=1:2.5; end", 2.0),
        ("i=77; for i=5:-2:0; end", 1.0),
        ("i=77; for i=1:3; i=100; end", 100.0),
    ] {
        let vars = execute_program(source);
        let actual: f64 = (&vars[0]).try_into().unwrap();
        assert_eq!(actual, expected, "{source}");
    }
    for source in ["i=77; for i=4:2; end", "i=77; for i=1:0:3; end"] {
        let vars = execute_program(source);
        let runmat_value::Value::Tensor(index) = &vars[0] else {
            panic!("empty loop must publish an empty index: {source}");
        };
        assert!(index.is_empty(), "{source}");
        assert_eq!(index.shape(), &[0, 0], "{source}");
    }
}

#[test]
fn stochastic_evolution_loop_emits_instruction() {
    let source = "
    M = 32;
    S = ones(M, 1, 'single');
    drift = single(0.1);
    scale = single(0.2);
    for t = 1:8
        Z = randn(M, 1, 'single');
        S = S .* exp(drift + scale .* Z);
    end
    ";
    let bytecode = compile_source(source).unwrap();
    let (selected_pc, selected) = bytecode
        .instructions
        .iter()
        .enumerate()
        .find(|(_, instr)| matches!(instr, Instr::StochasticEvolution { .. }))
        .expect("recognized source loop keeps its typed selected site");
    let Instr::StochasticEvolution {
        steps,
        drift,
        scale,
        randn_args,
        fallback_pc,
        ..
    } = selected
    else {
        unreachable!()
    };
    assert!(matches!(
        steps,
        runmat_vm::StochasticSourceOperand::Number(8.0)
    ));
    assert!(matches!(drift, runmat_vm::StochasticSourceOperand::Slot(_)));
    assert!(matches!(scale, runmat_vm::StochasticSourceOperand::Slot(_)));
    assert_eq!(randn_args.len(), 3);
    assert!(matches!(
        bytecode.instructions[selected_pc + 1],
        Instr::Jump(_)
    ));
    assert!(*fallback_pc > selected_pc + 1);
}

#[test]
fn stochastic_source_recognizes_parameter_and_output_bindings() {
    let bytecode = compile_source(
        "[S,Z,t] = evolve(ones(2,1,'single'), single(0.1), single(0.2));
         function [S,Z,t] = evolve(S, drift, scale)
             for t = 1:3
                 Z = randn(2,1,'single');
                 S = S .* exp(drift + scale .* Z);
             end
         end",
    )
    .unwrap();
    let function = bytecode
        .function_registry
        .functions
        .values()
        .find(|function| function.display_name == "evolve")
        .expect("compiled function");
    assert!(function
        .instructions
        .iter()
        .any(|instruction| matches!(instruction, Instr::StochasticEvolution { .. })));
}

#[test]
fn stochastic_evolution_keeps_original_for_when_body_has_other_effects() {
    for body in [
        "Z = randn(M, 1, 'single'); S = S .* exp(drift + scale .* Z); q = q + 1;",
        "Z = randn(M, 1, 'single'); scale = scale + 1; S = S .* exp(drift + scale .* Z);",
        "Z = randn(M, 1, 'single'); S = S .* exp(drift + scale .* Z); disp(S);",
        "Z = randn(M, 1, 'single'); scaled = scale .* Z; argument = drift + scaled; factor = exp(argument); S = S .* factor;",
    ] {
        let source = format!(
            "M = 2; S = ones(M, 1, 'single'); drift = single(0.1); \
             scale = single(0.2); q = 0; for t = 1:3; {body} end"
        );
        let bytecode = compile_source(&source).unwrap();
        assert!(
            !bytecode
                .instructions
                .iter()
                .any(|instr| matches!(instr, Instr::StochasticEvolution { .. })),
            "an extra loop effect must stay on the original For path: {body}"
        );
    }
}

#[test]
fn stochastic_snapshots_do_not_erase_named_aliases_or_shadowed_callbacks() {
    for source in [
        "M=2; S=ones(M,1,'single'); drift=single(0.1); scale=single(0.2); \
         for t=1:3; Z=randn(M,1,'single'); saved=S; S=S.*exp(drift+scale.*Z); end",
        "M=2; S=ones(M,1,'single'); drift=single(0.1); scale=single(0.2); \
         for t=1:3; Z=randn(M,1,'single'); S=S.*exp(drift+scale.*Z); end; \
         function y=exp(x); y=x; end",
    ] {
        let bytecode = compile_source(source).unwrap();
        assert!(
            !bytecode
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instr::StochasticEvolution { .. })),
            "only compiler-private copies and authenticated builtin calls may be selected: {source}"
        );
    }
}

#[test]
fn stochastic_evolution_loop_preserves_native_single_state() {
    let vars = execute_program(
        "
        M = 2;
        S = ones(M, 1, 'single');
        drift = single(0.1);
        scale = single(0);
        for t = 1:3
            Z = randn(M, 1, 'single');
            S = S .* exp(drift + scale .* Z);
        end
        ",
    );
    let runmat_value::Value::Tensor(state) = &vars[1] else {
        panic!("expected evolved tensor state, got {:?}", vars[1]);
    };
    assert_eq!(state.numeric_dtype(), runmat_value::NumericDType::F32);
    let factor = (0.1f32).exp();
    let expected = factor * factor * factor;
    for index in 0..state.len() {
        assert_eq!(
            state.numeric_value_at(index),
            Some(runmat_value::NumericScalar::F32(expected))
        );
    }
}
