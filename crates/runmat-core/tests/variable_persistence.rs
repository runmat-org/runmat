// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

use runmat_builtins::Value;
use runmat_core::RunMatSession;
use runmat_gc::gc_test_context;

#[test]
fn test_variable_persistence_basic() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Define a variable
        let result1 = runmat_core::execute_text_request_for_testing(&mut engine, "a = 10").unwrap();
        assert!(result1.error.is_none());
        if let Some(Value::Num(val)) = result1.value {
            assert_eq!(val, 10.0);
        } else {
            panic!("Expected Num(10.0), got {:?}", result1.value);
        }

        // Use the variable in an expression
        let result2 = runmat_core::execute_text_request_for_testing(&mut engine, "a + 5").unwrap();
        assert!(result2.error.is_none());
        if let Some(Value::Num(val)) = result2.value {
            assert_eq!(val, 15.0); // This should be 15.0!
        } else {
            panic!("Expected Num(15.0), got {:?}", result2.value);
        }

        // Access the variable directly
        let result3 = runmat_core::execute_text_request_for_testing(&mut engine, "a").unwrap();
        assert!(result3.error.is_none());
        if let Some(Value::Num(val)) = result3.value {
            assert_eq!(val, 10.0);
        } else {
            panic!("Expected Num(10.0), got {:?}", result3.value);
        }
    });
}

#[test]
fn test_variable_persistence_multiple_variables() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Define multiple variables
        let result1 = runmat_core::execute_text_request_for_testing(&mut engine, "x = 5").unwrap();
        assert!(result1.error.is_none());

        let result2 = runmat_core::execute_text_request_for_testing(&mut engine, "y = 3").unwrap();
        assert!(result2.error.is_none());

        let result3 = runmat_core::execute_text_request_for_testing(&mut engine, "z = 2").unwrap();
        assert!(result3.error.is_none());

        // Use them in expressions
        let result4 =
            runmat_core::execute_text_request_for_testing(&mut engine, "x + y + z").unwrap();
        assert!(result4.error.is_none());
        if let Some(Value::Num(val)) = result4.value {
            assert_eq!(val, 10.0); // 5 + 3 + 2 = 10
        } else {
            panic!("Expected Num(10.0), got {:?}", result4.value);
        }

        let result5 =
            runmat_core::execute_text_request_for_testing(&mut engine, "x * y * z").unwrap();
        assert!(result5.error.is_none());
        if let Some(Value::Num(val)) = result5.value {
            assert_eq!(val, 30.0); // 5 * 3 * 2 = 30
        } else {
            panic!("Expected Num(30.0), got {:?}", result5.value);
        }
    });
}

#[test]
fn test_variable_persistence_reassignment() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Initial assignment
        let result1 =
            runmat_core::execute_text_request_for_testing(&mut engine, "value = 100").unwrap();
        assert!(result1.error.is_none());

        // Use the variable
        let result2 =
            runmat_core::execute_text_request_for_testing(&mut engine, "value / 10").unwrap();
        assert!(result2.error.is_none());
        // Should be 10.0

        // Reassign the variable
        let result3 =
            runmat_core::execute_text_request_for_testing(&mut engine, "value = 50").unwrap();
        assert!(result3.error.is_none());

        // Use the new value
        let result4 =
            runmat_core::execute_text_request_for_testing(&mut engine, "value * 2").unwrap();
        assert!(result4.error.is_none());
        // Should be 100.0
    });
}

#[test]
fn test_expression_result_printing() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Constants should show results
        let result1 = runmat_core::execute_text_request_for_testing(&mut engine, "42").unwrap();
        assert!(result1.error.is_none());
        if let Some(Value::Num(val)) = result1.value {
            assert_eq!(val, 42.0);
        } else {
            panic!("Expected Num(42.0), got {:?}", result1.value);
        }

        // Arithmetic should show results
        let result2 =
            runmat_core::execute_text_request_for_testing(&mut engine, "10 + 20").unwrap();
        assert!(result2.error.is_none());
        if let Some(Value::Num(val)) = result2.value {
            assert_eq!(val, 30.0);
        } else {
            panic!("Expected Num(30.0), got {:?}", result2.value);
        }

        // Complex expressions should show results
        let result3 =
            runmat_core::execute_text_request_for_testing(&mut engine, "(5 + 3) * (10 - 2)")
                .unwrap();
        assert!(result3.error.is_none());
        if let Some(Value::Num(val)) = result3.value {
            assert_eq!(val, 64.0); // (5 + 3) * (10 - 2) = 8 * 8 = 64
        } else {
            panic!("Expected Num(64.0), got {:?}", result3.value);
        }
    });
}

#[test]
fn test_mixed_assignments_and_expressions() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Set up variables
        let result1 = runmat_core::execute_text_request_for_testing(&mut engine, "a = 7").unwrap();
        assert!(result1.error.is_none());

        let result2 = runmat_core::execute_text_request_for_testing(&mut engine, "b = 3").unwrap();
        assert!(result2.error.is_none());

        // Expression with variables
        let result3 = runmat_core::execute_text_request_for_testing(&mut engine, "a * b").unwrap();
        assert!(result3.error.is_none());
        if let Some(Value::Num(val)) = result3.value {
            assert_eq!(val, 21.0); // 7 * 3 = 21
        } else {
            panic!("Expected Num(21.0), got {:?}", result3.value);
        }

        // Assignment based on expression
        let result4 =
            runmat_core::execute_text_request_for_testing(&mut engine, "c = a + b").unwrap();
        assert!(result4.error.is_none());
        if let Some(Value::Num(val)) = result4.value {
            assert_eq!(val, 10.0); // 7 + 3 = 10
        } else {
            panic!("Expected Num(10.0), got {:?}", result4.value);
        }

        // Use the new variable
        let result5 = runmat_core::execute_text_request_for_testing(&mut engine, "c * 2").unwrap();
        assert!(result5.error.is_none());
        if let Some(Value::Num(val)) = result5.value {
            assert_eq!(val, 20.0); // 10 * 2 = 20
        } else {
            panic!("Expected Num(20.0), got {:?}", result5.value);
        }

        // All variables should still be accessible
        let result6 =
            runmat_core::execute_text_request_for_testing(&mut engine, "a + b + c").unwrap();
        assert!(result6.error.is_none());
        if let Some(Value::Num(val)) = result6.value {
            assert_eq!(val, 20.0); // 7 + 3 + 10 = 20
        } else {
            panic!("Expected Num(20.0), got {:?}", result6.value);
        }
    });
}

#[test]
fn test_variable_persistence_with_interpreter_only() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(false, false).unwrap(); // JIT disabled

        // Define a variable
        let result1 =
            runmat_core::execute_text_request_for_testing(&mut engine, "test_var = 123").unwrap();
        assert!(result1.error.is_none());
        assert!(!result1.used_jit); // Should use interpreter

        // Use the variable
        let result2 =
            runmat_core::execute_text_request_for_testing(&mut engine, "test_var + 77").unwrap();
        assert!(result2.error.is_none());
        assert!(!result2.used_jit); // Should use interpreter
                                    // Should be 200.0
    });
}

#[test]
fn test_variable_persistence_with_jit_hybrid() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap(); // JIT enabled

        // Assignment (should use JIT for assignments)
        let result1 =
            runmat_core::execute_text_request_for_testing(&mut engine, "jit_var = 456").unwrap();
        assert!(result1.error.is_none());

        // Expression (should use interpreter for expressions to capture results)
        let result2 =
            runmat_core::execute_text_request_for_testing(&mut engine, "jit_var - 56").unwrap();
        assert!(result2.error.is_none());
        // Should be 400.0

        // Another assignment using previous variable
        let result3 =
            runmat_core::execute_text_request_for_testing(&mut engine, "jit_var2 = jit_var / 4")
                .unwrap();
        assert!(result3.error.is_none());

        // Expression with both variables
        let result4 =
            runmat_core::execute_text_request_for_testing(&mut engine, "jit_var + jit_var2")
                .unwrap();
        assert!(result4.error.is_none());
        // Should be 570.0 (456 + 114)
    });
}

#[test]
fn test_large_number_of_variables() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Create many variables
        for i in 1..=10 {
            let cmd = format!("var{} = {}", i, i * 10);
            let result = runmat_core::execute_text_request_for_testing(&mut engine, &cmd).unwrap();
            assert!(result.error.is_none());
        }

        // Use them all in an expression
        let result = runmat_core::execute_text_request_for_testing(
            &mut engine,
            "var1 + var2 + var3 + var4 + var5",
        )
        .unwrap();
        assert!(result.error.is_none());
        // Should be 150.0 (10 + 20 + 30 + 40 + 50)

        // Use variables defined later
        let result =
            runmat_core::execute_text_request_for_testing(&mut engine, "var10 - var9 + var8")
                .unwrap();
        assert!(result.error.is_none());
        // Should be 90.0 (100 - 90 + 80)
    });
}

#[test]
fn test_zero_values_persistence() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Set a variable to zero
        let result1 =
            runmat_core::execute_text_request_for_testing(&mut engine, "zero_var = 0").unwrap();
        assert!(result1.error.is_none());

        // Use it in expressions
        let result2 =
            runmat_core::execute_text_request_for_testing(&mut engine, "zero_var + 5").unwrap();
        assert!(result2.error.is_none());
        // Should be 5.0

        let result3 =
            runmat_core::execute_text_request_for_testing(&mut engine, "10 - zero_var").unwrap();
        assert!(result3.error.is_none());
        // Should be 10.0
    });
}

#[test]
fn test_negative_values_persistence() {
    gc_test_context(|| {
        let mut engine = RunMatSession::with_options(true, false).unwrap();

        // Set negative values
        let result1 =
            runmat_core::execute_text_request_for_testing(&mut engine, "neg = -15").unwrap();
        assert!(result1.error.is_none());

        let result2 =
            runmat_core::execute_text_request_for_testing(&mut engine, "pos = 25").unwrap();
        assert!(result2.error.is_none());

        // Use them together
        let result3 =
            runmat_core::execute_text_request_for_testing(&mut engine, "neg + pos").unwrap();
        assert!(result3.error.is_none());
        // Should be 10.0

        let result4 =
            runmat_core::execute_text_request_for_testing(&mut engine, "neg * pos").unwrap();
        assert!(result4.error.is_none());
        // Should be -375.0
    });
}
