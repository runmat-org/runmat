#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_core::RunMatSession;
use runmat_gc::gc_test_context;
use std::sync::Once;

static INIT_ACCEL_REPROS: Once = Once::new();

fn ensure_fusion_regression_env() {
    INIT_ACCEL_REPROS.call_once(|| {
        std::env::set_var("RUNMAT_ACCEL_AUTO_OFFLOAD", "1");
        std::env::set_var("RUNMAT_ACCEL_CALIBRATE", "0");
        std::env::set_var("RUNMAT_ACCEL_THRESHOLD_ELEMWISE", "1");
    });
}

fn read_scalar(engine: &mut RunMatSession, expr: &str) -> f64 {
    let result = block_on(engine.execute(expr)).expect("evaluate scalar expression");
    match result.value.expect("scalar value should be available") {
        Value::Num(value) => value,
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => panic!("expected scalar numeric value, got {other:?}"),
    }
}

#[test]
fn atan2_after_assignment_chain_executes_without_stack_underflow_end_to_end() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    let script = r#"
        Rg = 0.01;
        wLg = 0.08;
        i_d0 = 0.8;
        V_pcc = sqrt(0.95);
        Vd_drop = Rg * i_d0;
        Vq_drop = wLg * i_d0;
        delta_g0 = atan2(Vq_drop, V_pcc + Vd_drop);
    "#;

    let result = block_on(engine.execute(script)).expect("execute atan2 chain script");
    assert!(
        result.error.is_none(),
        "unexpected execution error: {:?}",
        result.error
    );

    let parsed = read_scalar(&mut engine, "delta_g0");
    let expected = 0.064f64.atan2(0.01 * 0.8 + 0.95f64.sqrt());
    assert!(
        (parsed - expected).abs() < 1e-12,
        "expected {expected}, got {parsed}"
    );
}

#[test]
fn atan2_vector_assignment_boundary_executes_correctly_end_to_end() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    let script = r#"
        x = [0.1, 0.2, 0.3];
        a = x + 1;
        b = x + 2;
        y = atan2(a, b);
    "#;

    let result = block_on(engine.execute(script)).expect("execute atan2 vector script");
    assert!(
        result.error.is_none(),
        "unexpected execution error: {:?}",
        result.error
    );

    let expected = [
        (1.1f64).atan2(2.1),
        (1.2f64).atan2(2.2),
        (1.3f64).atan2(2.3),
    ];
    for (idx, value) in expected.iter().enumerate() {
        let actual = read_scalar(&mut engine, &format!("y({})", idx + 1));
        assert!(
            (actual - value).abs() < 1e-12,
            "expected {value}, got {actual}"
        );
    }
}

#[test]
fn mod_and_rem_real_session_parity_end_to_end() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    let script = r#"
        x = [-5.5, -1.25, 5.5, 6, 0/0, 5, 1/0, 4];
        d = [2,    2,     2,   0, 2,   1/0, 2,   0];
        m = mod(x, d);
        r = rem(x, d);
    "#;

    let result = block_on(engine.execute(script)).expect("execute mod/rem session script");
    assert!(
        result.error.is_none(),
        "unexpected execution error: {:?}",
        result.error
    );

    let expected_mod = [0.5, 0.75, 1.5, f64::NAN, f64::NAN, 5.0, f64::NAN, f64::NAN];
    let expected_rem = [
        -1.5,
        -1.25,
        1.5,
        f64::NAN,
        f64::NAN,
        5.0,
        f64::NAN,
        f64::NAN,
    ];

    for (idx, expected) in expected_mod.iter().enumerate() {
        let actual = read_scalar(&mut engine, &format!("m({})", idx + 1));
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "expected NaN for m({}), got {actual}",
                idx + 1
            );
        } else {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
    }

    for (idx, expected) in expected_rem.iter().enumerate() {
        let actual = read_scalar(&mut engine, &format!("r({})", idx + 1));
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "expected NaN for r({}), got {actual}",
                idx + 1
            );
        } else {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn compile_fusion_plan_exposes_semantic_planner_metadata() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    let script = r#"
        A = rand(8, 8);
        B = A + 1;
        C = B .* 2;
    "#;

    let snapshot = engine
        .compile_fusion_plan(script)
        .expect("compile fusion plan should succeed")
        .expect("expected at least one fusion group for regression script");

    assert_eq!(
        snapshot.planner.source,
        "semantic-mir-analysis",
        "unexpected planner source tag"
    );
    assert_eq!(
        snapshot.planner.accel_graph_state, "present",
        "unexpected accel graph state for compile fusion snapshot"
    );
    assert!(
        snapshot.planner.mir_local_fact_count > 0,
        "expected non-zero MIR local fact count"
    );
    assert!(
        snapshot.planner.mir_fusion_signal_count > 0,
        "expected non-zero MIR fusion signal count"
    );
    assert!(
        snapshot.planner.mir_fusion_candidate_group_count > 0,
        "expected non-zero MIR fusion candidate group count"
    );
    assert!(
        snapshot.planner.mir_semantic_instruction_window_count > 0,
        "expected non-zero MIR semantic instruction window count"
    );
    assert!(
        snapshot
            .nodes
            .iter()
            .any(|node| node.kind == "SemanticWindow"),
        "expected semantic window artifacts in compile fusion snapshot"
    );
}

#[test]
fn runtime_fusion_snapshot_exposes_semantic_planner_metadata() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    engine.set_emit_fusion_plan(true);
    let script = r#"
        A = rand(8, 8);
        B = A + 1;
        C = B .* 2;
    "#;

    let outcome = block_on(engine.execute_outcome(script)).expect("execute script");
    let snapshot = outcome
        .fusion_plan
        .expect("expected runtime fusion plan snapshot");

    assert_eq!(
        snapshot.planner.source,
        "semantic-mir-analysis-runtime",
        "unexpected runtime planner source tag"
    );
    assert_eq!(
        snapshot.planner.accel_graph_state, "present",
        "unexpected accel graph state for runtime fusion snapshot"
    );
    assert!(
        snapshot.planner.mir_local_fact_count > 0,
        "expected non-zero runtime MIR local fact count"
    );
    assert!(
        snapshot.planner.mir_fusion_signal_count > 0,
        "expected non-zero runtime MIR fusion signal count"
    );
    assert!(
        snapshot.planner.mir_fusion_candidate_group_count > 0,
        "expected non-zero runtime MIR fusion candidate group count"
    );
    assert!(
        snapshot.planner.mir_semantic_instruction_window_count > 0,
        "expected non-zero runtime MIR semantic instruction window count"
    );
    assert!(
        snapshot
            .nodes
            .iter()
            .any(|node| node.kind == "SemanticWindow"),
        "expected semantic window artifacts in runtime fusion snapshot"
    );
}

#[test]
fn compile_fusion_plan_exposes_semantic_candidates_without_bytecode_groups() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    let script = r#"
        x = 1 + 2;
        y = x * 3;
    "#;

    let snapshot = engine
        .compile_fusion_plan(script)
        .expect("compile fusion plan should succeed")
        .expect("semantic fusion candidate metadata should produce a fusion snapshot");

    assert!(
        snapshot.planner.mir_fusion_signal_count > 0,
        "expected non-zero MIR fusion signal count"
    );
    assert!(
        snapshot.planner.mir_fusion_candidate_group_count > 0,
        "expected non-zero MIR fusion candidate group count"
    );
    assert!(
        snapshot.planner.mir_semantic_instruction_window_count > 0,
        "expected non-zero MIR semantic instruction window count"
    );
    if snapshot.nodes.is_empty() {
        assert!(
            snapshot
                .decisions
                .iter()
                .any(|decision| decision.node_id == "semantic-candidate-summary"),
            "expected semantic candidate summary decision when bytecode groups are absent"
        );
    }
}

#[test]
fn compile_fusion_plan_scopes_local_fact_count_to_entrypoint() {
    ensure_fusion_regression_env();

    let mut engine = gc_test_context(RunMatSession::new).expect("session init");
    let base_script = r#"
        A = rand(8, 8);
        B = A + 1;
        C = B .* 2;
    "#;
    let helper_heavy_script = r#"
        A = rand(8, 8);
        B = A + 1;
        C = B .* 2;
        function out = helper(v)
            t1 = v + 1;
            t2 = t1 .* 2;
            t3 = t2 - 3;
            out = t3 ./ 4;
        end
    "#;

    let base_snapshot = engine
        .compile_fusion_plan(base_script)
        .expect("compile base fusion plan should succeed")
        .expect("expected base fusion snapshot");
    let helper_snapshot = engine
        .compile_fusion_plan(helper_heavy_script)
        .expect("compile helper fusion plan should succeed")
        .expect("expected helper fusion snapshot");

    assert_eq!(
        base_snapshot.planner.mir_local_fact_count, helper_snapshot.planner.mir_local_fact_count,
        "entrypoint MIR local fact count should not be inflated by non-entrypoint helper locals"
    );
}
