#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

#[test]
fn tf_constructs_object_through_vm_dispatch() {
    let program = r#"
        H = tf(20, [1 5]);
        c = class(H);
        n = H.Numerator;
        d = H.Denominator;
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name().is(runmat_types::StaticClassIdentity::new("tf")))));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::String(class_name) if class_name == "tf")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![1, 1] && tensor.materialize_f64() == vec![20.0])
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![1, 2] && tensor.materialize_f64() == vec![1.0, 5.0])
    }));
}

#[test]
fn compiled_model_property_indexing_retains_colon_selectors() {
    execute_source(
        r#"
        G = ss([0 1; -2 -3], [0;1], [1 0], 0);
        assert(isequal(G.A(:), [0;-2;1;-3]));
        assert(isequal(G.A(:,2), [1;-3]));
        holder = struct('model', G);
        assert(isequal(holder.model.A(:), [0;-2;1;-3]));
        "#,
    )
    .expect("property indexing resolves selectors after distinguishing stored values from calls");
}

#[test]
fn compiled_ss_integer_precision_boundary_reports_owning_identifiers() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (source, identifier) in [
        (
            "wide=uint64(9007199254740992)+uint64(1); S=ss(wide,1,1,1);",
            "RunMat:ss:InvalidMatrix",
        ),
        (
            "wide=uint64(9007199254740992)+uint64(1); S=ss(1,1,1,1,wide);",
            "RunMat:ss:InvalidSampleTime",
        ),
    ] {
        let error = execute_source(source).expect_err("inexact integer ss input");
        assert_eq!(error.identifier(), Some(identifier), "{source}: {error}");
    }
    execute_source(
        "wide=uint64(9007199254740992); S=ss(wide,1,1,1,wide); assert(S.A==double(wide)); assert(S.Ts==double(wide));",
    )
    .expect("exact wide integer ss inputs");
}

#[test]
fn tf_variable_control_workflow_runs_through_vm_dispatch() {
    let program = r#"
        s = tf('s');
        G = 2.5 / (0.4*s^2 + 1.8*s + 1);
        T = feedback(3.2 * G, 1);
        [y, t] = step(T, 0:0.5:2);
        info = stepinfo(y, t, dcgain(T));
        p = pole(T);
        zsys = tf([1 3 2], [1 4]);
        z = zero(zsys);
        sys_ss = ss([0 1; -4 -0.5], [0; 1], [1 0], 0);
        pss = pole(sys_ss);
        [pz_p, pz_z] = pzmap(zsys);
        [wn, zeta, dp] = damp(T);
        stable = isstable(T);
        ss = dcgain(T);
        H = 2 / s;
        hn = H.Numerator;
        hd = H.Denominator;
        assert(stable);
        assert(abs(ss - 8/9) < 1e-6);
        assert(length(y) == 5);
        assert(info.RiseTime >= 0 || isnan(info.RiseTime));
        assert(info.Overshoot >= 0);
        assert(all(real(p) < 0));
        assert(length(z) == 2);
        assert(any(abs(z + 1) < 1e-6));
        assert(any(abs(z + 2) < 1e-6));
        assert(length(pz_p) == 1);
        assert(abs(pz_p + 4) < 1e-6);
        assert(length(pz_z) == 2);
        assert(any(abs(pz_z + 1) < 1e-6));
        assert(any(abs(pz_z + 2) < 1e-6));
        assert(length(pss) == 2);
        assert(length(wn) == length(dp));
        assert(length(zeta) == length(dp));
        assert(all(wn >= 0));
        assert(all(zeta > 0));
        assert(hn(1) == 2);
        assert(hd(1) == 1);
        assert(hd(2) == 0);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn stepinfo_tf_model_form_executes_through_compiled_vm_call() {
    let program = r#"
        H = tf(1, [1 1]);
        info = stepinfo(H);
        [y, t] = step(H);
        sampled = stepinfo(y, t, dcgain(H));
        assert(abs(info.SteadyStateValue - 1) < 1e-12);
        assert(abs(info.SteadyStateValue - sampled.SteadyStateValue) < 1e-12);
        assert(abs(info.RiseTime - sampled.RiseTime) < 1e-10);
        assert(info.Overshoot == 0);
    "#;
    execute_source(program).expect("compiled tf model-form stepinfo call");
}

#[test]
fn stepinfo_ss_model_form_reports_owning_unsupported_model_error_in_vm() {
    let error = execute_source("S = ss(-1, 1, 1, 0); info = stepinfo(S);")
        .expect_err("stepinfo currently supports tf models, not ss models");
    assert_eq!(error.identifier(), Some("RunMat:stepinfo:UnsupportedModel"));
}

#[test]
fn tf_wide_integer_gain_rounds_only_at_model_arithmetic_boundary_in_vm() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let program = r#"
        wide = uint64(9007199254740992) + uint64(1);
        assert(wide ~= uint64(9007199254740992));
        H = tf(0, 1);
        left = wide + H;
        right = H + wide;
        ln = left.Numerator;
        rn = right.Numerator;
        assert(isa(ln, 'double') && isa(rn, 'double'));
        assert(ln(1) == 9007199254740992);
        assert(rn(1) == 9007199254740992);
    "#;
    execute_source(program).expect("compiled scalar model arithmetic admits rounded wide gain");
}

#[test]
fn tf_constructor_rejects_inexact_integer_coefficients_through_vm() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "wide = uint64(9007199254740992) + uint64(1); H = tf(wide, 1);",
        "wide = uint64(9007199254740992) + uint64(1); H = tf(1, [wide uint64(1)]);",
    ] {
        let error = execute_source(source).expect_err("inexact coefficient must be rejected");
        assert!(
            error.message().contains("exactly representable"),
            "{source}: {error}"
        );
    }
}

#[test]
fn tf_minimum_signed_integer_power_preserves_scalar_identity_in_vm() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let program = r#"
        exponent = intmin('int64');
        positive = tf(1, 1);
        negative = tf(-1, 1);
        p_pos = positive .^ exponent;
        m_pos = positive ^ exponent;
        p_neg = negative .^ exponent;
        m_neg = negative ^ exponent;
        assert(isa(p_pos, 'tf') && isa(m_pos, 'tf'));
        assert(isa(p_neg, 'tf') && isa(m_neg, 'tf'));
        pn = p_pos.Numerator; pd = p_pos.Denominator;
        mn = m_pos.Numerator; md = m_pos.Denominator;
        nn = p_neg.Numerator; nd = p_neg.Denominator;
        qn = m_neg.Numerator; qd = m_neg.Denominator;
        assert(pn(1) == 1 && pd(1) == 1);
        assert(mn(1) == 1 && md(1) == 1);
        assert(nn(1) == 1 && nd(1) == 1);
        assert(qn(1) == 1 && qd(1) == 1);
    "#;
    execute_source(program).expect("compiled power and mpower accept exact i64 minimum");
}

#[test]
fn lqr_matrix_and_state_space_forms_run_through_vm_dispatch() {
    let program = r#"
        A = [0 1; 0 0];
        B = [0; 1];
        Q = eye(2);
        R = 1;
        [K, S, e] = lqr(A, B, Q, R);
        sys = ss(A, B, [1 0], 0);
        Ksys = lqr(sys, Q, R);
        assert(abs(K(1) - 1) < 1e-8);
        assert(abs(K(2) - sqrt(3)) < 1e-8);
        assert(abs(S(1,1) - sqrt(3)) < 1e-8);
        assert(abs(S(1,2) - 1) < 1e-8);
        assert(length(e) == 2);
        assert(all(real(e) < 0));
        assert(all(abs(Ksys - K) < 1e-8));
        sysd = ss([1 0.1; 0 1], [0.005; 0.1], [1 0], 0, 0.1);
        [Kd, Sd, ed] = lqr(sysd, Q, R);
        assert(length(ed) == 2);
        assert(all(abs(ed) < 1));
        assert(all(isfinite(Kd)));
        assert(all(isfinite(Sd)));
    "#;
    execute_source(program).unwrap();
}

#[test]
fn tf_rhs_elementwise_division_preserves_operand_order() {
    let program = r#"
        s = tf('s');

        A = 2 ./ s;
        B = s ./ 2;
        C = 2 .\ s;
        D = s .\ 2;

        an = A.Numerator;
        ad = A.Denominator;
        bn = B.Numerator;
        bd = B.Denominator;
        cn = C.Numerator;
        cd = C.Denominator;
        dn = D.Numerator;
        dd = D.Denominator;

        assert(length(an) == 1);
        assert(an(1) == 2);
        assert(length(ad) == 2);
        assert(ad(1) == 1);
        assert(ad(2) == 0);

        assert(length(bn) == 2);
        assert(bn(1) == 1);
        assert(bn(2) == 0);
        assert(length(bd) == 1);
        assert(bd(1) == 2);

        assert(length(cn) == 2);
        assert(cn(1) == 1);
        assert(cn(2) == 0);
        assert(length(cd) == 1);
        assert(cd(1) == 2);

        assert(length(dn) == 1);
        assert(dn(1) == 2);
        assert(length(dd) == 2);
        assert(dd(1) == 1);
        assert(dd(2) == 0);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn tf_rhs_matrix_operators_preserve_operand_order() {
    let program = r#"
        s = tf('s');

        A = 5 - s;
        B = 2 / s;
        C = 2 \ s;

        an = A.Numerator;
        ad = A.Denominator;
        bn = B.Numerator;
        bd = B.Denominator;
        cn = C.Numerator;
        cd = C.Denominator;

        assert(length(an) == 2);
        assert(an(1) == -1);
        assert(an(2) == 5);
        assert(length(ad) == 1);
        assert(ad(1) == 1);

        assert(length(bn) == 1);
        assert(bn(1) == 2);
        assert(length(bd) == 2);
        assert(bd(1) == 1);
        assert(bd(2) == 0);

        assert(length(cn) == 2);
        assert(cn(1) == 1);
        assert(cn(2) == 0);
        assert(length(cd) == 1);
        assert(cd(1) == 2);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn step_statement_form_accepts_multiple_system_plot_syntax() {
    let program = r#"
        s = tf('s');
        H1 = 1 / (s + 1);
        H2 = 2 / (s + 2);
        step(H1, 'b');
        step(H1, H2);
        step(H1, 'b', H2, 'r--', 2);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn ss_constructs_object_through_vm_dispatch() {
    let program = r#"
        G = ss([0 1; -2 -3], [0; 1], [1 0], 0, 0.1);
        c = class(G);
        a = G.A;
        b = G.B;
        ts = G.Ts;
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name().is(runmat_types::StaticClassIdentity::new("ss")))));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::String(class_name) if class_name == "ss")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![2, 2] && tensor.materialize_f64() == vec![0.0, -2.0, 1.0, -3.0])
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![2, 1] && tensor.materialize_f64() == vec![0.0, 1.0])
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 0.1).abs() < 1.0e-12)));
}

#[test]
fn step_returns_siso_response_through_vm_dispatch() {
    let program = r#"
        H = tf(1, [1 1]);
        [y, t] = step(H, [0 1 2]);
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape() == vec![3, 1]
                && (tensor.materialize_f64()[0] - 0.0).abs() < 1.0e-8
                && (tensor.materialize_f64()[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5
                && (tensor.materialize_f64()[2] - (1.0 - (-2.0_f64).exp())).abs() < 1.0e-5)
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape() == vec![3, 1] && tensor.materialize_f64() == vec![0.0, 1.0, 2.0])
    }));
}

#[test]
fn step_single_output_assignment_returns_response() {
    let program = r#"
        H = tf(1, [1 1]);
        y = step(H, 0:0.5:1);
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape() == vec![3, 1]
                && (tensor.materialize_f64()[0] - 0.0).abs() < 1.0e-8
                && (tensor.materialize_f64()[2] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5)
    }));
}

#[test]
fn step_statement_form_plots_without_error() {
    let program = r#"
        H = tf(1, [1 1]);
        step(H);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn pzmap_statement_form_plots_without_error() {
    let program = r#"
        H = tf([1 2], [1 3 4]);
        pzmap(H);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn impulse_returns_siso_response_through_vm_dispatch() {
    let program = r#"
        H = tf(20, [1 5]);
        [y, t] = impulse(H, [0 0.1 0.2]);
        y1 = y(1);
        y2 = y(2);
        t3 = t(3);
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![3, 1]) }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 20.0).abs() < 1.0e-8,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - (20.0 * (-0.5f64).exp())).abs() < 1.0e-8,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 0.2).abs() < 1.0e-12,
        _ => false,
    }));
}

#[test]
fn impulse_discrete_response_through_vm_dispatch() {
    let discrete = r#"
        H = tf(1, [1 -0.5], 0.1);
        [y, t] = impulse(H, 0:0.1:0.5);
        y1 = y(1);
        y2 = y(2);
        y6 = y(6);
        t6 = t(6);
    "#;
    let vars = execute_source(discrete).unwrap();
    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![6, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if n.abs() < 1.0e-12)));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 10.0).abs() < 1.0e-12,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 0.625).abs() < 1.0e-12,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 0.5).abs() < 1.0e-12,
        _ => false,
    }));
}

#[test]
fn impulse_statement_form_plots_without_error() {
    let program = r#"
        H = tf(20, [1 5]);
        impulse(H);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn nyquist_returns_frequency_response_through_vm_dispatch() {
    let program = r#"
        H = tf(1, [1 1]);
        [re, im, w] = nyquist(H, [0 1 2]);
        re1 = re(1);
        re2 = re(2);
        im2 = im(2);
        w3 = w(3);
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape() == vec![3, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 1.0).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 0.5).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n + 0.5).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 2.0).abs() < 1.0e-12)));
}

#[test]
fn nyquist_statement_form_plots_without_error() {
    let program = r#"
        H = tf(1, [1 2 1]);
        nyquist(H);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn rlocus_returns_roots_through_vm_dispatch() {
    let program = r#"
        H = tf(1, [1 1]);
        [r, k] = rlocus(H, [0 1 3]);
        assert(r(1, 1) == -1);
        assert(r(1, 2) == -2);
        assert(r(1, 3) == -4);
        assert(k(1) == 0);
        assert(k(3) == 3);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn rlocus_statement_form_accepts_control_workflow() {
    let program = r#"
        s = tf('s');
        H = (s + 2) / (s^2 + 3*s + 4);
        rlocus(H);
        step(feedback(H, 1));
    "#;
    execute_source(program).unwrap();
}

#[test]
fn rlocus_accepts_siso_state_space_models() {
    let program = r#"
        H = ss(-1, 1, 1, 0);
        [r, k] = rlocus(H, [0 1 3]);
        assert(abs(r(1, 1) + 1) < 1e-8);
        assert(abs(r(1, 2) + 2) < 1e-8);
        assert(abs(r(1, 3) + 4) < 1e-8);
        assert(k(3) == 3);
    "#;
    execute_source(program).unwrap();
}
