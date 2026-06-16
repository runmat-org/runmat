#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
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
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "tf")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::String(class_name) if class_name == "tf")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![1, 1] && tensor.data == vec![20.0])
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![1, 2] && tensor.data == vec![1.0, 5.0])
    }));
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
        assert(hn(1) == 2);
        assert(hd(1) == 1);
        assert(hd(2) == 0);
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
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "ss")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::String(class_name) if class_name == "ss")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.data == vec![0.0, -2.0, 1.0, -3.0])
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.data == vec![0.0, 1.0])
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
            if tensor.shape == vec![3, 1]
                && (tensor.data[0] - 0.0).abs() < 1.0e-8
                && (tensor.data[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5
                && (tensor.data[2] - (1.0 - (-2.0_f64).exp())).abs() < 1.0e-5)
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape == vec![3, 1] && tensor.data == vec![0.0, 1.0, 2.0])
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
            if tensor.shape == vec![3, 1]
                && (tensor.data[0] - 0.0).abs() < 1.0e-8
                && (tensor.data[2] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5)
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
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 1]) }));
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
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![6, 1]) }));
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
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 1]) }));
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
