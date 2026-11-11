use runmat_circuit_sim::devices::{Method, States};
use runmat_circuit_sim::num::Dual;
use runmat_circuit_sim::{simulate, Analysis, Design, TranSpec};

#[test]
fn steady_capacitor_reports_zero_current_through_the_transient_load_path() {
    for method in ["euler", "trap", "gear"] {
        let design = Design::from_netlist(
            &format!("steady capacitor\nV1 node 0 1\nC1 node 0 1e-8\n.options method={method}\n"),
            &|_| None,
        )
        .unwrap();
        let out = simulate(
            &design,
            &Analysis::Tran(TranSpec {
                tstep: 2e-9,
                tstop: 1.4e-7,
                tstart: 0.0,
                tmax: None,
                uic: false,
            }),
        )
        .unwrap();
        let wave = out.result.real().unwrap();
        assert!(wave.x.len() > 50);
        for current in wave.get("i(c1)").unwrap() {
            assert_eq!(
                *current, 0.0,
                "{method} introduced current in a steady capacitor"
            );
        }
    }
}

#[test]
fn unchanged_charge_and_flux_have_zero_derivative_at_small_timesteps() {
    for method in [Method::Euler, Method::Trapezoidal, Method::Gear2] {
        for flux in [false, true] {
            for h in [2e-9, 2e-11, 1.4e-17, 1e-16] {
                let mut states = States::new(vec![flux]);
                states.method = method;
                states.h = [h, h * 2.0];
                states.q[0] = [1e-8; 4];
                let derivative = states.derivative(0, Dual::<1>::var(1e-8, 0));
                assert_eq!(derivative.v, 0.0, "{method:?}, flux={flux}, h={h}");
                let (a0, _) = states.coefficients(0);
                assert!((derivative.d[0] / a0 - 1.0).abs() < 4.0 * f64::EPSILON);
            }
        }
    }
}

#[test]
fn linear_charge_history_preserves_derivative_and_newton_jacobian() {
    for method in [Method::Euler, Method::Trapezoidal, Method::Gear2] {
        for h in [0.25, 0.5, 1.0, 2.0] {
            let mut states = States::new(vec![false]);
            states.method = method;
            states.h = [h, 1.0];
            states.q[0] = [0.0, 3.0, 2.0, 1.0];
            states.i[0] = [0.0, 1.0];
            let derivative = states.derivative(0, Dual::<1>::var(3.0 + h, 0));
            assert!(
                (derivative.v - 1.0).abs() < 4.0 * f64::EPSILON,
                "{method:?}, h={h}"
            );
            let (a0, _) = states.coefficients(0);
            assert!((derivative.d[0] / a0 - 1.0).abs() < 4.0 * f64::EPSILON);
        }
    }
}
