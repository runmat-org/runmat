//! MOSFET levels 2 and 3 against the published SPICE2 equations in closed form (Vladimirescu and
//! Liu, UCB/ERL M80/7): each case picks parameters where the drain current has an explicit
//! expression, and the reductions to level 1 hold exactly.

use runmat_circuit_sim::{simulate, Analysis, Design};

const BETA: f64 = 1e-4 * 10.0; // kp · W/L

/// Drain current at `vgs`, `vds` (and `vbs`) for the model card's parameters.
fn id(level: u32, params: &str, vgs: f64, vds: f64, vbs: f64) -> f64 {
    let deck = format!(
        "m\nVD d 0 {vds}\nVG g 0 {vgs}\nVB b 0 {vbs}\nM1 d g 0 b MX L=1u W=10u\n.model MX NMOS(level={level} vto=0.7 kp=1e-4 {params})\n"
    );
    let d = Design::from_netlist(&deck, &|_| None).unwrap_or_else(|e| panic!("{e}"));
    let r = simulate(&d, &Analysis::Op).unwrap_or_else(|e| panic!("{e}")).result;
    -r.op().unwrap()["i(vd)"]
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs() + 1e-15, "{what}: {got} vs {want}");
}

#[test]
fn levels_two_and_three_reduce_to_level_one() {
    for (vgs, vds) in [(1.5, 0.1), (1.5, 0.5), (2.0, 3.0), (3.0, 1.0)] {
        let one = id(1, "", vgs, vds, 0.0);
        close(id(2, "", vgs, vds, 0.0), one, 1e-8, "level 2, γ = 0");
        close(id(3, "kappa=0", vgs, vds, 0.0), one, 1e-8, "level 3, θ = η = vmax = 0");
        let vgst: f64 = vgs - 0.7;
        let want = if vds < vgst { BETA * (vgst - vds / 2.0) * vds } else { BETA / 2.0 * vgst * vgst };
        close(one, want, 1e-8, "level 1 square law");
    }
}

#[test]
fn level_two_charge_sheet_with_body_effect() {
    let (gamma, phi) = (0.5f64, 0.6f64);
    let vbi = 0.7 - gamma * phi.sqrt();
    let card = format!("gamma={gamma} phi={phi}");
    let triode = |vgs: f64, vds: f64, vbs: f64| {
        BETA * ((vgs - vbi - vds / 2.0) * vds - 2.0 / 3.0 * gamma * ((vds + phi - vbs).powf(1.5) - (phi - vbs).powf(1.5)))
    };
    for vbs in [0.0, -1.0] {
        let vgs = 2.0;
        // ∂Ids/∂Vds = 0 at s = √(Vdsat + φ − Vbs): s² + γ·s − (Vgs − Vbi + φ − Vbs) = 0.
        let x = vgs - vbi + phi - vbs;
        let s = ((gamma * gamma + 4.0 * x).sqrt() - gamma) / 2.0;
        let vdsat = s * s - phi + vbs;
        close(id(2, &card, vgs, 0.3, vbs), triode(vgs, 0.3, vbs), 1e-8, "triode");
        close(id(2, &card, vgs, vdsat + 1.0, vbs), triode(vgs, vdsat, vbs), 1e-8, "saturation holds Ids(Vdsat)");
        // The current is flat at Vdsat (no step between the regions).
        let below = id(2, &card, vgs, vdsat - 1e-4, vbs);
        close(below, triode(vgs, vdsat, vbs), 1e-6, "continuous at Vdsat");
        // Body effect raises the threshold to Vbi + γ·√(φ − Vbs).
        if vbs < 0.0 {
            let vth = vbi + gamma * (phi - vbs).sqrt();
            let g0 = id(2, &card, vgs, 1e-4, vbs) / 1e-4;
            close(g0, BETA * (vgs - vth), 2e-3, "channel conductance at Vds → 0");
        }
    }
}

#[test]
fn level_three_mobility_dibl_and_velocity_saturation() {
    let vgs: f64 = 2.0;
    let vgst0 = vgs - 0.7;
    // θ: µs = µo/(1 + θ·Vgst); saturation current β/(1 + θ·Vgst)·Vgst²/2.
    let theta = 0.1;
    close(id(3, &format!("theta={theta} kappa=0"), vgs, 3.0, 0.0), BETA / (1.0 + theta * vgst0) * vgst0 * vgst0 / 2.0, 1e-8, "θ");
    // η (DIBL): Vth = Vto − σ·Vds with σ = η·8.15e−22/(Cox·L³).
    let (eta, tox) = (1.0, 20e-9);
    let cox = 3.9 * 8.854_187_8e-12 / tox;
    let sigma = eta * 8.15e-22 / (cox * 1e-18);
    for vds in [3.0, 4.0] {
        let vgst: f64 = vgst0 + sigma * vds;
        close(id(3, &format!("eta={eta} tox={tox} kappa=0"), vgs, vds, 0.0), BETA / 2.0 * vgst * vgst, 1e-8, "η");
    }
    // vmax: Vc = vmax·L/µo, Vdsat = Vgst + Vc − √(Vgst² + Vc²), Fd = 1/(1 + Vde/Vc).
    let (vmax, uo) = (1e5, 600.0);
    let vc = vmax * 1e-6 / (uo * 1e-4);
    let vdsat = vgst0 + vc - (vgst0 * vgst0 + vc * vc).sqrt();
    let law = |vde: f64| BETA * (vgst0 - vde / 2.0) * vde / (1.0 + vde / vc);
    let card = format!("vmax={vmax} uo={uo} kappa=0");
    close(id(3, &card, vgs, 3.0, 0.0), law(vdsat), 1e-8, "velocity-saturated Idsat");
    close(id(3, &card, vgs, 0.2, 0.0), law(0.2), 1e-8, "velocity-limited triode");
}

#[test]
fn substrate_doping_gives_gamma_and_phi() {
    // γ = √(2·q·εsi·Nsub)/Cox: at Vbs = −2 V the threshold rises by γ·(√(φ + 2) − √φ).
    let (nsub, tox, phi) = (1e16f64, 20e-9, 0.7);
    let cox = 3.9 * 8.854_187_8e-12 / tox;
    let gamma = (2.0 * 1.602_176_634e-19 * 11.7 * 8.854_187_8e-12 * nsub * 1e6).sqrt() / cox;
    let card = format!("nsub={nsub} tox={tox} phi={phi}");
    let vth = 0.7 + gamma * ((phi + 2.0f64).sqrt() - phi.sqrt());
    let vgst = 3.0 - vth;
    close(id(1, &card, 3.0, 5.0, -2.0), BETA / 2.0 * vgst * vgst, 1e-8, "level 1 with Nsub");
}
