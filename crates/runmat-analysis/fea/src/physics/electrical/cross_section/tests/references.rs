//! Published closed forms the cross-section solver is checked against. Each is written from
//! the cited equations.

/// Free-space wave impedance used by Hammerstad and Jensen, Ω.
const ETA0: f64 = 376.73;

/// IPC-2141A (2004), surface microstrip:
/// Z0 = 87 / sqrt(εr + 1.41) · ln(5.98 h / (0.8 w + t)), stated for 0.1 < w/h < 2.0 and
/// 1 < εr < 15.
pub fn ipc_microstrip(w: f64, h: f64, t: f64, er: f64) -> f64 {
    assert!(w / h > 0.1 && w / h < 2.0 && er > 1.0 && er < 15.0, "outside IPC-2141A range");
    87.0 / (er + 1.41).sqrt() * (5.98 * h / (0.8 * w + t)).ln()
}

/// IPC-2141A (2004), symmetric stripline centred between planes `b` apart:
/// Z0 = 60 / sqrt(εr) · ln(1.9 b / (0.8 w + t)), stated for w/b < 0.35 and t/b < 0.25.
pub fn ipc_stripline(w: f64, b: f64, t: f64, er: f64) -> f64 {
    assert!(w / b < 0.35 && t / b < 0.25, "outside IPC-2141A range");
    60.0 / er.sqrt() * (1.9 * b / (0.8 * w + t)).ln()
}

/// Hammerstad and Jensen (1980), zero-thickness microstrip of width ratio u = w/h on a
/// substrate of relative permittivity εr: (Z0, εeff).
///
/// Z01 = η0/(2π) ln(f(u)/u + sqrt(1 + (2/u)²)), f(u) = 6 + (2π − 6) exp(−(30.666/u)^0.7528);
/// εeff = (εr + 1)/2 + (εr − 1)/2 (1 + 10/u)^(−a b), with
/// a = 1 + ln((u⁴ + (u/52)²)/(u⁴ + 0.432))/49 + ln(1 + (u/18.1)³)/18.7 and
/// b = 0.564 ((εr − 0.9)/(εr + 3))^0.053; Z0 = Z01 / sqrt(εeff).
/// The authors state Z01 within 0.01 % for u ≤ 1 and 0.03 % for u ≤ 1000, and εeff within
/// 0.2 % for εr ≤ 128 and 0.01 ≤ u ≤ 100.
pub fn hammerstad_jensen(u: f64, er: f64) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let f = 6.0 + (2.0 * pi - 6.0) * (-(30.666 / u).powf(0.7528)).exp();
    let z01 = ETA0 / (2.0 * pi) * (f / u + (1.0 + 4.0 / (u * u)).sqrt()).ln();
    let a = 1.0
        + ((u.powi(4) + (u / 52.0).powi(2)) / (u.powi(4) + 0.432)).ln() / 49.0
        + (1.0 + (u / 18.1).powi(3)).ln() / 18.7;
    let b = 0.564 * ((er - 0.9) / (er + 3.0)).powf(0.053);
    let eeff = 0.5 * (er + 1.0) + 0.5 * (er - 1.0) * (1.0 + 10.0 / u).powf(-a * b);
    (z01 / eeff.sqrt(), eeff)
}

/// Complete elliptic integral of the first kind K(k), by the arithmetic-geometric mean.
fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut g) = (1.0_f64, (1.0 - k * k).sqrt());
    // The mean converges quadratically; 32 steps reach the double-precision limit for any
    // k < 1, where a fixed tolerance can stall on rounding.
    for _ in 0..32 {
        (a, g) = (0.5 * (a + g), (a * g).sqrt());
    }
    std::f64::consts::PI / (2.0 * a)
}

/// K(k') / K(k), with k' = sqrt(1 − k²).
fn ratio(k: f64) -> f64 {
    elliptic_k((1.0 - k * k).sqrt()) / elliptic_k(k)
}

/// Cohn (1954), zero-thickness strip of width w centred between planes b apart (exact,
/// conformal mapping): Z0 = 30π / sqrt(εr) · K(k)/K(k'), k = sech(πw / 2b).
pub fn cohn_stripline(w: f64, b: f64, er: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let k = 1.0 / (pi * w / (2.0 * b)).cosh();
    30.0 * pi / er.sqrt() / ratio(k)
}

/// Cohn (1955), edge-coupled zero-thickness strips of width w and gap s centred between
/// planes b apart (exact): (Z0e, Z0o) = 30π / sqrt(εr) · K(k')/K(k) with
/// k_e = tanh(πw/2b) tanh(π(w + s)/2b) and k_o = tanh(πw/2b) coth(π(w + s)/2b).
pub fn cohn_coupled_stripline(w: f64, s: f64, b: f64, er: f64) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let a = (pi * w / (2.0 * b)).tanh();
    let c = (pi * (w + s) / (2.0 * b)).tanh();
    let scale = 30.0 * pi / er.sqrt();
    (scale * ratio(a * c), scale * ratio(a / c))
}

#[test]
fn closed_forms_reproduce_their_published_values() {
    // Cohn's stripline: a strip of w/b = 0.5 in air is near 100 Ω... checked through the
    // limit forms: wide strips approach the parallel-plate value 30π/sqrt(εr) · b/w · 1/2
    // per side; narrow strips grow logarithmically.
    let wide = cohn_stripline(4.0, 1.0, 1.0);
    let parallel = 376.73 / 2.0 / (2.0 * 4.0); // two parallel-plate halves of gap b/2
    assert!((wide / parallel - 1.0).abs() < 0.12, "{wide} vs {parallel}");
    // Coupled strips far apart tend to the single strip.
    let single = cohn_stripline(0.3, 1.0, 2.2);
    let (even, odd) = cohn_coupled_stripline(0.3, 5.0, 1.0, 2.2);
    assert!((even / single - 1.0).abs() < 1e-3 && (odd / single - 1.0).abs() < 1e-3);
    // Hammerstad–Jensen at w/h = 1 in air: Z01 ≈ 126 Ω.
    let (z, eeff) = hammerstad_jensen(1.0, 1.0);
    assert!((z - 126.0).abs() < 1.0, "{z}");
    assert!((eeff - 1.0).abs() < 1e-12);
}
