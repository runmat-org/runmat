//! Evaluation of compiled behavioural expressions: the value and its partials with respect to
//! each control, by forward differentiation. Comparisons, steps and rounding have zero partials
//! (they are flat almost everywhere); `?:`, `if`, `min`, `max` and `limit` take the partials of
//! the branch they choose; a table takes its segment's slope.

use super::behave::{Bop, Ex, Fun};

/// A value with partials.
#[derive(Clone, Debug, PartialEq)]
pub struct G {
    pub v: f64,
    pub d: Vec<f64>,
}

impl G {
    fn c(v: f64, n: usize) -> G {
        G { v, d: vec![0.0; n] }
    }
    /// `f(self)` with derivative `df` at `self.v`.
    fn chain(mut self, v: f64, df: f64) -> G {
        self.v = v;
        for x in &mut self.d {
            *x *= df;
        }
        self
    }
    /// `a·self + b·other` partials, with value `v`.
    fn mix(mut self, other: &G, a: f64, b: f64, v: f64) -> G {
        self.v = v;
        for (x, y) in self.d.iter_mut().zip(&other.d) {
            *x = a * *x + b * y;
        }
        self
    }
}

impl Ex {
    /// The value and partials at controls `x` and time `time`; adds the real operations to `ops`.
    pub fn eval(&self, x: &[f64], time: f64, ops: &mut u64) -> G {
        let n = x.len();
        *ops += 1 + n as u64;
        match self {
            Ex::Num(v) => G::c(*v, n),
            Ex::Time => G::c(time, n),
            Ex::Var(k) => {
                let mut g = G::c(x[*k], n);
                g.d[*k] = 1.0;
                g
            }
            Ex::Neg(a) => {
                let a = a.eval(x, time, ops);
                let v = -a.v;
                a.chain(v, -1.0)
            }
            Ex::Bin(op, a, b) => binary(*op, a.eval(x, time, ops), b.eval(x, time, ops)),
            Ex::Cond(c, a, b) => {
                if c.eval(x, time, ops).v != 0.0 {
                    a.eval(x, time, ops)
                } else {
                    b.eval(x, time, ops)
                }
            }
            Ex::Table(e, pts) => {
                let a = e.eval(x, time, ops);
                let (v, slope) = lookup(pts, a.v);
                a.chain(v, slope)
            }
            Ex::Call(Fun::If, args) => {
                if args[0].eval(x, time, ops).v != 0.0 {
                    args[1].eval(x, time, ops)
                } else {
                    args[2].eval(x, time, ops)
                }
            }
            Ex::Call(f, args) => {
                let mut g: Vec<G> = args.iter().map(|a| a.eval(x, time, ops)).collect();
                call(*f, &mut g)
            }
        }
    }
}

fn lookup(pts: &[(f64, f64)], x: f64) -> (f64, f64) {
    let (first, last) = (pts[0], pts[pts.len() - 1]);
    if x <= first.0 {
        return (first.1, 0.0);
    }
    if x >= last.0 {
        return (last.1, 0.0);
    }
    let k = pts.partition_point(|p| p.0 <= x).max(1);
    let ((x0, y0), (x1, y1)) = (pts[k - 1], pts[k]);
    let s = (y1 - y0) / (x1 - x0);
    (y0 + s * (x - x0), s)
}

fn flag(b: bool, n: usize) -> G {
    G::c(b as i32 as f64, n)
}

fn binary(op: Bop, a: G, b: G) -> G {
    let n = a.d.len();
    match op {
        Bop::Add => {
            let v = a.v + b.v;
            a.mix(&b, 1.0, 1.0, v)
        }
        Bop::Sub => {
            let v = a.v - b.v;
            a.mix(&b, 1.0, -1.0, v)
        }
        Bop::Mul => {
            let v = a.v * b.v;
            let (av, bv) = (a.v, b.v);
            a.mix(&b, bv, av, v)
        }
        Bop::Div => {
            let v = a.v / b.v;
            let bv = b.v;
            a.mix(&b, 1.0 / bv, -v / bv, v)
        }
        Bop::Pow => pow(a, b),
        Bop::Lt => flag(a.v < b.v, n),
        Bop::Le => flag(a.v <= b.v, n),
        Bop::Gt => flag(a.v > b.v, n),
        Bop::Ge => flag(a.v >= b.v, n),
        Bop::Eq => flag(a.v == b.v, n),
        Bop::Ne => flag(a.v != b.v, n),
        Bop::And => flag(a.v != 0.0 && b.v != 0.0, n),
        Bop::Or => flag(a.v != 0.0 || b.v != 0.0, n),
    }
}

/// `a^b`: `b·a^(b−1)` on `a`, and `a^b·ln a` on `b` where `a > 0`.
fn pow(a: G, b: G) -> G {
    let v = a.v.powf(b.v);
    let da = if b.v == 0.0 { 0.0 } else { b.v * a.v.powf(b.v - 1.0) };
    let db = if a.v > 0.0 { v * a.v.ln() } else { 0.0 };
    let db = if b.d.iter().all(|x| *x == 0.0) { 0.0 } else { db };
    a.mix(&b, da, db, v)
}

fn call(f: Fun, g: &mut [G]) -> G {
    use Fun::*;
    let n = g[0].d.len();
    let a = g[0].clone();
    let x = a.v;
    let one = |v: f64, d: f64| a.clone().chain(v, d);
    match f {
        Sqrt => {
            let v = x.sqrt();
            one(v, 0.5 / v)
        }
        Exp => {
            let v = x.exp();
            one(v, v)
        }
        Ln => one(x.ln(), 1.0 / x),
        Log10 => one(x.log10(), 1.0 / (x * std::f64::consts::LN_10)),
        Abs => one(x.abs(), if x < 0.0 { -1.0 } else { 1.0 }),
        Sin => one(x.sin(), x.cos()),
        Cos => one(x.cos(), -x.sin()),
        Tan => {
            let c = x.cos();
            one(x.tan(), 1.0 / (c * c))
        }
        Asin => one(x.asin(), 1.0 / (1.0 - x * x).sqrt()),
        Acos => one(x.acos(), -1.0 / (1.0 - x * x).sqrt()),
        Atan => one(x.atan(), 1.0 / (1.0 + x * x)),
        Sinh => one(x.sinh(), x.cosh()),
        Cosh => one(x.cosh(), x.sinh()),
        Tanh => {
            let t = x.tanh();
            one(t, 1.0 - t * t)
        }
        Floor => G::c(x.floor(), n),
        Ceil => G::c(x.ceil(), n),
        Int => G::c(x.trunc(), n),
        Sgn => G::c(if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }, n),
        U => G::c(if x > 0.0 { 1.0 } else { 0.0 }, n),
        Uramp => one(x.max(0.0), if x > 0.0 { 1.0 } else { 0.0 }),
        Pow => pow(a, g[1].clone()),
        Pwr => {
            // sign(a)·|a|^b
            let s = if x < 0.0 { -1.0 } else { 1.0 };
            let m = pow(a.clone().chain(x.abs(), s), g[1].clone());
            let v = s * m.v;
            m.chain(v, s)
        }
        Atan2 => {
            let (y, xx) = (x, g[1].v);
            let r2 = y * y + xx * xx;
            a.mix(&g[1], xx / r2, -y / r2, y.atan2(xx))
        }
        Min => {
            if x <= g[1].v {
                a
            } else {
                g[1].clone()
            }
        }
        Max => {
            if x >= g[1].v {
                a
            } else {
                g[1].clone()
            }
        }
        Limit => {
            let (lo, hi) = (g[1].v.min(g[2].v), g[1].v.max(g[2].v));
            if x < lo {
                if g[1].v <= g[2].v { g[1].clone() } else { g[2].clone() }
            } else if x > hi {
                if g[1].v >= g[2].v { g[1].clone() } else { g[2].clone() }
            } else {
                a
            }
        }
        If => unreachable!("`if` is evaluated lazily"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::behave::{compile_card, Ctl, Drives, Env};
    use super::super::expr::Scope;

    fn env_eval(raw: &str, x: &[f64]) -> (f64, Vec<f64>, Vec<Ctl>) {
        let scope: Scope = [("gain".to_string(), 3.0)].into();
        let node = |n: &str| format!("x1.{n}");
        let src = |s: &str| format!("x1.{s}");
        let b = compile_card('b', raw, &Env { scope: &scope, funcs: &[], node: &node, source: &src }).unwrap();
        assert_eq!(b.drives, Drives::Voltage);
        let mut ops = 0;
        let g = b.expr.eval(x, 0.0, &mut ops);
        (g.v, g.d, b.ctl)
    }

    #[test]
    fn partials_are_exact_and_controls_map_through_the_instance() {
        let (v, d, ctl) = env_eval("out 0 V={gain*V(a,b)*V(a,b) + exp(I(vs)) - V(a,b)/V(c)}", &[2.0, 0.5, 4.0]);
        assert_eq!(ctl, [Ctl::V("x1.a".into(), "x1.b".into()), Ctl::I("x1.vs".into()), Ctl::V("x1.c".into(), "0".into())]);
        assert!((v - (12.0 + 0.5f64.exp() - 0.5)).abs() < 1e-12);
        assert!((d[0] - (12.0 - 0.25)).abs() < 1e-12);
        assert!((d[1] - 0.5f64.exp()).abs() < 1e-12);
        assert!((d[2] - 2.0 / 16.0).abs() < 1e-12);
    }

    #[test]
    fn tables_interpolate_and_hold_past_the_ends() {
        let scope = Scope::new();
        let id = |n: &str| n.to_string();
        let b = compile_card('e', "out 0 TABLE {V(in)} = (-1, -2) (1, 2)", &Env { scope: &scope, funcs: &[], node: &id, source: &id }).unwrap();
        let mut ops = 0;
        for (x, v, s) in [(-3.0, -2.0, 0.0), (0.5, 1.0, 2.0), (5.0, 2.0, 0.0)] {
            let g = b.expr.eval(&[x], 0.0, &mut ops);
            assert_eq!((g.v, g.d[0]), (v, s));
        }
    }
}
