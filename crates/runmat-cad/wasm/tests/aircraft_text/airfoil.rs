//! NACA four-digit sections (camber `m` at `p` of chord, thickness `t`), sampled for sketch
//! splines. Unit-chord coordinates: `x` aft from the leading edge, `z` up.

#[derive(Clone, Copy, Debug)]
pub struct Naca {
    pub m: f64,
    pub p: f64,
    pub t: f64,
}

impl Naca {
    pub fn symmetric(t: f64) -> Self {
        Naca { m: 0.0, p: 0.4, t }
    }

    fn thickness(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        5.0 * self.t
            * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x + 0.2843 * x.powi(3)
                - 0.1015 * x.powi(4))
    }

    fn camber(&self, x: f64) -> (f64, f64) {
        let (m, p) = (self.m, self.p);
        if m == 0.0 {
            return (0.0, 0.0);
        }
        if x < p {
            (
                m / (p * p) * (2.0 * p * x - x * x),
                2.0 * m / (p * p) * (p - x),
            )
        } else {
            (
                m / ((1.0 - p) * (1.0 - p)) * (1.0 - 2.0 * p + 2.0 * p * x - x * x),
                2.0 * m / ((1.0 - p) * (1.0 - p)) * (p - x),
            )
        }
    }

    /// The surface point at chord station `x` on the upper (`up`) or lower surface.
    pub fn at(&self, x: f64, up: bool) -> (f64, f64) {
        let yt = self.thickness(x);
        let (yc, slope) = self.camber(x);
        let th = slope.atan();
        let s = if up { 1.0 } else { -1.0 };
        (x - s * yt * th.sin(), yc + s * yt * th.cos())
    }

    /// The point `depth` (in chords) inside the surface at `x`, along the inward normal.
    pub fn inside(&self, x: f64, up: bool, depth: f64) -> (f64, f64) {
        let h = 1e-5;
        let (a, b) = (self.at((x - h).max(0.0), up), self.at((x + h).min(1.0), up));
        let (tx, tz) = (b.0 - a.0, b.1 - a.1);
        let l = tx.hypot(tz);
        // Upper surface: inward is the tangent turned clockwise; lower: counter-clockwise.
        let (nx, nz) = if up {
            (tz / l, -tx / l)
        } else {
            (-tz / l, tx / l)
        };
        let p = self.at(x, up);
        (p.0 + depth * nx, p.1 + depth * nz)
    }
}

/// `n + 1` chord stations from `a` to `b`, closer together toward the leading edge.
pub fn stations(a: f64, b: f64, n: usize) -> Vec<f64> {
    // Cosine spacing on the whole chord, mapped to [a, b].
    let (ta, tb) = (
        ((1.0 - 2.0 * a).clamp(-1.0, 1.0)).acos(),
        ((1.0 - 2.0 * b).clamp(-1.0, 1.0)).acos(),
    );
    (0..=n)
        .map(|k| {
            let t = ta + (tb - ta) * k as f64 / n as f64;
            (1.0 - t.cos()) / 2.0
        })
        .collect()
}

/// One section placed in a sketch plane: unit-chord `(x, z)` goes to sketch `(u, v)`.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub chord: f64,
    /// Leading edge in the sketch: chord runs along `chord_axis`, thickness along the other.
    pub le: (f64, f64),
    /// True when the chord runs along the sketch's v axis (zx planes: u = z, v = x).
    pub chord_along_v: bool,
    /// Flip the thickness direction (a port-side vertical section).
    pub flip: bool,
}

impl Placed {
    pub fn map(&self, p: (f64, f64)) -> (f64, f64) {
        let (x, mut z) = (p.0 * self.chord, p.1 * self.chord);
        if self.flip {
            z = -z;
        }
        if self.chord_along_v {
            (self.le.0 + z, self.le.1 + x)
        } else {
            (self.le.0 + x, self.le.1 + z)
        }
    }
}

/// The surface from station `a` to `b` (upper) or `b` to `a` (lower runs leading edge aft
/// when `aft`), `depth` chords inside.
pub fn run(f: &Naca, from: f64, to: f64, up: bool, depth: f64, n: usize) -> Vec<(f64, f64)> {
    let (lo, hi) = (from.min(to), from.max(to));
    let mut xs = stations(lo, hi, n);
    if from > to {
        xs.reverse();
    }
    xs.into_iter()
        .map(|x| {
            if depth == 0.0 {
                f.at(x, up)
            } else {
                f.inside(x, up, depth)
            }
        })
        .collect()
}

/// The leading-edge wrap: upper surface from `a` forward round the nose to the lower surface at
/// `a`.
pub fn nose(f: &Naca, a: f64, n: usize) -> Vec<(f64, f64)> {
    let mut pts = run(f, a, 0.0, true, 0.0, n);
    pts.extend(run(f, 0.0, a, false, 0.0, n).into_iter().skip(1));
    pts
}
