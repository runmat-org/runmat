//! A small exact engine for tests: modified nodal analysis of resistors, independent DC
//! voltage and current sources, inductors (shorts) and capacitors (open) at DC. It runs
//! operating points and DC sweeps; anything else (subcircuits, semiconductors, AC,
//! transient) is [`SpiceError::Unsupported`]. Results use the same names and sign
//! conventions as the ngspice engine.

use crate::netlist::Deck;
use crate::{Analysis, OpPoint, SimResult, SpiceEngine, SpiceError, Sweep};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct FakeEngine {
    /// Every deck it was given, for assertions.
    pub decks: Vec<String>,
}

fn number(s: &str) -> Option<f64> {
    let t = s.trim().to_ascii_lowercase();
    if let Ok(v) = t.parse::<f64>() {
        return Some(v);
    }
    // SPICE scale suffixes.
    for (suf, k) in [
        ("meg", 1e6),
        ("mil", 25.4e-6),
        ("t", 1e12),
        ("g", 1e9),
        ("k", 1e3),
        ("m", 1e-3),
        ("u", 1e-6),
        ("n", 1e-9),
        ("p", 1e-12),
        ("f", 1e-15),
    ] {
        if let Some(x) = t.strip_suffix(suf) {
            return x.parse::<f64>().ok().map(|v| v * k);
        }
    }
    None
}

/// The DC value of a source card (`0`, `5`, `DC 5`, `DC 5 AC 1 0`, `PULSE(v1 …)`).
fn source_dc(value: &str) -> Option<f64> {
    let t: Vec<&str> = value.split_whitespace().collect();
    match t.first().map(|s| s.to_ascii_uppercase()) {
        None => Some(0.0),
        Some(h) if h == "DC" => t.get(1).and_then(|v| number(v)),
        Some(h) if h.starts_with("PULSE(") || h.starts_with("SIN(") || h.starts_with("PWL(") => {
            let inner = value.split_once('(')?.1;
            let first = inner.split_whitespace().next()?.trim_end_matches(')');
            let v = number(first)?;
            // PWL's first pair is (time, value).
            if h.starts_with("PWL(") {
                inner
                    .split_whitespace()
                    .nth(1)
                    .and_then(|x| number(x.trim_end_matches(')')))
            } else {
                Some(v)
            }
        }
        Some(_) => number(t[0]),
    }
}

impl FakeEngine {
    pub fn new() -> Self {
        FakeEngine::default()
    }

    fn solve(&self, deck: &Deck, overrides: &BTreeMap<String, f64>) -> Result<OpPoint, SpiceError> {
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        for n in deck.nodes() {
            let k = index.len();
            index.insert(n, k);
        }
        let nn = index.len();
        let vsrc: Vec<usize> = (0..deck.elements.len())
            .filter(|&i| {
                let c = deck.elements[i]
                    .name
                    .chars()
                    .next()
                    .unwrap_or(' ')
                    .to_ascii_uppercase();
                c == 'V' || c == 'L'
            })
            .collect();
        let m = nn + vsrc.len();
        let mut a = vec![vec![0.0f64; m]; m];
        let mut z = vec![0.0f64; m];
        let idx = |n: &str| index.get(n).copied();
        for (i, e) in deck.elements.iter().enumerate() {
            let kind = e.name.chars().next().unwrap_or(' ').to_ascii_uppercase();
            let (p, q) = (
                e.nodes.first().and_then(|n| idx(n)),
                e.nodes.get(1).and_then(|n| idx(n)),
            );
            match kind {
                'R' => {
                    let r = number(&e.value).ok_or_else(|| SpiceError::Simulation {
                        message: format!("bad resistance on {}", e.name),
                        log: vec![],
                    })?;
                    let g = 1.0 / r;
                    if let Some(p) = p {
                        a[p][p] += g;
                    }
                    if let Some(q) = q {
                        a[q][q] += g;
                    }
                    if let (Some(p), Some(q)) = (p, q) {
                        a[p][q] -= g;
                        a[q][p] -= g;
                    }
                }
                'C' => {}
                'I' => {
                    let v = overrides
                        .get(&e.name)
                        .copied()
                        .or_else(|| source_dc(&e.value))
                        .unwrap_or(0.0);
                    // Current flows from p through the source to q.
                    if let Some(p) = p {
                        z[p] -= v;
                    }
                    if let Some(q) = q {
                        z[q] += v;
                    }
                }
                'V' | 'L' => {
                    let k = nn + vsrc.iter().position(|&x| x == i).unwrap();
                    if let Some(p) = p {
                        a[p][k] += 1.0;
                        a[k][p] += 1.0;
                    }
                    if let Some(q) = q {
                        a[q][k] -= 1.0;
                        a[k][q] -= 1.0;
                    }
                    z[k] = if kind == 'L' {
                        0.0
                    } else {
                        overrides
                            .get(&e.name)
                            .copied()
                            .or_else(|| source_dc(&e.value))
                            .unwrap_or(0.0)
                    };
                }
                other => {
                    return Err(SpiceError::Unsupported(format!(
                        "the fake engine has no `{other}` devices ({})",
                        e.name
                    )))
                }
            }
        }
        // Gaussian elimination with partial pivoting.
        for col in 0..m {
            let piv = (col..m)
                .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
                .unwrap();
            if a[piv][col].abs() < 1e-18 {
                return Err(SpiceError::Simulation {
                    message: "singular matrix (floating node or source loop)".into(),
                    log: vec![],
                });
            }
            a.swap(col, piv);
            z.swap(col, piv);
            for r in col + 1..m {
                let f = a[r][col] / a[col][col];
                if f != 0.0 {
                    for c in col..m {
                        a[r][c] -= f * a[col][c];
                    }
                    z[r] -= f * z[col];
                }
            }
        }
        let mut x = vec![0.0; m];
        for r in (0..m).rev() {
            let s: f64 = (r + 1..m).map(|c| a[r][c] * x[c]).sum();
            x[r] = (z[r] - s) / a[r][r];
        }
        let mut op = OpPoint::default();
        for (net, node) in &deck.node_of {
            let v = if node == "0" {
                0.0
            } else {
                idx(node).map(|k| x[k]).unwrap_or(0.0)
            };
            op.voltages.insert(net.clone(), v);
        }
        for (j, &i) in vsrc.iter().enumerate() {
            // MNA solves the current from + through the source to −, as SPICE reports it.
            op.currents.insert(deck.elements[i].name.clone(), x[nn + j]);
        }
        Ok(op)
    }
}

impl SpiceEngine for FakeEngine {
    fn run(&mut self, deck: &Deck, analysis: &Analysis) -> Result<SimResult, SpiceError> {
        self.decks.push(deck.text());
        match analysis {
            Analysis::Op => Ok(SimResult::Op(self.solve(deck, &BTreeMap::new())?)),
            Analysis::Dc {
                source,
                start,
                stop,
                step,
            } => {
                if !deck
                    .elements
                    .iter()
                    .any(|e| e.name.eq_ignore_ascii_case(source))
                {
                    return Err(SpiceError::Simulation {
                        message: format!("no source `{source}`"),
                        log: vec![],
                    });
                }
                let name = deck
                    .elements
                    .iter()
                    .find(|e| e.name.eq_ignore_ascii_case(source))
                    .unwrap()
                    .name
                    .clone();
                let n = (((stop - start) / step).abs().round() as usize) + 1;
                let mut sw = Sweep::default();
                for k in 0..n {
                    let v = start + step * k as f64;
                    let op = self.solve(deck, &BTreeMap::from([(name.clone(), v)]))?;
                    sw.x.push(v);
                    for (net, val) in op.voltages {
                        sw.voltages.entry(net).or_default().push(val);
                    }
                    for (s, val) in op.currents {
                        sw.currents.entry(s).or_default().push(val);
                    }
                }
                Ok(SimResult::Dc(sw))
            }
            other => Err(SpiceError::Unsupported(format!(
                "the fake engine runs op and dc only, not {}",
                other.kind()
            ))),
        }
    }
    fn describe(&self) -> String {
        "fake MNA engine".into()
    }
}
