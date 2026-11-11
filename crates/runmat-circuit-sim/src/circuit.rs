//! A flattened netlist built into devices and unknowns.
//!
//! Unknowns: node voltages first (in order of appearance), then device internals (series
//! resistance nodes, op-amp gain nodes) and branch currents (`i(<name>)`).

use crate::devices::bjt::Bjt;
use crate::devices::controlled::{Controlled, Law, Output, Poly};
use crate::devices::diode::Diode;
use crate::devices::junction::Params;
use crate::devices::mosfet::Mosfet;
use crate::devices::opamp::OpAmp;
use crate::devices::passive::{Capacitor, Inductor, Resistor};
use crate::devices::source::{ISource, VSource};
use crate::devices::switch::Switch;
use crate::devices::waveform::SourceSpec;
use crate::devices::{bjt, diode, mosfet, opamp, Device, Setup};
use crate::netlist::behave::{Behave, Ctl, Drives};
use crate::netlist::{is_ground, parse_number, Flat, FlatElement, FlatModel, Tok};
use crate::sparse::GND;
use crate::Options;
use std::collections::BTreeMap;

pub struct Circuit {
    pub names: Vec<String>,
    pub n_nodes: usize,
    pub node_index: BTreeMap<String, usize>,
    pub devices: Vec<Box<dyn Device>>,
    pub device_index: BTreeMap<String, usize>,
    pub state_is_flux: Vec<bool>,
    pub mem_init: Vec<f64>,
    pub options: Options,
    /// `.ic` node voltages.
    pub ics: Vec<(usize, f64)>,
    pub nodesets: Vec<(usize, f64)>,
    /// Model parameters accepted without effect, and similar notes.
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BuildError {
    pub element: String,
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "line {}: `{}`: {}",
            self.line, self.element, self.message
        )
    }
}

impl Circuit {
    pub fn size(&self) -> usize {
        self.names.len()
    }
    pub fn node(&self, name: &str) -> Option<usize> {
        if is_ground(name) {
            return Some(GND);
        }
        self.node_index.get(&name.to_ascii_lowercase()).copied()
    }
    pub fn device(&self, name: &str) -> Option<&dyn Device> {
        self.device_index
            .get(&name.to_ascii_lowercase())
            .map(|&i| self.devices[i].as_ref())
    }

    pub fn build(
        flat: &Flat,
        options: Options,
        ics: &[(String, f64)],
        nodesets: &[(String, f64)],
    ) -> Result<Circuit, BuildError> {
        let mut s = Setup::default();
        let mut node_index = BTreeMap::new();
        for e in &flat.elements {
            for n in &e.nodes {
                if !is_ground(n) && !node_index.contains_key(n) {
                    node_index.insert(n.clone(), s.names.len());
                    s.names.push(n.clone());
                }
            }
        }
        s.n_nodes = s.names.len();
        let mut b = Builder {
            flat,
            s,
            node_index,
            devices: vec![],
            inductors: vec![],
            branches: BTreeMap::new(),
            warnings: vec![],
            temp_k: options.temp_c + 273.15,
        };
        let (first, later): (Vec<&FlatElement>, Vec<&FlatElement>) = flat
            .elements
            .iter()
            .partition(|e| !matches!(e.kind, 'f' | 'h' | 'w' | 'k') && e.behave.is_none());
        for e in first.into_iter().chain(later) {
            b.element(e).map_err(|message| BuildError {
                element: e.name.clone(),
                line: e.line,
                message,
            })?;
        }
        let Builder {
            s,
            node_index,
            mut devices,
            inductors,
            warnings,
            ..
        } = b;
        for l in inductors {
            devices.push(Box::new(l));
        }
        let device_index = devices
            .iter()
            .enumerate()
            .map(|(i, d)| (d.name().to_string(), i))
            .collect();
        let lookup = |list: &[(String, f64)]| -> Vec<(usize, f64)> {
            list.iter()
                .filter_map(|(n, v)| node_index.get(n).map(|&i| (i, *v)))
                .collect()
        };
        Ok(Circuit {
            n_nodes: s.n_nodes,
            names: s.names,
            ics: lookup(ics),
            nodesets: lookup(nodesets),
            node_index,
            devices,
            device_index,
            state_is_flux: s.state_is_flux,
            mem_init: s.mem_init,
            options,
            warnings,
        })
    }
}

struct Builder<'a> {
    flat: &'a Flat,
    s: Setup,
    node_index: BTreeMap<String, usize>,
    devices: Vec<Box<dyn Device>>,
    inductors: Vec<Inductor>,
    branches: BTreeMap<String, usize>,
    warnings: Vec<String>,
    temp_k: f64,
}

fn num(t: Option<&Tok>) -> Option<f64> {
    t.and_then(|t| t.word()).and_then(parse_number)
}

fn kv(args: &[Tok], key: &str) -> Option<f64> {
    args.iter().find_map(|t| match t {
        Tok::Kv(k, v) if k == key => parse_number(v),
        _ => None,
    })
}

fn has_word(args: &[Tok], w: &str) -> bool {
    args.iter().any(|t| t.word() == Some(w))
}

impl<'a> Builder<'a> {
    fn node(&self, n: &str) -> usize {
        if is_ground(n) {
            GND
        } else {
            self.node_index[n]
        }
    }
    fn model(&self, name: Option<&str>, kinds: &[&str]) -> Result<&'a FlatModel, String> {
        let name = name.ok_or("names no model")?;
        let m = self
            .flat
            .models
            .get(name)
            .ok_or_else(|| format!("unknown model `{name}`"))?;
        if !kinds.contains(&m.kind.as_str()) {
            return Err(format!(
                "model `{name}` is a `{}`, expected {}",
                m.kind,
                kinds.join(" or ")
            ));
        }
        Ok(m)
    }
    fn note_unknown(&mut self, m: &FlatModel, known: &[&str], ignored: &[&str]) {
        let p = Params { model: Some(m) };
        let unknown = p.unknown(known, ignored);
        if !unknown.is_empty() {
            self.warnings.push(format!(
                "model `{}`: parameters not modelled: {}",
                m.name,
                unknown.join(", ")
            ));
        }
        let ign: Vec<&str> = ignored
            .iter()
            .copied()
            .filter(|k| p.has(k) && *k != "level" && *k != "mfg" && *k != "type")
            .collect();
        if !ign.is_empty() {
            self.warnings.push(format!(
                "model `{}`: accepted without effect: {}",
                m.name,
                ign.join(", ")
            ));
        }
    }
    fn branch_of(&self, name: &str) -> Result<usize, String> {
        self.branches.get(name).copied().ok_or_else(|| {
            format!("controlling source `{name}` is not a voltage source or branch device")
        })
    }

    /// A behavioural source: its controls become unknown pairs, its expression the law.
    fn behavioural(&mut self, e: &FlatElement, b: &Behave) -> Result<(), String> {
        let mut ctrl = vec![];
        for c in &b.ctl {
            ctrl.push(match c {
                Ctl::V(p, n) => {
                    let node = |x: &str| {
                        if is_ground(x) {
                            Ok(GND)
                        } else {
                            self.node_index
                                .get(x)
                                .copied()
                                .ok_or_else(|| format!("V({x}): no element connects node `{x}`"))
                        }
                    };
                    (node(p)?, node(n)?)
                }
                Ctl::I(s) => (self.branch_of(s)?, GND),
            });
        }
        let output = match b.drives {
            Drives::Voltage => {
                let branch = self.s.branch(&e.name);
                self.branches.insert(e.name.clone(), branch);
                Output::Voltage { branch }
            }
            Drives::Current => Output::Current,
        };
        let (p, n) = (self.node(&e.nodes[0]), self.node(&e.nodes[1]));
        self.devices.push(Box::new(Controlled {
            name: e.name.clone(),
            p,
            n,
            output,
            ctrl,
            law: Law::Expr(b.expr.clone()),
        }));
        Ok(())
    }

    fn element(&mut self, e: &FlatElement) -> Result<(), String> {
        if let Some(b) = &e.behave {
            return self.behavioural(e, b);
        }
        let nodes: Vec<usize> = e.nodes.iter().map(|n| self.node(n)).collect();
        let name = e.name.as_str();
        let value = || num(e.args.first()).ok_or_else(|| "needs a numeric value".to_string());
        match e.kind {
            'r' => {
                let r = num(e.args.first())
                    .or_else(|| kv(&e.args, "r"))
                    .ok_or("needs a resistance")?;
                if r == 0.0 {
                    return Err("zero resistance; use a 0 V source for a short".into());
                }
                self.devices.push(Box::new(Resistor {
                    name: name.into(),
                    a: nodes[0],
                    b: nodes[1],
                    ohms: r,
                }));
            }
            'c' => {
                let c = num(e.args.first())
                    .or_else(|| kv(&e.args, "c"))
                    .ok_or("needs a capacitance")?;
                let state = self.s.state(false);
                self.devices.push(Box::new(Capacitor {
                    name: name.into(),
                    a: nodes[0],
                    b: nodes[1],
                    farads: c,
                    state,
                    ic: kv(&e.args, "ic"),
                }));
            }
            'l' => {
                let l = num(e.args.first())
                    .or_else(|| kv(&e.args, "l"))
                    .ok_or("needs an inductance")?;
                let branch = self.s.branch(name);
                let state = self.s.state(true);
                self.branches.insert(name.into(), branch);
                self.inductors.push(Inductor {
                    name: name.into(),
                    a: nodes[0],
                    b: nodes[1],
                    henries: l,
                    branch,
                    state,
                    ic: kv(&e.args, "ic"),
                    couplings: vec![],
                });
            }
            'v' | 'i' => {
                let spec = SourceSpec::parse(&e.args)?;
                let mem = self.s.mem(spec.dc_value());
                if e.kind == 'v' {
                    let branch = self.s.branch(name);
                    self.branches.insert(name.into(), branch);
                    self.devices.push(Box::new(VSource {
                        name: name.into(),
                        p: nodes[0],
                        n: nodes[1],
                        branch,
                        spec,
                        mem,
                    }));
                } else {
                    self.devices.push(Box::new(ISource {
                        name: name.into(),
                        p: nodes[0],
                        n: nodes[1],
                        spec,
                        mem,
                    }));
                }
            }
            'e' | 'g' | 'f' | 'h' => {
                let poly = has_word(&e.args[..e.args.len().min(1)], "poly");
                let (ctrl, coeffs): (Vec<(usize, usize)>, Vec<f64>) = match (e.kind, poly) {
                    ('e' | 'g', false) => (vec![(nodes[2], nodes[3])], vec![0.0, value()?]),
                    ('e' | 'g', true) => {
                        let ctrl = nodes[2..].chunks(2).map(|c| (c[0], c[1])).collect();
                        (
                            ctrl,
                            e.args[2..]
                                .iter()
                                .filter_map(|t| t.word().and_then(parse_number))
                                .collect(),
                        )
                    }
                    (_, false) => {
                        let src = e
                            .args
                            .first()
                            .and_then(|t| t.word())
                            .ok_or("names no controlling source")?;
                        (
                            vec![(self.branch_of(src)?, GND)],
                            vec![0.0, num(e.args.get(1)).ok_or("needs a gain")?],
                        )
                    }
                    (_, true) => {
                        let n: usize = e
                            .args
                            .get(1)
                            .and_then(|t| t.word())
                            .and_then(|s| s.parse().ok())
                            .ok_or("POLY needs a dimension")?;
                        let mut ctrl = vec![];
                        for t in &e.args[2..2 + n] {
                            ctrl.push((self.branch_of(t.word().unwrap_or_default())?, GND));
                        }
                        (
                            ctrl,
                            e.args[2 + n..]
                                .iter()
                                .filter_map(|t| t.word().and_then(parse_number))
                                .collect(),
                        )
                    }
                };
                let dim = ctrl.len();
                let output = if matches!(e.kind, 'e' | 'h') {
                    let branch = self.s.branch(name);
                    self.branches.insert(name.into(), branch);
                    Output::Voltage { branch }
                } else {
                    Output::Current
                };
                let law = Law::Poly(if poly {
                    Poly::new(dim, coeffs)
                } else {
                    Poly::linear(coeffs[1])
                });
                self.devices.push(Box::new(Controlled {
                    name: name.into(),
                    p: nodes[0],
                    n: nodes[1],
                    output,
                    ctrl,
                    law,
                }));
            }
            'd' => {
                let m = self.model(e.args.first().and_then(|t| t.word()), &["d"])?;
                self.note_unknown(m, diode::KNOWN, diode::IGNORED);
                let area = num(e.args.get(1))
                    .or_else(|| kv(&e.args, "area"))
                    .unwrap_or(1.0);
                let d = Diode::new(
                    name,
                    nodes[0],
                    nodes[1],
                    &Params { model: Some(m) },
                    area,
                    has_word(&e.args, "off"),
                    self.temp_k,
                    &mut self.s,
                );
                self.devices.push(Box::new(d));
            }
            'q' => {
                let m = self.model(e.args.first().and_then(|t| t.word()), &["npn", "pnp"])?;
                self.note_unknown(m, bjt::KNOWN, bjt::IGNORED);
                let area = num(e.args.get(1))
                    .or_else(|| kv(&e.args, "area"))
                    .unwrap_or(1.0);
                let q = Bjt::new(
                    name,
                    m.kind == "pnp",
                    [nodes[0], nodes[1], nodes[2]],
                    &Params { model: Some(m) },
                    area,
                    has_word(&e.args, "off"),
                    self.temp_k,
                    &mut self.s,
                );
                self.devices.push(Box::new(q));
            }
            'm' => {
                let m = self.model(e.args.first().and_then(|t| t.word()), &["nmos", "pmos"])?;
                self.note_unknown(m, mosfet::KNOWN, mosfet::IGNORED);
                let d = Mosfet::new(
                    name,
                    m.kind == "pmos",
                    [nodes[0], nodes[1], nodes[2], nodes[3]],
                    &Params { model: Some(m) },
                    kv(&e.args, "l"),
                    kv(&e.args, "w"),
                    has_word(&e.args, "off"),
                    &mut self.s,
                )?;
                self.devices.push(Box::new(d));
            }
            's' | 'w' => {
                let (ctrl, model_tok) = if e.kind == 's' {
                    ((nodes[2], nodes[3]), e.args.first())
                } else {
                    let src = e
                        .args
                        .first()
                        .and_then(|t| t.word())
                        .ok_or("names no controlling source")?;
                    ((self.branch_of(src)?, GND), e.args.get(1))
                };
                let m = self.model(
                    model_tok.and_then(|t| t.word()),
                    if e.kind == 's' { &["sw"] } else { &["csw"] },
                )?;
                let p = Params { model: Some(m) };
                let (vt, vh) = if e.kind == 's' {
                    (p.get("vt", 0.0), p.get("vh", 0.0))
                } else {
                    (p.get("it", 0.0), p.get("ih", 0.0))
                };
                let on = has_word(&e.args, "on");
                let mem = self.s.mem(on as i32 as f64);
                self.s.mem(on as i32 as f64);
                self.devices.push(Box::new(Switch {
                    name: name.into(),
                    a: nodes[0],
                    b: nodes[1],
                    ctrl,
                    threshold: vt,
                    hysteresis: vh.abs(),
                    g_on: 1.0 / p.get("ron", 1.0),
                    g_off: 1.0 / p.get("roff", 1e12),
                    mem,
                }));
            }
            'x' => {
                let m = self.model(e.args.first().and_then(|t| t.word()), &["opamp"])?;
                self.note_unknown(m, opamp::KNOWN, &[]);
                let o = OpAmp::new(name, &nodes, &Params { model: Some(m) }, &mut self.s)?;
                self.branches.insert(name.into(), o.branch);
                self.devices.push(Box::new(o));
            }
            'k' => {
                let l1 = e
                    .args
                    .first()
                    .and_then(|t| t.word())
                    .ok_or("names no inductor")?
                    .to_string();
                let l2 = e
                    .args
                    .get(1)
                    .and_then(|t| t.word())
                    .ok_or("names one inductor; needs two")?
                    .to_string();
                let k = num(e.args.get(2)).ok_or("needs a coupling coefficient")?;
                if !(0.0..=1.0).contains(&k.abs()) {
                    return Err(format!("coupling coefficient {k} is outside [−1, 1]"));
                }
                let find = |n: &str| {
                    self.inductors
                        .iter()
                        .position(|l| l.name == n)
                        .ok_or(format!("`{n}` is not an inductor"))
                };
                let (i1, i2) = (find(&l1)?, find(&l2)?);
                let m = k * (self.inductors[i1].henries * self.inductors[i2].henries).sqrt();
                let (b1, b2) = (self.inductors[i1].branch, self.inductors[i2].branch);
                self.inductors[i1].couplings.push((b2, m));
                self.inductors[i2].couplings.push((b1, m));
            }
            k => return Err(format!("element type `{k}` is not supported")),
        }
        Ok(())
    }
}
