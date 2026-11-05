//! SPICE decks from the circuit IR.
//!
//! Each fitted component of the netlist becomes one element:
//!
//! - its part's [`SpiceModel`] subcircuit: `X<designator> <nodes in pin_order> <model>`, with
//!   the model library `.include`d once;
//! - or, with `device` set, that primitive with the part's nodes in `pin_order`;
//! - or, without a model, a primitive inferred from the entity prefix and the component's
//!   (or part's) parameters: `R` from `resistance`, `C` from `capacitance`, `L` from
//!   `inductance`.
//!
//! Components that have none of these are reported as [`EmitError`]s (or skipped with
//! [`EmitOptions::skip_unmodelled`]). Nets become nodes named after the net (sanitised);
//! the ground nets become node `0`. With [`EmitOptions::sense_pins`], every pin of every
//! element is fed through a zero-volt source `VS_<designator>_<k>` so the engine reports
//! each pin current.

use runmat_circuit::elab::{FitState, Netlist};
use runmat_circuit::library::{Pool, SpiceModel};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub struct EmitOptions {
    pub title: String,
    /// Nets tied to node 0.
    pub ground: Vec<String>,
    /// Feed every element pin through a 0 V sense source (for pin currents and power).
    pub sense_pins: bool,
    /// Leave out components with no model instead of failing.
    pub skip_unmodelled: bool,
    /// Sources and loads added to the circuit.
    pub stimuli: Vec<Stimulus>,
    /// Extra lines (models, options) placed before `.end`.
    pub extra: Vec<String>,
    /// Models for parts that carry none, by part number or designator (designator wins).
    pub part_models: BTreeMap<String, SpiceModel>,
}

impl Default for EmitOptions {
    fn default() -> Self {
        EmitOptions {
            title: "runmat".into(),
            ground: vec!["GND".into(), "0".into()],
            sense_pins: true,
            skip_unmodelled: false,
            stimuli: vec![],
            extra: vec![],
            part_models: BTreeMap::new(),
        }
    }
}

/// A source time function.
#[derive(Clone, Debug, PartialEq)]
pub enum Waveform {
    Dc(f64),
    /// DC value plus AC magnitude (and phase in degrees) for `.ac`.
    Ac {
        dc: f64,
        mag: f64,
        phase: f64,
    },
    Pulse {
        v1: f64,
        v2: f64,
        delay: f64,
        rise: f64,
        fall: f64,
        width: f64,
        period: f64,
    },
    Sin {
        offset: f64,
        amplitude: f64,
        freq: f64,
    },
    Pwl(Vec<(f64, f64)>),
}

impl Waveform {
    fn card(&self) -> String {
        match self {
            Waveform::Dc(v) => format!("DC {v:e}"),
            Waveform::Ac { dc, mag, phase } => format!("DC {dc:e} AC {mag:e} {phase:e}"),
            Waveform::Pulse {
                v1,
                v2,
                delay,
                rise,
                fall,
                width,
                period,
            } => format!("PULSE({v1:e} {v2:e} {delay:e} {rise:e} {fall:e} {width:e} {period:e})"),
            Waveform::Sin {
                offset,
                amplitude,
                freq,
            } => format!("SIN({offset:e} {amplitude:e} {freq:e})"),
            Waveform::Pwl(p) => {
                let pts: Vec<String> = p.iter().map(|(t, v)| format!("{t:e} {v:e}")).collect();
                format!("PWL({})", pts.join(" "))
            }
        }
    }
    /// The DC value (for operating points).
    pub fn dc(&self) -> f64 {
        match self {
            Waveform::Dc(v) => *v,
            Waveform::Ac { dc, .. } => *dc,
            Waveform::Pulse { v1, .. } => *v1,
            Waveform::Sin { offset, .. } => *offset,
            Waveform::Pwl(p) => p.first().map(|x| x.1).unwrap_or(0.0),
        }
    }
}

/// A source or load between two nets.
#[derive(Clone, Debug, PartialEq)]
pub enum Stimulus {
    Voltage {
        name: String,
        pos: String,
        neg: String,
        wave: Waveform,
    },
    Current {
        name: String,
        pos: String,
        neg: String,
        wave: Waveform,
    },
    Resistor {
        name: String,
        a: String,
        b: String,
        ohms: f64,
    },
}

/// One element of the deck.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    /// Deck name (`R1`, `XU3`, `VS_U3_2`, `VIN`).
    pub name: String,
    /// Designator of the component it realises (`None` for stimuli and senses).
    pub designator: Option<String>,
    pub nodes: Vec<String>,
    /// The rest of the card (value or model and parameters).
    pub value: String,
}

/// A pin sense source: current into `designator`'s pin from `net`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sense {
    pub designator: String,
    pub pin: String,
    pub net: String,
    pub source: String,
}

/// A SPICE deck.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Deck {
    pub title: String,
    pub includes: BTreeSet<String>,
    pub elements: Vec<Element>,
    pub senses: Vec<Sense>,
    pub extra: Vec<String>,
    /// Net name → node name.
    pub node_of: BTreeMap<String, String>,
}

impl Deck {
    /// The deck as text lines (title first, `.end` last; no analysis card).
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![format!("* {}", self.title)];
        for i in &self.includes {
            out.push(format!(".include \"{i}\""));
        }
        for e in &self.elements {
            let mut l = e.name.clone();
            for n in &e.nodes {
                let _ = write!(l, " {n}");
            }
            if !e.value.is_empty() {
                let _ = write!(l, " {}", e.value);
            }
            out.push(l);
        }
        out.extend(self.extra.iter().cloned());
        out.push(".end".into());
        out
    }
    pub fn text(&self) -> String {
        let mut s = self.lines().join("\n");
        s.push('\n');
        s
    }
    /// Every node except ground.
    pub fn nodes(&self) -> BTreeSet<String> {
        self.elements
            .iter()
            .flat_map(|e| e.nodes.iter().cloned())
            .filter(|n| n != "0")
            .collect()
    }
    /// The node of a net (or the net name itself if unknown).
    pub fn node(&self, net: &str) -> String {
        self.node_of
            .get(net)
            .cloned()
            .unwrap_or_else(|| node_name(net))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmitError {
    pub designator: String,
    pub message: String,
}

/// A SPICE-safe node name for a net.
pub fn node_name(net: &str) -> String {
    let s: String = net
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("n{s}")
    } else {
        s.to_lowercase()
    }
}

fn eng(v: f64) -> String {
    format!("{v:e}")
}

/// Emits a deck for the fitted components of `nl`.
pub fn emit(nl: &Netlist, pool: &Pool, opts: &EmitOptions) -> Result<Deck, Vec<EmitError>> {
    let mut deck = Deck {
        title: opts.title.clone(),
        extra: opts.extra.clone(),
        ..Default::default()
    };
    let ground: BTreeSet<&str> = opts.ground.iter().map(|s| s.as_str()).collect();
    let mut used: BTreeMap<String, String> = BTreeMap::new();
    let mut node = |net: &str, deck: &mut Deck| -> String {
        if ground.contains(net) {
            deck.node_of.insert(net.to_string(), "0".into());
            return "0".into();
        }
        if let Some(n) = deck.node_of.get(net) {
            return n.clone();
        }
        // Distinct nets keep distinct nodes even when sanitising collides.
        let mut n = node_name(net);
        let mut k = 1;
        while used.get(&n).is_some_and(|other| other != net) {
            k += 1;
            n = format!("{}_{k}", node_name(net));
        }
        used.insert(n.clone(), net.to_string());
        deck.node_of.insert(net.to_string(), n.clone());
        n
    };
    let mut errors = Vec::new();
    let mut nc = 0usize;
    for c in &nl.components {
        if c.fitting == FitState::NotFitted {
            continue;
        }
        let part = c
            .part
            .and_then(|u| pool.parts.get(&u))
            .map(|p| pool.effective_part(p));
        let entity_prefix = c
            .entity
            .and_then(|u| pool.entities.get(&u))
            .map(|e| e.prefix.clone())
            .unwrap_or_else(|| {
                c.designator
                    .trim_end_matches(|ch: char| ch.is_ascii_digit())
                    .to_string()
            });
        let param = |k: &str| {
            c.params
                .get(k)
                .or_else(|| part.as_ref().and_then(|p| p.params.get(k)))
                .and_then(|p| p.as_quantity())
                .map(|q| q.value)
        };
        let pin_net = |gate: &str, pin: &str| {
            c.pins
                .iter()
                .find(|p| p.pin == pin && (gate.is_empty() || p.gate == gate))
                .or_else(|| c.pins.iter().find(|p| p.pin == pin))
                .and_then(|p| p.net)
                .map(|n| nl.nets[n].name.clone())
        };
        // (pin label, net) in element order, plus the element's model text.
        let (letter, pins, value): (char, Vec<(String, Option<String>)>, String) =
            match part.as_ref().and_then(|p| p.spice.clone()).or_else(|| {
                opts.part_models
                    .get(&c.designator)
                    .or_else(|| part.as_ref().and_then(|p| opts.part_models.get(&p.mpn)))
                    .cloned()
            }) {
                Some(SpiceModel {
                    model,
                    library,
                    pin_order,
                    device,
                }) => {
                    if let Some(l) = library {
                        deck.includes.insert(l);
                    }
                    let pins = pin_order
                        .iter()
                        .map(|(g, p)| (format!("{g}.{p}"), pin_net(g, p)))
                        .collect();
                    (device.unwrap_or('X'), pins, model)
                }
                None => {
                    let two: Vec<(String, Option<String>)> = c
                        .pins
                        .iter()
                        .map(|p| (p.pin.clone(), p.net.map(|n| nl.nets[n].name.clone())))
                        .collect();
                    let prim = match entity_prefix.as_str() {
                        "R" => param("resistance").map(|v| ('R', eng(v))),
                        "C" => param("capacitance").map(|v| ('C', eng(v))),
                        "L" => param("inductance").map(|v| ('L', eng(v))),
                        _ => None,
                    };
                    match prim {
                        Some((l, v)) if two.len() == 2 => (l, two, v),
                        _ => {
                            if !opts.skip_unmodelled {
                                errors.push(EmitError {
                                    designator: c.designator.clone(),
                                    message: "no SPICE model on the part and no primitive value \
                                              (resistance, capacitance or inductance)"
                                        .into(),
                                });
                            }
                            continue;
                        }
                    }
                }
            };
        let mut nodes = Vec::new();
        for (k, (label, net)) in pins.iter().enumerate() {
            let n = match net {
                Some(net) => node(net, &mut deck),
                None => {
                    nc += 1;
                    format!("nc_{nc}")
                }
            };
            if opts.sense_pins && net.is_some() {
                let pin_node = format!("p_{}_{k}", node_name(&c.designator));
                let source = format!("VS_{}_{k}", node_name(&c.designator));
                deck.elements.push(Element {
                    name: source.clone(),
                    designator: None,
                    nodes: vec![n.clone(), pin_node.clone()],
                    value: "0".into(),
                });
                deck.senses.push(Sense {
                    designator: c.designator.clone(),
                    pin: label.clone(),
                    net: net.clone().unwrap(),
                    source,
                });
                nodes.push(pin_node);
            } else {
                nodes.push(n);
            }
        }
        let name = if letter == 'X' {
            format!("X{}", c.designator)
        } else if c.designator.starts_with(letter) {
            c.designator.clone()
        } else {
            format!("{letter}{}", c.designator)
        };
        deck.elements.push(Element {
            name,
            designator: Some(c.designator.clone()),
            nodes,
            value,
        });
    }
    for s in &opts.stimuli {
        let (name, a, b, value) = match s {
            Stimulus::Voltage {
                name,
                pos,
                neg,
                wave,
            } => (format!("V{name}"), pos, neg, wave.card()),
            Stimulus::Current {
                name,
                pos,
                neg,
                wave,
            } => (format!("I{name}"), pos, neg, wave.card()),
            Stimulus::Resistor { name, a, b, ohms } => (format!("R{name}"), a, b, eng(*ohms)),
        };
        let (na, nb) = (node(a, &mut deck), node(b, &mut deck));
        deck.elements.push(Element {
            name,
            designator: None,
            nodes: vec![na, nb],
            value,
        });
    }
    if errors.is_empty() {
        Ok(deck)
    } else {
        Err(errors)
    }
}
