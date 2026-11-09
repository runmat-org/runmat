//! Electrical rules check over an elaborated [`Netlist`].
//!
//! Checks, each with a stable code:
//!
//! | Code | Severity | Meaning |
//! |---|---|---|
//! | `pin-conflict` | from the matrix | two pin types on one net that the [`ErcMatrix`] flags |
//! | `multiple-drivers` | error | more than one push-pull driver (output or power output) on a net |
//! | `power-not-driven` | error | power inputs on a net with no power output and no power flag |
//! | `input-not-driven` | warning | inputs on a net with nothing that can drive it |
//! | `pin-unconnected` | warning | a pin on no net, not no-connect and not marked no-ERC |
//! | `single-pin-net` | warning | a net with exactly one pin, not on a root-sheet port or a power flag (both reach outside the design) |
//! | `no-connect-connected` | error | a no-connect pin that is on a net with other pins |
//! | `no-erc-on-connected` | info | a no-ERC marker on a pin that is connected anyway |
//! | `blanket-class-kept` | info | a net a blanket covers keeps the other class its own statement gives it |
//! | `blanket-net-empty` | warning | a net a blanket covers has no pins, so its class and parameters reach nothing |
//!
//! Not-fitted components (in the elaborated variant) are left out, so ERC is variant-aware.
//! Every finding names its item (`R1.2` or a net name) and the other items involved.

use crate::elab::{Diagnostic, Netlist, Severity};
use crate::library::PinType;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErcLevel {
    Ok,
    Warning,
    Error,
}

/// Symmetric pin-type compatibility matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct ErcMatrix {
    m: [[ErcLevel; 12]; 12],
}

impl Default for ErcMatrix {
    fn default() -> Self {
        use ErcLevel::*;
        use PinType as P;
        let mut x = ErcMatrix { m: [[Ok; 12]; 12] };
        let rules: &[(P, P, ErcLevel)] = &[
            (P::Output, P::Output, Error),
            (P::Output, P::PowerOut, Error),
            (P::PowerOut, P::PowerOut, Error),
            (P::Output, P::Bidirectional, Warning),
            (P::Output, P::TriState, Warning),
            (P::Output, P::OpenCollector, Error),
            (P::Output, P::OpenEmitter, Error),
            (P::PowerOut, P::Bidirectional, Warning),
            (P::PowerOut, P::TriState, Error),
            (P::PowerOut, P::OpenCollector, Error),
            (P::PowerOut, P::OpenEmitter, Error),
            (P::OpenCollector, P::OpenEmitter, Warning),
            (P::Bidirectional, P::OpenEmitter, Warning),
            (P::TriState, P::OpenCollector, Warning),
            (P::TriState, P::OpenEmitter, Warning),
        ];
        for &(a, b, l) in rules {
            x.set(a, b, l);
        }
        for t in PinType::ALL {
            if t != P::Free {
                x.set(P::Unspecified, t, Warning);
                x.set(P::NotConnected, t, Error);
            }
        }
        for t in PinType::ALL {
            x.set(P::Free, t, Ok);
        }
        x
    }
}

impl ErcMatrix {
    pub fn set(&mut self, a: PinType, b: PinType, l: ErcLevel) {
        self.m[a.index()][b.index()] = l;
        self.m[b.index()][a.index()] = l;
    }
    pub fn get(&self, a: PinType, b: PinType) -> ErcLevel {
        self.m[a.index()][b.index()]
    }
}

#[derive(Clone, Debug, Default)]
pub struct ErcOptions {
    pub matrix: ErcMatrix,
    /// Nets treated as driven by a power source outside the design (a power flag).
    pub power_flags: BTreeSet<String>,
    /// Skip the single-pin-net check for these nets (test points, intentional stubs).
    pub allow_single: BTreeSet<String>,
}

/// The net-declaration parameter that marks a net as driven from outside the design (a
/// power flag): `net "+5V" = { class = power, global = true, power_flag = true }`.
pub const POWER_FLAG: &str = "power_flag";

impl ErcOptions {
    /// The options a design's own text sets: its power flags.
    pub fn for_design(design: &crate::ir::Design, nl: &Netlist) -> Self {
        let flagged: BTreeSet<&str> = design
            .sheets
            .values()
            .flat_map(|s| s.nets.iter())
            .filter(|n| {
                matches!(
                    n.params.get(POWER_FLAG).map(|p| &p.value),
                    Some(crate::ir::ParamValue::Bool(true))
                )
            })
            .map(|n| n.name.as_str())
            .collect();
        let power_flags = nl
            .nets
            .iter()
            .filter(|n| {
                flagged.contains(n.name.as_str())
                    || flagged.contains(n.path.leaf())
                    || n.aliases.iter().any(|a| flagged.contains(a.as_str()))
            })
            .map(|n| n.name.clone())
            .collect();
        ErcOptions {
            power_flags,
            ..ErcOptions::default()
        }
    }
}

fn pin_ref(nl: &Netlist, ci: usize, pi: usize) -> String {
    let c = &nl.components[ci];
    let p = &c.pins[pi];
    let pad = p.pads.first().cloned().unwrap_or_else(|| p.pin.clone());
    if pad == p.pin {
        format!("{}.{}", c.designator, p.pin)
    } else {
        format!("{}.{} ({})", c.designator, p.pin, pad)
    }
}

fn is_driver(t: PinType) -> bool {
    matches!(
        t,
        PinType::Output
            | PinType::Bidirectional
            | PinType::TriState
            | PinType::PowerOut
            | PinType::OpenCollector
            | PinType::OpenEmitter
            | PinType::Passive
            | PinType::Unspecified
    )
}

pub fn run(nl: &Netlist, opts: &ErcOptions) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    // The root sheet's ports are the design's own terminals: a net on one connects outside it.
    let root_ports: BTreeSet<String> = nl
        .sheets
        .iter()
        .filter(|(path, _)| path.depth() == 0)
        .flat_map(|(path, sheet)| {
            nl.sheet_ports
                .get(sheet)
                .into_iter()
                .flatten()
                .map(|p| crate::elab::port_net_name(path, p))
        })
        .collect();
    for net in &nl.nets {
        let pins: Vec<(usize, usize, PinType)> = net
            .pins
            .iter()
            .filter(|(ci, _)| nl.components[*ci].fitted())
            .map(|&(ci, pi)| (ci, pi, nl.components[ci].pins[pi].kind))
            .collect();
        if pins.is_empty() {
            continue;
        }
        let flagged = opts.power_flags.contains(&net.name)
            || net.aliases.iter().any(|a| opts.power_flags.contains(a));
        // Matrix: report each conflicting type pair once per net.
        let mut by_type: BTreeMap<PinType, Vec<String>> = BTreeMap::new();
        for &(ci, pi, t) in &pins {
            if !nl.no_erc.contains(&(ci, pi)) {
                by_type.entry(t).or_default().push(pin_ref(nl, ci, pi));
            }
        }
        let types: Vec<PinType> = by_type.keys().copied().collect();
        for (i, &a) in types.iter().enumerate() {
            for &b in &types[i..] {
                if a == b && by_type[&a].len() < 2 {
                    continue;
                }
                if a == PinType::NotConnected || b == PinType::NotConnected {
                    continue;
                }
                let lvl = opts.matrix.get(a, b);
                if lvl == ErcLevel::Ok {
                    continue;
                }
                // Push-pull driver pairs are reported as multiple-drivers below.
                let drv = |t: PinType| matches!(t, PinType::Output | PinType::PowerOut);
                if drv(a) && drv(b) {
                    continue;
                }
                let mut d = Diagnostic::new(
                    if lvl == ErcLevel::Error {
                        Severity::Error
                    } else {
                        Severity::Warning
                    },
                    "pin-conflict",
                    net.name.clone(),
                    format!("{} pin connected to {} pin", a.name(), b.name()),
                );
                for r in by_type[&a].iter().chain(if a == b {
                    [].iter()
                } else {
                    by_type[&b].iter()
                }) {
                    d = d.with(r.clone());
                }
                out.push(d);
            }
        }
        let drivers: Vec<String> = pins
            .iter()
            .filter(|(ci, pi, t)| {
                matches!(t, PinType::Output | PinType::PowerOut) && !nl.no_erc.contains(&(*ci, *pi))
            })
            .map(|&(ci, pi, _)| pin_ref(nl, ci, pi))
            .collect();
        if drivers.len() > 1 {
            let mut d = Diagnostic::new(
                Severity::Error,
                "multiple-drivers",
                net.name.clone(),
                format!("{} push-pull drivers on one net", drivers.len()),
            );
            for r in drivers {
                d = d.with(r);
            }
            out.push(d);
        }
        let has = |k: PinType| pins.iter().any(|p| p.2 == k);
        if has(PinType::PowerIn) && !has(PinType::PowerOut) && !flagged {
            let mut d = Diagnostic::new(
                Severity::Error,
                "power-not-driven",
                net.name.clone(),
                "power input pins with no power output on the net (add a source or a power flag)",
            );
            for &(ci, pi, t) in &pins {
                if t == PinType::PowerIn {
                    d = d.with(pin_ref(nl, ci, pi));
                }
            }
            out.push(d);
        }
        if has(PinType::Input) && !pins.iter().any(|p| is_driver(p.2)) && !flagged {
            let mut d = Diagnostic::new(
                Severity::Warning,
                "input-not-driven",
                net.name.clone(),
                "input pins with nothing able to drive the net",
            );
            for &(ci, pi, t) in &pins {
                if t == PinType::Input {
                    d = d.with(pin_ref(nl, ci, pi));
                }
            }
            out.push(d);
        }
        let outside = flagged
            || root_ports.contains(&net.name)
            || net.aliases.iter().any(|a| root_ports.contains(a));
        if pins.len() == 1 && !outside && !opts.allow_single.contains(&net.name) {
            let (ci, pi, t) = pins[0];
            if t != PinType::NotConnected && !nl.no_erc.contains(&(ci, pi)) {
                out.push(
                    Diagnostic::new(
                        Severity::Warning,
                        "single-pin-net",
                        net.name.clone(),
                        "net has only one pin",
                    )
                    .with(pin_ref(nl, ci, pi)),
                );
            }
        }
        if pins.len() > 1 {
            for &(ci, pi, t) in &pins {
                if t == PinType::NotConnected {
                    out.push(
                        Diagnostic::new(
                            Severity::Error,
                            "no-connect-connected",
                            pin_ref(nl, ci, pi),
                            format!("no-connect pin is on net `{}`", net.name),
                        )
                        .with(net.name.clone()),
                    );
                }
                if nl.no_erc.contains(&(ci, pi)) {
                    out.push(
                        Diagnostic::new(
                            Severity::Info,
                            "no-erc-on-connected",
                            pin_ref(nl, ci, pi),
                            format!("no-ERC marker on a pin connected to `{}`", net.name),
                        )
                        .with(net.name.clone()),
                    );
                }
            }
        }
    }
    for (ci, c) in nl.components.iter().enumerate() {
        if !c.fitted() {
            continue;
        }
        for (pi, p) in c.pins.iter().enumerate() {
            if p.net.is_none()
                && p.kind != PinType::NotConnected
                && p.kind != PinType::Free
                && !nl.no_erc.contains(&(ci, pi))
            {
                out.push(
                    Diagnostic::new(
                        Severity::Warning,
                        "pin-unconnected",
                        pin_ref(nl, ci, pi),
                        format!(
                            "{} pin {} is unconnected; mark it no-connect if intended",
                            p.kind.name(),
                            pin_ref(nl, ci, pi)
                        ),
                    )
                    .at(&c.source),
                );
            }
        }
    }
    for b in &nl.blankets {
        for &ni in &b.nets {
            let net = &nl.nets[ni];
            if let (Some(own), Some(set)) = (&net.class, &b.class) {
                if own != set {
                    out.push(
                        Diagnostic::new(
                            Severity::Info,
                            "blanket-class-kept",
                            net.name.clone(),
                            format!(
                                "net {} keeps its class `{own}`; blanket {} sets `{set}`",
                                net.name, b.name
                            ),
                        )
                        .with(b.name.clone()),
                    );
                }
            }
            if net.pins.is_empty() {
                out.push(
                    Diagnostic::new(
                        Severity::Warning,
                        "blanket-net-empty",
                        net.name.clone(),
                        format!(
                            "blanket {} covers net {}, which has no pins",
                            b.name, net.name
                        ),
                    )
                    .with(b.name.clone()),
                );
            }
        }
    }
    out.sort();
    out
}
