//! Named design-review suites: deterministic checks over the elaborated design, one finding
//! per failing rule, each with a typed fix offered as a [`ModelEdit`].
//!
//! Severity follows the review convention: [`Severity::Error`] means "won't work",
//! [`Severity::Warning`] means "won't run at full specification".
//!
//! | Suite | Rules |
//! |---|---|
//! | `decoupling` | every IC power-input pin has a capacitor from its rail to ground (capacitors on a rail are shared out in designator order) |
//! | `i2c` | every SCL/SDA net with a device has a pull-up to a rail, 1 kΩ–10 kΩ |
//! | `crystal` | each crystal pin has a load capacitor to ground and `C1·C2/(C1+C2) + C_stray` is within 10 % of the crystal's load capacitance |
//! | `straps` | reset and boot pins are not floating: `NRST`/`RESET` pulled up (or driven), `BOOT0`/`BOOT` pulled down |
//! | `usb-c` | a USB-C sink has 5.1 kΩ pull-downs (Rd) on CC1 and CC2; D+/D− are connected and paired |
//! | `switcher` | a switching regulator has an inductor on SW, input and output capacitors, and a feedback divider setting the output within 3 % of the rail's nominal voltage |
//!
//! Fixes are expressed against the sheet that holds the anchor component, using pins of
//! components in that sheet as endpoints, so they apply to the authored hierarchy.

use crate::elab::{CompInst, Netlist, Severity};
use crate::ir::*;
use crate::library::{PinType, Pool};
use runmat_rules::{Dim, Quantity};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub enum ModelEdit {
    /// Add a component in `sheet`, connecting its pins to endpoints in that sheet.
    AddComponent {
        sheet: String,
        decl: ComponentDecl,
        connect: Vec<(String, Endpoint)>,
    },
    /// Several edits applied together.
    Batch(Vec<ModelEdit>),
    /// Set a parameter on a component (by local name) in `sheet`.
    SetParam {
        sheet: String,
        component: String,
        key: String,
        value: Param,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReviewFinding {
    pub suite: String,
    pub rule: String,
    pub severity: Severity,
    /// Designators, `designator.pin` refs or net names.
    pub items: Vec<String>,
    pub message: String,
    pub fix: Option<ModelEdit>,
}

/// Applies a fix to the authored design.
pub fn apply(design: &mut Design, edit: &ModelEdit) -> Result<(), String> {
    match edit {
        ModelEdit::AddComponent {
            sheet,
            decl,
            connect,
        } => {
            let s = design
                .sheets
                .get_mut(sheet)
                .ok_or_else(|| format!("no sheet `{sheet}`"))?;
            // Unique local name in the sheet.
            let mut d = decl.clone();
            let base = d.name.trim_end_matches('?').to_string();
            let mut k = 0;
            loop {
                d.name = format!("{base}?fix{k}");
                if !s.components.iter().any(|c| c.name == d.name) {
                    break;
                }
                k += 1;
            }
            let name = d.name.clone();
            s.components.push(d);
            for (pin, ep) in connect {
                s.connections.push(Connection {
                    ends: vec![Endpoint::pin(&name, pin), ep.clone()],
                    source: Default::default(),
                });
            }
            Ok(())
        }
        ModelEdit::Batch(v) => v.iter().try_for_each(|e| apply(design, e)),
        ModelEdit::SetParam {
            sheet,
            component,
            key,
            value,
        } => {
            let s = design
                .sheets
                .get_mut(sheet)
                .ok_or_else(|| format!("no sheet `{sheet}`"))?;
            let c = s
                .components
                .iter_mut()
                .find(|c| &c.name == component)
                .ok_or_else(|| format!("no component `{component}` in `{sheet}`"))?;
            c.params.insert(key.clone(), value.clone());
            Ok(())
        }
    }
}

/// Options shared by the suites.
#[derive(Clone, Debug)]
pub struct ReviewOptions {
    /// Part used for added decoupling capacitors.
    pub decoupling_part: String,
    pub pullup_part: String,
    /// Stray capacitance assumed per crystal pin pair, F.
    pub crystal_stray: f64,
    /// Feedback reference voltage when the regulator part has no `vref` parameter, V.
    pub default_vref: f64,
}

impl Default for ReviewOptions {
    fn default() -> Self {
        ReviewOptions {
            decoupling_part: "C0402-100N".into(),
            pullup_part: "R0402-10K".into(),
            crystal_stray: 3e-12,
            default_vref: 0.8,
        }
    }
}

struct Ctx<'a> {
    nl: &'a Netlist,
    pool: &'a Pool,
    /// Two-pin components: index → (net a, net b).
    two: BTreeMap<usize, (Option<usize>, Option<usize>)>,
}

fn is_ground(name: &str) -> bool {
    let n = name.rsplit('/').next().unwrap_or(name).to_ascii_uppercase();
    n.starts_with("GND") || matches!(n.as_str(), "AGND" | "DGND" | "PGND" | "VSS" | "0V")
}

fn rail_voltage(name: &str) -> Option<f64> {
    // +3V3, 3V3, +5V, 1V8, +12V, VCC_3V3 → volts.
    let n = name.rsplit('/').next().unwrap_or(name);
    let t = n.rsplit('_').next().unwrap_or(n).trim_start_matches('+');
    let (a, b) = t.split_once(['V', 'v'])?;
    let whole: f64 = a.parse().ok()?;
    let frac = if b.is_empty() {
        0.0
    } else {
        format!("0.{b}").parse().ok()?
    };
    Some(whole + frac)
}

impl<'a> Ctx<'a> {
    fn new(nl: &'a Netlist, pool: &'a Pool) -> Self {
        let mut two = BTreeMap::new();
        for (i, c) in nl.components.iter().enumerate() {
            if c.fitted() && c.pins.len() == 2 {
                two.insert(i, (c.pins[0].net, c.pins[1].net));
            }
        }
        Ctx { nl, pool, two }
    }
    fn prefix(&self, c: &CompInst) -> String {
        c.entity
            .and_then(|e| self.pool.entities.get(&e))
            .map(|e| e.prefix.clone())
            .unwrap_or_else(|| {
                c.designator
                    .chars()
                    .take_while(|x| x.is_ascii_alphabetic())
                    .collect()
            })
    }
    fn is_kind(&self, i: usize, prefix: &str) -> bool {
        self.prefix(&self.nl.components[i]) == prefix
    }
    fn value(&self, i: usize, key: &str) -> Option<f64> {
        self.nl.components[i]
            .params
            .get(key)
            .and_then(|p| p.as_quantity())
            .map(|q| q.value)
    }
    /// Two-pin parts of `prefix` between nets `a` and `b` (either order).
    fn between(&self, prefix: &str, a: usize, b: impl Fn(usize) -> bool) -> Vec<usize> {
        self.two
            .iter()
            .filter(|(i, (x, y))| {
                self.is_kind(**i, prefix)
                    && ((*x == Some(a) && y.is_some_and(&b))
                        || (*y == Some(a) && x.is_some_and(&b)))
            })
            .map(|(i, _)| *i)
            .collect()
    }
    fn net_name(&self, n: usize) -> &str {
        &self.nl.nets[n].name
    }
    fn ground(&self, n: usize) -> bool {
        is_ground(self.net_name(n))
    }
    fn rail(&self, n: usize) -> bool {
        let net = &self.nl.nets[n];
        if self.ground(n) {
            return false;
        }
        net.pins.iter().any(|(c, p)| {
            matches!(
                self.nl.components[*c].pins[*p].kind,
                PinType::PowerOut | PinType::PowerIn
            )
        }) || rail_voltage(&net.name).is_some()
            || net.class.as_deref() == Some("power")
    }
    fn local(&self, c: &CompInst) -> String {
        c.path.leaf().to_string()
    }
    fn ground_anchor(&self, sheet_comp: &CompInst) -> Endpoint {
        // A ground pin on the same component, else the global GND net.
        for p in &sheet_comp.pins {
            if p.net.is_some_and(|n| self.ground(n)) {
                return Endpoint::pin(&self.local(sheet_comp), &format!("{}.{}", p.gate, p.pin));
            }
        }
        Endpoint::net("GND")
    }
}

fn add(
    suite: &str,
    rule: &str,
    sev: Severity,
    items: Vec<String>,
    message: String,
    fix: Option<ModelEdit>,
) -> ReviewFinding {
    ReviewFinding {
        suite: suite.into(),
        rule: rule.into(),
        severity: sev,
        items,
        message,
        fix,
    }
}

fn two_pin(part: &str, name: &str) -> ComponentDecl {
    ComponentDecl::new(name, PartRef::Part(part.into()))
}

pub fn decoupling(nl: &Netlist, pool: &Pool, o: &ReviewOptions) -> Vec<ReviewFinding> {
    let cx = Ctx::new(nl, pool);
    let mut out = Vec::new();
    // Capacitors available per rail, consumed in designator order.
    let mut caps: BTreeMap<usize, usize> = BTreeMap::new();
    for (i, (a, b)) in &cx.two {
        if !cx.is_kind(*i, "C") {
            continue;
        }
        if let (Some(a), Some(b)) = (a, b) {
            if cx.ground(*b) && !cx.ground(*a) {
                *caps.entry(*a).or_default() += 1;
            } else if cx.ground(*a) && !cx.ground(*b) {
                *caps.entry(*b).or_default() += 1;
            }
        }
    }
    let mut ics: Vec<(String, usize)> = nl
        .components
        .iter()
        .enumerate()
        .filter(|(_, c)| c.fitted() && cx.prefix(c) == "U")
        .map(|(i, c)| (c.designator.clone(), i))
        .collect();
    ics.sort_by_key(|(d, _)| crate::fab::natural_key(d));
    for (_, ci) in ics {
        let c = &nl.components[ci];
        let mut seen = BTreeSet::new();
        for p in &c.pins {
            if p.kind != PinType::PowerIn {
                continue;
            }
            let Some(n) = p.net else { continue };
            if cx.ground(n) || !seen.insert(n) {
                continue;
            }
            let left = caps.entry(n).or_default();
            if *left > 0 {
                *left -= 1;
                continue;
            }
            let fix = ModelEdit::AddComponent {
                sheet: c.sheet.clone(),
                decl: two_pin(&o.decoupling_part, "C?"),
                connect: vec![
                    (
                        "1".into(),
                        Endpoint::pin(&cx.local(c), &format!("{}.{}", p.gate, p.pin)),
                    ),
                    ("2".into(), cx.ground_anchor(c)),
                ],
            };
            out.push(add(
                "decoupling",
                "cap-per-power-pin",
                Severity::Warning,
                vec![
                    format!("{}.{}", c.designator, p.pin),
                    nl.nets[n].name.clone(),
                ],
                format!(
                    "{} power pin {} on {} has no decoupling capacitor",
                    c.designator, p.pin, nl.nets[n].name
                ),
                Some(fix),
            ));
        }
    }
    out
}

pub fn i2c(nl: &Netlist, pool: &Pool, o: &ReviewOptions) -> Vec<ReviewFinding> {
    let cx = Ctx::new(nl, pool);
    let mut out = Vec::new();
    for (ni, net) in nl.nets.iter().enumerate() {
        let leaf = net
            .name
            .rsplit('/')
            .next()
            .unwrap_or(&net.name)
            .to_ascii_uppercase();
        let named = leaf.contains("SCL") || leaf.contains("SDA");
        let pins: Vec<(usize, usize)> = net
            .pins
            .iter()
            .copied()
            .filter(|(c, _)| nl.components[*c].fitted())
            .collect();
        let by_pin = pins
            .iter()
            .any(|(c, p)| matches!(nl.components[*c].pins[*p].pin.as_str(), "SCL" | "SDA"));
        if !(named || by_pin) || pins.len() < 2 {
            continue;
        }
        let pulls = cx.between("R", ni, |m| cx.rail(m));
        if pulls.is_empty() {
            // Anchor: a device pin on the bus and that device's supply pin.
            let dev = pins.iter().map(|(c, _)| &nl.components[*c]).find(|c| {
                c.pins
                    .iter()
                    .any(|p| p.kind == PinType::PowerIn && p.net.is_some_and(|n| cx.rail(n)))
            });
            let fix = dev.map(|d| {
                let bus_pin = d.pins.iter().find(|p| p.net == Some(ni)).unwrap();
                let vdd = d
                    .pins
                    .iter()
                    .find(|p| p.kind == PinType::PowerIn && p.net.is_some_and(|n| cx.rail(n)))
                    .unwrap();
                ModelEdit::AddComponent {
                    sheet: d.sheet.clone(),
                    decl: two_pin(&o.pullup_part, "R?"),
                    connect: vec![
                        (
                            "1".into(),
                            Endpoint::pin(
                                &cx.local(d),
                                &format!("{}.{}", bus_pin.gate, bus_pin.pin),
                            ),
                        ),
                        (
                            "2".into(),
                            Endpoint::pin(&cx.local(d), &format!("{}.{}", vdd.gate, vdd.pin)),
                        ),
                    ],
                }
            });
            out.push(add(
                "i2c",
                "pull-up",
                Severity::Error,
                vec![net.name.clone()],
                format!("I²C line {} has no pull-up resistor", net.name),
                fix,
            ));
            continue;
        }
        for r in pulls {
            if let Some(v) = cx.value(r, "resistance") {
                if !(1e3..=10e3 + 1e-6).contains(&v) {
                    out.push(add(
                        "i2c",
                        "pull-up-value",
                        Severity::Warning,
                        vec![nl.components[r].designator.clone(), net.name.clone()],
                        format!(
                            "pull-up {} is {v} Ω; 1 kΩ–10 kΩ expected",
                            nl.components[r].designator
                        ),
                        None,
                    ));
                }
            }
        }
    }
    out
}

pub fn crystal(nl: &Netlist, pool: &Pool, o: &ReviewOptions) -> Vec<ReviewFinding> {
    let cx = Ctx::new(nl, pool);
    let mut out = Vec::new();
    for (i, c) in nl.components.iter().enumerate() {
        if !c.fitted() || cx.prefix(c) != "Y" {
            continue;
        }
        let Some(cl) = cx.value(i, "load_capacitance") else {
            continue;
        };
        let mut loads = Vec::new();
        for p in &c.pins {
            let Some(n) = p.net else { continue };
            let caps = cx.between("C", n, |m| cx.ground(m));
            match caps.first() {
                Some(&k) => loads.push((k, cx.value(k, "capacitance"))),
                None => {
                    out.push(add(
                        "crystal",
                        "load-cap",
                        Severity::Error,
                        vec![format!("{}.{}", c.designator, p.pin)],
                        format!(
                            "crystal {} pin {} has no load capacitor to ground",
                            c.designator, p.pin
                        ),
                        Some(ModelEdit::AddComponent {
                            sheet: c.sheet.clone(),
                            decl: two_pin(&o.decoupling_part, "C?"),
                            connect: vec![
                                ("1".into(), Endpoint::pin(&cx.local(c), &p.pin)),
                                ("2".into(), Endpoint::net("GND")),
                            ],
                        }),
                    ));
                }
            }
        }
        if let [(a, Some(c1)), (b, Some(c2))] = loads[..] {
            let eff = c1 * c2 / (c1 + c2) + o.crystal_stray;
            if (eff - cl).abs() > 0.1 * cl {
                let want = 2.0 * (cl - o.crystal_stray);
                let mk = |k: usize| ModelEdit::SetParam {
                    sheet: nl.components[k].sheet.clone(),
                    component: cx.local(&nl.components[k]),
                    key: "capacitance".into(),
                    value: Param::quantity(
                        Quantity::new(want, Dim::CAPACITANCE),
                        Tolerance::pm_percent(5.0),
                    ),
                };
                out.push(add(
                    "crystal",
                    "load-match",
                    Severity::Warning,
                    vec![
                        c.designator.clone(),
                        nl.components[a].designator.clone(),
                        nl.components[b].designator.clone(),
                    ],
                    format!(
                        "effective load {:.1} pF vs crystal CL {:.1} pF; use {:.1} pF each",
                        eff * 1e12,
                        cl * 1e12,
                        want * 1e12
                    ),
                    Some(ModelEdit::Batch(vec![mk(a), mk(b)])),
                ));
            }
        }
    }
    out
}

pub fn straps(nl: &Netlist, pool: &Pool, o: &ReviewOptions) -> Vec<ReviewFinding> {
    let cx = Ctx::new(nl, pool);
    let mut out = Vec::new();
    for c in nl
        .components
        .iter()
        .filter(|c| c.fitted() && cx.prefix(c) == "U")
    {
        for p in &c.pins {
            let up = matches!(p.pin.as_str(), "NRST" | "RESET" | "nRESET" | "RST");
            let down = matches!(p.pin.as_str(), "BOOT0" | "BOOT");
            if !up && !down {
                continue;
            }
            let ok = p.net.is_some_and(|n| {
                if up {
                    !cx.between("R", n, |m| cx.rail(m)).is_empty()
                        || nl.nets[n]
                            .pins
                            .iter()
                            .any(|(ci, pi)| nl.components[*ci].pins[*pi].kind == PinType::Output)
                } else {
                    !cx.between("R", n, |m| cx.ground(m)).is_empty() || cx.ground(n)
                }
            });
            if ok {
                continue;
            }
            let rail_pin = c
                .pins
                .iter()
                .find(|q| q.kind == PinType::PowerIn && q.net.is_some_and(|n| cx.rail(n)));
            let other = if up {
                rail_pin.map(|q| Endpoint::pin(&cx.local(c), &format!("{}.{}", q.gate, q.pin)))
            } else {
                Some(cx.ground_anchor(c))
            };
            let fix = other.map(|e| ModelEdit::AddComponent {
                sheet: c.sheet.clone(),
                decl: two_pin(&o.pullup_part, "R?"),
                connect: vec![
                    (
                        "1".into(),
                        Endpoint::pin(&cx.local(c), &format!("{}.{}", p.gate, p.pin)),
                    ),
                    ("2".into(), e),
                ],
            });
            out.push(add(
                "straps",
                if up {
                    "reset-pull-up"
                } else {
                    "boot-pull-down"
                },
                Severity::Error,
                vec![format!("{}.{}", c.designator, p.pin)],
                format!(
                    "{} {} is floating; it needs a pull-{}",
                    c.designator,
                    p.pin,
                    if up { "up" } else { "down" }
                ),
                fix,
            ));
        }
    }
    out
}

pub fn usb_c(nl: &Netlist, pool: &Pool, _o: &ReviewOptions) -> Vec<ReviewFinding> {
    let cx = Ctx::new(nl, pool);
    let mut out = Vec::new();
    for c in nl.components.iter().filter(|c| c.fitted()) {
        let has = |n: &str| c.pins.iter().any(|p| p.pin == n);
        if !(has("CC1") && has("CC2") && has("VBUS")) {
            continue;
        }
        for cc in ["CC1", "CC2"] {
            let p = c.pins.iter().find(|p| p.pin == cc).unwrap();
            let rd = p
                .net
                .map(|n| cx.between("R", n, |m| cx.ground(m)))
                .unwrap_or_default();
            match rd.first() {
                None => out.push(add(
                    "usb-c",
                    "cc-rd",
                    Severity::Error,
                    vec![format!("{}.{cc}", c.designator)],
                    format!(
                        "{} {cc} has no 5.1 kΩ pull-down: a USB-C source will not supply VBUS",
                        c.designator
                    ),
                    Some(ModelEdit::AddComponent {
                        sheet: c.sheet.clone(),
                        decl: {
                            let mut d = two_pin("R0402-10K", "R?");
                            d.params.insert(
                                "resistance".into(),
                                Param::quantity(
                                    Quantity::new(5.1e3, Dim::RESISTANCE),
                                    Tolerance::pm_percent(1.0),
                                ),
                            );
                            d
                        },
                        connect: vec![
                            ("1".into(), Endpoint::pin(&cx.local(c), cc)),
                            ("2".into(), cx.ground_anchor(c)),
                        ],
                    }),
                )),
                Some(&r) => {
                    if let Some(v) = cx.value(r, "resistance") {
                        if (v - 5.1e3).abs() > 0.1 * 5.1e3 {
                            out.push(add(
                                "usb-c",
                                "cc-rd-value",
                                Severity::Warning,
                                vec![nl.components[r].designator.clone()],
                                format!("Rd on {cc} is {v} Ω; 5.1 kΩ is required for a sink"),
                                Some(ModelEdit::SetParam {
                                    sheet: nl.components[r].sheet.clone(),
                                    component: cx.local(&nl.components[r]),
                                    key: "resistance".into(),
                                    value: Param::quantity(
                                        Quantity::new(5.1e3, Dim::RESISTANCE),
                                        Tolerance::pm_percent(1.0),
                                    ),
                                }),
                            ));
                        }
                    }
                }
            }
        }
        let dp = c.pins.iter().find(|p| p.pin == "DP").and_then(|p| p.net);
        let dn = c.pins.iter().find(|p| p.pin == "DN").and_then(|p| p.net);
        match (dp, dn) {
            (Some(p), Some(n)) => {
                let paired = nl
                    .diff_pairs
                    .iter()
                    .any(|d| (d.p == p && d.n == n) || (d.p == n && d.n == p));
                if !paired {
                    out.push(add(
                        "usb-c",
                        "data-pair",
                        Severity::Warning,
                        vec![nl.nets[p].name.clone(), nl.nets[n].name.clone()],
                        "USB D+/D− are not declared as a differential pair".into(),
                        None,
                    ));
                }
                for x in [p, n] {
                    if nl.nets[x].pins.len() < 2 {
                        out.push(add(
                            "usb-c",
                            "data-connected",
                            Severity::Error,
                            vec![nl.nets[x].name.clone()],
                            format!("USB data net {} goes nowhere", nl.nets[x].name),
                            None,
                        ));
                    }
                }
            }
            _ => out.push(add(
                "usb-c",
                "data-connected",
                Severity::Error,
                vec![c.designator.clone()],
                "USB D+/D− are not connected".into(),
                None,
            )),
        }
    }
    out
}

pub fn switcher(nl: &Netlist, pool: &Pool, o: &ReviewOptions) -> Vec<ReviewFinding> {
    let cx = Ctx::new(nl, pool);
    let mut out = Vec::new();
    for (ci, c) in nl.components.iter().enumerate().filter(|(_, c)| c.fitted()) {
        let pin = |n: &str| c.pins.iter().find(|p| p.pin == n);
        let (Some(sw), Some(fb)) = (pin("SW"), pin("FB")) else {
            continue;
        };
        let sw_n = sw.net;
        let inductor = sw_n.and_then(|n| {
            cx.two
                .iter()
                .find(|(i, (a, b))| cx.is_kind(**i, "L") && (*a == Some(n) || *b == Some(n)))
                .map(|(i, (a, b))| (*i, if *a == Some(n) { *b } else { *a }))
        });
        let Some((_l, vout)) = inductor else {
            out.push(add(
                "switcher",
                "inductor",
                Severity::Error,
                vec![format!("{}.SW", c.designator)],
                format!("{} SW has no inductor", c.designator),
                None,
            ));
            continue;
        };
        if let Some(vin) = pin("VIN").and_then(|p| p.net) {
            if cx.between("C", vin, |m| cx.ground(m)).is_empty() {
                out.push(add(
                    "switcher",
                    "input-cap",
                    Severity::Error,
                    vec![format!("{}.VIN", c.designator)],
                    format!("{} input has no capacitor to ground", c.designator),
                    Some(ModelEdit::AddComponent {
                        sheet: c.sheet.clone(),
                        decl: two_pin("C0805-10U", "C?"),
                        connect: vec![
                            ("1".into(), Endpoint::pin(&cx.local(c), "VIN")),
                            ("2".into(), cx.ground_anchor(c)),
                        ],
                    }),
                ));
            }
        }
        let Some(vout) = vout else { continue };
        if cx.between("C", vout, |m| cx.ground(m)).is_empty() {
            out.push(add(
                "switcher",
                "output-cap",
                Severity::Error,
                vec![nl.nets[vout].name.clone()],
                format!("{} output has no capacitor to ground", c.designator),
                None,
            ));
        }
        let Some(fbn) = fb.net else {
            out.push(add(
                "switcher",
                "feedback",
                Severity::Error,
                vec![format!("{}.FB", c.designator)],
                "feedback pin is unconnected".into(),
                None,
            ));
            continue;
        };
        let top = cx.between("R", fbn, |m| m == vout);
        let bot = cx.between("R", fbn, |m| cx.ground(m));
        match (top.first(), bot.first()) {
            (Some(&t), Some(&b)) => {
                let (Some(rt), Some(rb)) = (cx.value(t, "resistance"), cx.value(b, "resistance"))
                else {
                    continue;
                };
                let vref = cx.value(ci, "vref").unwrap_or(o.default_vref);
                let v = vref * (1.0 + rt / rb);
                // The rail's nominal voltage comes from the net names on the output.
                let nominal = std::iter::once(nl.nets[vout].name.as_str())
                    .chain(nl.nets[vout].aliases.iter().map(|s| s.as_str()))
                    .find_map(rail_voltage);
                if let Some(nom) = nominal {
                    if (v - nom).abs() > 0.03 * nom {
                        let want_rt = rb * (nom / vref - 1.0);
                        out.push(add(
                            "switcher",
                            "output-voltage",
                            Severity::Warning,
                            vec![c.designator.clone(), nl.components[t].designator.clone(), nl.components[b].designator.clone()],
                            format!("divider sets {v:.3} V on a {nom} V rail; top resistor should be {want_rt:.0} Ω"),
                            Some(ModelEdit::SetParam {
                                sheet: nl.components[t].sheet.clone(),
                                component: cx.local(&nl.components[t]),
                                key: "resistance".into(),
                                value: Param::quantity(Quantity::new(want_rt, Dim::RESISTANCE), Tolerance::pm_percent(1.0)),
                            }),
                        ));
                    }
                }
            }
            _ => out.push(add(
                "switcher",
                "feedback",
                Severity::Error,
                vec![format!("{}.FB", c.designator)],
                format!(
                    "{} feedback has no divider between the output and ground",
                    c.designator
                ),
                None,
            )),
        }
    }
    out
}

/// Every suite, in a fixed order.
pub fn run_all(nl: &Netlist, pool: &Pool, o: &ReviewOptions) -> Vec<ReviewFinding> {
    let mut v = Vec::new();
    v.extend(decoupling(nl, pool, o));
    v.extend(i2c(nl, pool, o));
    v.extend(crystal(nl, pool, o));
    v.extend(straps(nl, pool, o));
    v.extend(usb_c(nl, pool, o));
    v.extend(switcher(nl, pool, o));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rail_names() {
        assert_eq!(rail_voltage("+3V3"), Some(3.3));
        assert_eq!(rail_voltage("/pwr/VOUT_1V8"), Some(1.8));
        assert_eq!(rail_voltage("+5V"), Some(5.0));
        assert_eq!(rail_voltage("SDA"), None);
        assert!(is_ground("GND") && is_ground("/a/GND_A") && !is_ground("SIG"));
    }
}
