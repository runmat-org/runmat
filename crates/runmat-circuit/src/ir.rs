//! The authored circuit model: what a `.circuit` source (or a KiCad import, or a runtime
//! script) says, before hierarchy is flattened.
//!
//! # Front-end seam
//!
//! The `.circuit` text dialect is parsed later on the shared lossless syntax tree (M1). It
//! feeds this module through [`DesignBuilder`] and plain struct construction, carrying a
//! [`SourceKey`] on every declaration so diagnostics and explain point back at text. Nothing
//! here knows about syntax.
//!
//! # Shape
//!
//! A [`Design`] holds sheet definitions ([`SheetDef`], the reusable modules), the root sheet,
//! design-wide net classes, interface (harness) types and variants. A sheet declares ports,
//! components, nets, buses, harnesses, sub-sheet instances (optionally repeated as channels),
//! connections, differential pairs, no-ERC markers and parameters. [`crate::elab`] flattens it.

use crate::ids::NamePath;
use runmat_rules::{Quantity, SourceKey};
use std::collections::BTreeMap;

/// A parameter value: a quantity with a tolerance, text, or a flag.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Quantity { nominal: Quantity, tol: Tolerance },
    Text(String),
    Bool(bool),
}

/// Tolerance band. `Relative(0.01)` is ±1 %; `Absolute` bounds are in SI units around the
/// nominal; `Range` is an explicit `[min, max]`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Tolerance {
    #[default]
    Exact,
    Relative {
        minus: f64,
        plus: f64,
    },
    Absolute {
        minus: f64,
        plus: f64,
    },
    Range {
        min: f64,
        max: f64,
    },
}

impl Tolerance {
    pub fn pm_percent(p: f64) -> Self {
        Tolerance::Relative {
            minus: p / 100.0,
            plus: p / 100.0,
        }
    }
    /// `[min, max]` around `nominal`.
    pub fn bounds(&self, nominal: f64) -> (f64, f64) {
        match *self {
            Tolerance::Exact => (nominal, nominal),
            Tolerance::Relative { minus, plus } => {
                let a = nominal * (1.0 - minus);
                let b = nominal * (1.0 + plus);
                (a.min(b), a.max(b))
            }
            Tolerance::Absolute { minus, plus } => (nominal - minus, nominal + plus),
            Tolerance::Range { min, max } => (min, max),
        }
    }
    /// Whether this band lies within `other` around the same nominal.
    pub fn within(&self, nominal: f64, other: &Tolerance) -> bool {
        let (a, b) = self.bounds(nominal);
        let (c, d) = other.bounds(nominal);
        a >= c - 1e-15 * c.abs() && b <= d + 1e-15 * d.abs()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub value: ParamValue,
    pub source: SourceKey,
}

impl Param {
    pub fn quantity(q: Quantity, tol: Tolerance) -> Self {
        Param {
            value: ParamValue::Quantity { nominal: q, tol },
            source: SourceKey::default(),
        }
    }
    pub fn text(s: impl Into<String>) -> Self {
        Param {
            value: ParamValue::Text(s.into()),
            source: SourceKey::default(),
        }
    }
    pub fn as_quantity(&self) -> Option<Quantity> {
        match &self.value {
            ParamValue::Quantity { nominal, .. } => Some(*nominal),
            _ => None,
        }
    }
    pub fn as_text(&self) -> Option<&str> {
        match &self.value {
            ParamValue::Text(s) => Some(s),
            _ => None,
        }
    }
}

/// Which library item a component instantiates.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PartRef {
    /// An orderable part from the pool, by name (MPN) or UUID text.
    Part(String),
    /// A logical entity with a package chosen (or not yet chosen); the part is selected later
    /// from parameters.
    Entity {
        entity: String,
        package: Option<String>,
    },
}

/// Fitting of a component in a variant.
#[derive(Clone, Debug, PartialEq)]
pub enum Fitting {
    Fitted,
    NotFitted,
    /// Fitted with a different part and/or parameter overrides.
    Alternate {
        part: Option<PartRef>,
        params: BTreeMap<String, Param>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentDecl {
    /// Local name. A designator (`U1`) or a prefix with `?` (`R?`) for auto-annotation.
    pub name: String,
    pub part: PartRef,
    pub params: BTreeMap<String, Param>,
    /// Free-form typed properties (`[properties]` schema fields).
    pub properties: BTreeMap<String, String>,
    pub classes: Vec<String>,
    /// Fitting overrides by variant name.
    pub variants: BTreeMap<String, Fitting>,
    /// Authored dissipation for the thermal power map (W), when known.
    pub power: Option<Quantity>,
    pub renamed_from: Option<String>,
    pub source: SourceKey,
}

impl ComponentDecl {
    pub fn new(name: &str, part: PartRef) -> Self {
        ComponentDecl {
            name: name.to_string(),
            part,
            params: BTreeMap::new(),
            properties: BTreeMap::new(),
            classes: Vec::new(),
            variants: BTreeMap::new(),
            power: None,
            renamed_from: None,
            source: SourceKey::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PortDir {
    In,
    Out,
    InOut,
    Passive,
    Power,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortKind {
    Signal,
    /// A bus port carrying these member names (`D0`…`D7`).
    Bus(Vec<String>),
    /// A harness port of this interface type.
    Interface(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortDecl {
    pub name: String,
    pub dir: PortDir,
    pub kind: PortKind,
    pub source: SourceKey,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetDecl {
    pub name: String,
    pub class: Option<String>,
    /// Global nets (power rails, `GND`) join by name across the whole design.
    pub global: bool,
    pub params: BTreeMap<String, Param>,
    pub renamed_from: Option<String>,
    pub source: SourceKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusDecl {
    pub name: String,
    pub members: Vec<String>,
    pub source: SourceKey,
}

/// An interface (harness) type: named members, each a signal or a nested interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterfaceDef {
    pub name: String,
    pub members: Vec<String>,
    /// Members that must be a differential pair: `(p, n)`.
    pub diff_pairs: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessDecl {
    pub name: String,
    pub interface: String,
    pub source: SourceKey,
}

/// One end of a connection.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Endpoint {
    /// Component pin by pin name, gate-qualified name (`A.IN+`) or pad number.
    Pin {
        component: String,
        pin: String,
    },
    Net(String),
    /// A port of the sheet being defined.
    Port(String),
    /// A port of a sub-sheet instance (all channels unless `channel` is given).
    InstancePort {
        instance: String,
        port: String,
        channel: Option<u32>,
    },
    BusMember {
        bus: String,
        member: String,
    },
    HarnessMember {
        harness: String,
        member: String,
    },
}

impl Endpoint {
    pub fn pin(c: &str, p: &str) -> Self {
        Endpoint::Pin {
            component: c.into(),
            pin: p.into(),
        }
    }
    pub fn net(n: &str) -> Self {
        Endpoint::Net(n.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connection {
    pub ends: Vec<Endpoint>,
    pub source: SourceKey,
}

/// How a port of a repeated instance binds in the parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// Every channel binds to the same endpoint (shared rails, shared buses).
    Shared(Endpoint),
    /// Channel `i` binds to member `offset + i` of a bus.
    BusIndexed { bus: String, offset: u32 },
    /// Channel `i` binds to net `prefix{i + offset}`.
    NetIndexed { prefix: String, offset: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SheetInstance {
    pub name: String,
    pub sheet: String,
    /// `Some(n)` repeats the sheet as `n` channels named `name[0]`…`name[n-1]`.
    pub repeat: Option<u32>,
    pub bindings: Vec<(String, Binding)>,
    pub renamed_from: Option<String>,
    pub source: SourceKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffPairDecl {
    pub name: String,
    pub p: String,
    pub n: String,
    pub class: Option<String>,
    pub source: SourceKey,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SheetDef {
    pub name: String,
    pub ports: Vec<PortDecl>,
    pub components: Vec<ComponentDecl>,
    pub nets: Vec<NetDecl>,
    pub buses: Vec<BusDecl>,
    pub harnesses: Vec<HarnessDecl>,
    pub instances: Vec<SheetInstance>,
    pub connections: Vec<Connection>,
    pub diff_pairs: Vec<DiffPairDecl>,
    /// Pins marked "no connect" / no-ERC.
    pub no_erc: Vec<Endpoint>,
    pub params: BTreeMap<String, Param>,
}

/// Net class: electrical and board-rule defaults for a group of nets.
#[derive(Clone, Debug, PartialEq)]
pub struct NetClass {
    pub name: String,
    pub params: BTreeMap<String, Param>,
    pub source: SourceKey,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantDef {
    pub name: String,
    /// Variant this one inherits from.
    pub base: Option<String>,
    /// Fitting overrides by component path (`/power/U3`) or designator.
    pub overrides: BTreeMap<String, Fitting>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Design {
    pub name: String,
    pub sheets: BTreeMap<String, SheetDef>,
    pub root: String,
    pub net_classes: BTreeMap<String, NetClass>,
    pub interfaces: BTreeMap<String, InterfaceDef>,
    pub variants: BTreeMap<String, VariantDef>,
    pub params: BTreeMap<String, Param>,
}

impl Design {
    pub fn new(name: &str) -> Self {
        Design {
            name: name.to_string(),
            root: "main".to_string(),
            ..Default::default()
        }
    }
    pub fn sheet_mut(&mut self, name: &str) -> &mut SheetDef {
        self.sheets
            .entry(name.to_string())
            .or_insert_with(|| SheetDef {
                name: name.to_string(),
                ..Default::default()
            })
    }
}

/// Fluent construction used by front ends and tests.
pub struct DesignBuilder<'a> {
    pub design: &'a mut Design,
    pub sheet: String,
}

impl<'a> DesignBuilder<'a> {
    pub fn new(design: &'a mut Design, sheet: &str) -> Self {
        design.sheet_mut(sheet);
        DesignBuilder {
            design,
            sheet: sheet.to_string(),
        }
    }
    fn s(&mut self) -> &mut SheetDef {
        self.design.sheets.get_mut(&self.sheet).unwrap()
    }
    pub fn component(&mut self, name: &str, part: &str) -> &mut ComponentDecl {
        self.s()
            .components
            .push(ComponentDecl::new(name, PartRef::Part(part.into())));
        self.s().components.last_mut().unwrap()
    }
    pub fn net(&mut self, name: &str) -> &mut NetDecl {
        self.s().nets.push(NetDecl {
            name: name.into(),
            class: None,
            global: false,
            params: BTreeMap::new(),
            renamed_from: None,
            source: SourceKey::default(),
        });
        self.s().nets.last_mut().unwrap()
    }
    pub fn global(&mut self, name: &str, class: Option<&str>) {
        let n = self.net(name);
        n.global = true;
        n.class = class.map(|c| c.to_string());
    }
    pub fn port(&mut self, name: &str, dir: PortDir) {
        self.s().ports.push(PortDecl {
            name: name.into(),
            dir,
            kind: PortKind::Signal,
            source: SourceKey::default(),
        });
    }
    pub fn bus_port(&mut self, name: &str, dir: PortDir, members: &[&str]) {
        self.s().ports.push(PortDecl {
            name: name.into(),
            dir,
            kind: PortKind::Bus(members.iter().map(|m| m.to_string()).collect()),
            source: SourceKey::default(),
        });
    }
    pub fn bus(&mut self, name: &str, members: &[&str]) {
        self.s().buses.push(BusDecl {
            name: name.into(),
            members: members.iter().map(|m| m.to_string()).collect(),
            source: SourceKey::default(),
        });
    }
    pub fn connect(&mut self, ends: Vec<Endpoint>) {
        self.s().connections.push(Connection {
            ends,
            source: SourceKey::default(),
        });
    }
    /// `connect_pins("NET", &[("U1","VDD"), ("C1","1")])`.
    pub fn wire(&mut self, net: &str, pins: &[(&str, &str)]) {
        let mut ends = vec![Endpoint::net(net)];
        ends.extend(pins.iter().map(|(c, p)| Endpoint::pin(c, p)));
        self.connect(ends);
    }
    pub fn instance(&mut self, name: &str, sheet: &str, repeat: Option<u32>) -> &mut SheetInstance {
        self.s().instances.push(SheetInstance {
            name: name.into(),
            sheet: sheet.into(),
            repeat,
            bindings: Vec::new(),
            renamed_from: None,
            source: SourceKey::default(),
        });
        self.s().instances.last_mut().unwrap()
    }
    pub fn diff_pair(&mut self, name: &str, p: &str, n: &str, class: Option<&str>) {
        self.s().diff_pairs.push(DiffPairDecl {
            name: name.into(),
            p: p.into(),
            n: n.into(),
            class: class.map(|c| c.to_string()),
            source: SourceKey::default(),
        });
    }
    pub fn no_erc(&mut self, c: &str, p: &str) {
        self.s().no_erc.push(Endpoint::pin(c, p));
    }
}

/// Expands `D[0..7]` / `D[7:0]` style bus notation into member names; other names pass through.
pub fn expand_bus(name: &str) -> Vec<String> {
    if let (Some(o), Some(c)) = (name.find('['), name.rfind(']')) {
        let base = &name[..o];
        let inner = &name[o + 1..c];
        let sep = if inner.contains("..") { ".." } else { ":" };
        if let Some((a, b)) = inner.split_once(sep) {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<i64>(), b.trim().parse::<i64>()) {
                let range: Vec<i64> = if a <= b {
                    (a..=b).collect()
                } else {
                    (b..=a).rev().collect()
                };
                return range.into_iter().map(|i| format!("{base}{i}")).collect();
            }
        }
    }
    vec![name.to_string()]
}

/// Item path helper: sheet-instance path plus a local name.
pub fn item_path(inst: &NamePath, local: &str) -> NamePath {
    inst.child(local)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_notation() {
        assert_eq!(expand_bus("D[0..3]"), vec!["D0", "D1", "D2", "D3"]);
        assert_eq!(expand_bus("A[2:0]"), vec!["A2", "A1", "A0"]);
        assert_eq!(expand_bus("SDA"), vec!["SDA"]);
    }

    #[test]
    fn tolerance_bounds() {
        let t = Tolerance::pm_percent(1.0);
        let (a, b) = t.bounds(10e3);
        assert!((a - 9900.0).abs() < 1e-9 && (b - 10100.0).abs() < 1e-9);
        assert!(t.within(10e3, &Tolerance::pm_percent(5.0)));
        assert!(!Tolerance::pm_percent(5.0).within(10e3, &t));
    }
}
