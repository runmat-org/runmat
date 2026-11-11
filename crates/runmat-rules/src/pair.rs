//! Pair rules: values that depend on two items, such as the clearance between two copper
//! objects or the gap of a differential pair.
//!
//! A [`PairRule`] has two selectors `a` and `b` and applies to the unordered pair `{x, y}`
//! when `a` matches one and `b` matches the other. Its specificity for the pair is the sum of
//! the two sides' specificities (taking the better of the two assignments when both work).
//! Precedence is then the same as for single-item rules: `(specificity, priority, source
//! order)`, highest wins; see [`crate::cascade`].
//!
//! Resolution only depends on which rule sides each item matches, so results are memoised
//! per pair of *match signatures*: a board with 50 000 copper items and 30 clearance rules
//! typically has a few dozen distinct signatures, so DRC queries are map lookups.

use crate::cascade::{LostTo, RuleHandle, SourceKey, SourceOrder};
use crate::graph::ItemGraph;
use crate::selector::{Selector, Specificity};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct PairRule<V> {
    pub id: String,
    pub a: Selector,
    pub b: Selector,
    pub priority: i32,
    pub source: SourceKey,
    pub order: SourceOrder,
    pub decls: Vec<(String, V)>,
}

impl<V> PairRule<V> {
    pub fn new(id: impl Into<String>, a: Selector, b: Selector) -> Self {
        PairRule {
            id: id.into(),
            a,
            b,
            priority: 0,
            source: SourceKey::default(),
            order: SourceOrder::default(),
            decls: Vec::new(),
        }
    }
    pub fn priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }
    pub fn at(mut self, source: SourceKey, order: SourceOrder) -> Self {
        self.source = source;
        self.order = order;
        self
    }
    pub fn set(mut self, property: impl Into<String>, value: V) -> Self {
        self.decls.push((property.into(), value));
        self
    }
    fn declared(&self, property: &str) -> Option<&V> {
        self.decls
            .iter()
            .rev()
            .find(|(p, _)| p == property)
            .map(|(_, v)| v)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairResolved<V> {
    pub value: V,
    /// `None` when the value is the property default.
    pub rule: Option<PairWinner>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairWinner {
    pub rule_id: String,
    pub handle: RuleHandle,
    pub source: SourceKey,
    pub specificity: Specificity,
    pub priority: i32,
    pub order: SourceOrder,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairCandidate<V> {
    pub winner: PairWinner,
    pub value: V,
    pub lost_to: LostTo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairExplanation<V> {
    pub property: String,
    pub effective: Option<PairResolved<V>>,
    pub overridden: Vec<PairCandidate<V>>,
}

impl<V: fmt::Debug> fmt::Display for PairExplanation<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.effective {
            Some(PairResolved {
                value,
                rule: Some(w),
            }) => writeln!(
                f,
                "{} = {:?} ← rule `{}` at {} (specificity {}, priority {})",
                self.property, value, w.rule_id, w.source, w.specificity, w.priority
            )?,
            Some(PairResolved { value, rule: None }) => {
                writeln!(f, "{} = {:?} ← default", self.property, value)?
            }
            None => writeln!(f, "{} has no value", self.property)?,
        }
        for c in &self.overridden {
            writeln!(
                f,
                "  overridden: {:?} from rule `{}` at {} ({:?})",
                c.value, c.winner.rule_id, c.winner.source, c.lost_to
            )?;
        }
        Ok(())
    }
}

type Sig = Vec<(RuleHandle, u8, Specificity)>;
/// Memo keyed by `(lower signature, higher signature, property id)`: no allocation per lookup.
type Memo<V> = BTreeMap<(u32, u32, u32), Option<PairResolved<V>>>;

/// The rule-side matches of an item outside the rebuilt set (see [`PairCascade::probe`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    sig: Sig,
    class: Option<u32>,
}

/// An interned property name: resolve with it to skip even the name lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PropId(pub u32);

/// Pair-rule store with per-side match sets and a signature memo.
#[derive(Debug)]
pub struct PairCascade<V, Id: Ord> {
    rules: BTreeMap<RuleHandle, PairRule<V>>,
    next: u32,
    defaults: BTreeMap<String, V>,
    item_sig: BTreeMap<Id, u32>,
    sigs: BTreeMap<Sig, u32>,
    sig_list: Vec<Sig>,
    /// Interned property names (ids are dense, in first-seen order).
    props: RefCell<BTreeMap<String, u32>>,
    memo: RefCell<Memo<V>>,
}

impl<V: Clone, Id: Copy + Ord + fmt::Debug> Default for PairCascade<V, Id> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone, Id: Copy + Ord + fmt::Debug> PairCascade<V, Id> {
    pub fn new() -> Self {
        let mut s = PairCascade {
            rules: BTreeMap::new(),
            next: 0,
            defaults: BTreeMap::new(),
            item_sig: BTreeMap::new(),
            sigs: BTreeMap::new(),
            sig_list: Vec::new(),
            props: RefCell::new(BTreeMap::new()),
            memo: RefCell::new(BTreeMap::new()),
        };
        s.intern(Vec::new());
        s
    }

    fn intern(&mut self, sig: Sig) -> u32 {
        if let Some(i) = self.sigs.get(&sig) {
            return *i;
        }
        let i = self.sig_list.len() as u32;
        self.sigs.insert(sig.clone(), i);
        self.sig_list.push(sig);
        i
    }

    pub fn set_default(&mut self, property: &str, value: V) {
        self.defaults.insert(property.to_string(), value);
        self.memo.borrow_mut().clear();
    }

    pub fn push_rule(&mut self, rule: PairRule<V>) -> RuleHandle {
        let h = RuleHandle(self.next);
        self.next += 1;
        self.rules.insert(h, rule);
        self.memo.borrow_mut().clear();
        h
    }
    /// Removing or replacing a rule changes which items match it: call
    /// [`PairCascade::rebuild`] afterwards.
    pub fn remove_rule(&mut self, h: RuleHandle) -> Option<PairRule<V>> {
        self.memo.borrow_mut().clear();
        self.rules.remove(&h)
    }
    pub fn replace_rule(&mut self, h: RuleHandle, rule: PairRule<V>) {
        self.memo.borrow_mut().clear();
        self.rules.insert(h, rule);
    }
    pub fn rules(&self) -> impl Iterator<Item = (RuleHandle, &PairRule<V>)> {
        self.rules.iter().map(|(h, r)| (*h, r))
    }

    fn signature<G: ItemGraph<Id = Id>>(&self, g: &G, id: Id) -> Sig {
        let mut sig = Vec::new();
        for (h, r) in &self.rules {
            if let Some(s) = r.a.matches(g, id) {
                sig.push((*h, 0u8, s));
            }
            if let Some(s) = r.b.matches(g, id) {
                sig.push((*h, 1u8, s));
            }
        }
        sig
    }

    /// Matches every rule side against every item.
    pub fn rebuild<G: ItemGraph<Id = Id>>(&mut self, g: &G) {
        self.item_sig.clear();
        self.sigs.clear();
        self.sig_list.clear();
        self.intern(Vec::new());
        let mut items = g.items();
        items.sort();
        items.dedup();
        for id in items {
            let s = self.signature(g, id);
            let k = self.intern(s);
            if k != 0 {
                self.item_sig.insert(id, k);
            }
        }
        self.memo.borrow_mut().clear();
    }

    /// Re-matches changed items; drops removed ones. Returns items whose signature changed.
    pub fn items_changed<G: ItemGraph<Id = Id>>(
        &mut self,
        g: &G,
        changed: &[Id],
        removed: &[Id],
    ) -> BTreeSet<Id> {
        let mut out = BTreeSet::new();
        for r in removed {
            if self.item_sig.remove(r).is_some() {
                out.insert(*r);
            }
        }
        for &c in changed {
            let s = self.signature(g, c);
            let k = self.intern(s);
            let before = self.item_sig.get(&c).copied().unwrap_or(0);
            if k != before {
                out.insert(c);
                if k == 0 {
                    self.item_sig.remove(&c);
                } else {
                    self.item_sig.insert(c, k);
                }
            }
        }
        out
    }

    /// The signature class of an item (equal classes resolve identically against any other).
    pub fn class_of(&self, id: Id) -> u32 {
        self.item_sig.get(&id).copied().unwrap_or(0)
    }

    fn candidates(&self, sx: u32, sy: u32, property: &str) -> Vec<(PairWinner, V)> {
        self.candidates_sig(
            &self.sig_list[sx as usize],
            &self.sig_list[sy as usize],
            property,
        )
    }

    fn candidates_sig(&self, gx: &Sig, gy: &Sig, property: &str) -> Vec<(PairWinner, V)> {
        // Best specificity per rule over both assignments (a→x,b→y) and (a→y,b→x).
        let mut best: BTreeMap<RuleHandle, Specificity> = BTreeMap::new();
        let side = |sig: &Sig, h: RuleHandle, s: u8| {
            sig.iter()
                .find(|(hh, ss, _)| *hh == h && *ss == s)
                .map(|t| t.2)
        };
        for (h, r) in &self.rules {
            if r.declared(property).is_none() {
                continue;
            }
            let mut b: Option<Specificity> = None;
            if let (Some(a), Some(c)) = (side(gx, *h, 0), side(gy, *h, 1)) {
                b = Some(a + c);
            }
            if let (Some(a), Some(c)) = (side(gy, *h, 0), side(gx, *h, 1)) {
                let s = a + c;
                b = Some(b.map_or(s, |x| x.max(s)));
            }
            if let Some(s) = b {
                best.insert(*h, s);
            }
        }
        let mut out: Vec<(PairWinner, V)> = best
            .into_iter()
            .map(|(h, s)| {
                let r = &self.rules[&h];
                (
                    PairWinner {
                        rule_id: r.id.clone(),
                        handle: h,
                        source: r.source.clone(),
                        specificity: s,
                        priority: r.priority,
                        order: r.order,
                    },
                    r.declared(property).unwrap().clone(),
                )
            })
            .collect();
        out.sort_by(|a, b| {
            let ka = (a.0.specificity, a.0.priority, a.0.order, a.0.handle);
            let kb = (b.0.specificity, b.0.priority, b.0.order, b.0.handle);
            kb.cmp(&ka)
        });
        out
    }

    /// The id of a property name, interning it on first use.
    pub fn prop_id(&self, property: &str) -> PropId {
        if let Some(i) = self.props.borrow().get(property) {
            return PropId(*i);
        }
        let mut m = self.props.borrow_mut();
        let i = m.len() as u32;
        m.insert(property.to_string(), i);
        PropId(i)
    }

    fn prop_name(&self, id: PropId) -> String {
        self.props
            .borrow()
            .iter()
            .find(|(_, i)| **i == id.0)
            .map(|(k, _)| k.clone())
            .unwrap_or_default()
    }

    /// The match signature of an item that is not (yet) part of the rebuilt set, such as the
    /// track an interactive router is about to place: `g` must answer for `id`.
    pub fn probe<G: ItemGraph<Id = Id>>(&self, g: &G, id: Id) -> Probe {
        let sig = self.signature(g, id);
        Probe {
            class: self.sigs.get(&sig).copied(),
            sig,
        }
    }

    /// The effective value of `property` between a probed item and item `y`.
    pub fn resolve_probe(&self, p: &Probe, y: Id, property: &str) -> Option<PairResolved<V>> {
        if let Some(c) = p.class {
            let pid = self.prop_id(property);
            let (sx, sy) = (c.min(self.class_of(y)), c.max(self.class_of(y)));
            if let Some(v) = self.memo.borrow().get(&(sx, sy, pid.0)) {
                return v.clone();
            }
        }
        let sy = &self.sig_list[self.class_of(y) as usize];
        self.candidates_sig(&p.sig, sy, property)
            .into_iter()
            .next()
            .map(|(w, v)| PairResolved {
                value: v,
                rule: Some(w),
            })
            .or_else(|| {
                self.defaults.get(property).map(|v| PairResolved {
                    value: v.clone(),
                    rule: None,
                })
            })
    }

    /// The effective value of `property` for the pair `{x, y}`.
    pub fn resolve(&self, x: Id, y: Id, property: &str) -> Option<PairResolved<V>> {
        self.with_resolved(x, y, self.prop_id(property), |r| r.cloned())
    }

    /// Like [`PairCascade::resolve`] by property id, lending the memoised result to `f`
    /// instead of cloning it (the DRC hot path reads only the value).
    pub fn with_resolved<R>(
        &self,
        x: Id,
        y: Id,
        property: PropId,
        f: impl FnOnce(Option<&PairResolved<V>>) -> R,
    ) -> R {
        let (sx, sy) = {
            let a = self.class_of(x);
            let b = self.class_of(y);
            (a.min(b), a.max(b))
        };
        let key = (sx, sy, property.0);
        if let Some(v) = self.memo.borrow().get(&key) {
            return f(v.as_ref());
        }
        let name = self.prop_name(property);
        let r = self
            .candidates(sx, sy, &name)
            .into_iter()
            .next()
            .map(|(w, v)| PairResolved {
                value: v,
                rule: Some(w),
            })
            .or_else(|| {
                self.defaults.get(&name).map(|v| PairResolved {
                    value: v.clone(),
                    rule: None,
                })
            });
        let out = f(r.as_ref());
        self.memo.borrow_mut().insert(key, r);
        out
    }

    pub fn explain(&self, x: Id, y: Id, property: &str) -> PairExplanation<V> {
        let sx = self.class_of(x);
        let sy = self.class_of(y);
        let c = self.candidates(sx.min(sy), sx.max(sy), property);
        let effective = self.resolve(x, y, property);
        let mut overridden = Vec::new();
        if let Some((w, _)) = c.first() {
            for (cw, v) in c.iter().skip(1) {
                let lost_to = if cw.specificity < w.specificity {
                    LostTo::LowerSpecificity
                } else if cw.priority < w.priority {
                    LostTo::LowerPriority
                } else {
                    LostTo::EarlierSource
                };
                overridden.push(PairCandidate {
                    winner: cw.clone(),
                    value: v.clone(),
                    lost_to,
                });
            }
        }
        PairExplanation {
            property: property.to_string(),
            effective,
            overridden,
        }
    }

    /// The largest value any rule (or the default) gives `property`, under `key`; DRC uses it
    /// as the spatial query radius.
    pub fn max_value_by<K: PartialOrd + Copy>(
        &self,
        property: &str,
        key: impl Fn(&V) -> K,
    ) -> Option<K> {
        let mut best: Option<K> = self.defaults.get(property).map(&key);
        for r in self.rules.values() {
            if let Some(v) = r.declared(property) {
                let k = key(v);
                if best.is_none_or(|b| k > b) {
                    best = Some(k);
                }
            }
        }
        best
    }
}
