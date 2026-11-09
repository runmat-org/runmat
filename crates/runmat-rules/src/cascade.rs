//! Rules, the cascade and explain.
//!
//! # Precedence (the stated tie-break)
//!
//! For one item and one property the effective value is chosen in this order:
//!
//! 1. **Inline value.** A value authored on the item itself ([`Cascade::set_inline`]) beats
//!    every selector rule ("object over selector").
//! 2. **Selector rules.** Among rules whose selector matches the item and that declare the
//!    property, the winner has the greatest key `(specificity, priority, source order)`,
//!    compared left to right:
//!    - **specificity** of the selector for this item (see [`crate::selector`]);
//!    - then the rule's explicit **priority** (higher wins; default 0);
//!    - then **source order**: the later rule wins. Source order is
//!      [`SourceOrder`]`(file rank, byte offset)`; hosts give files their rank in load order
//!      (packages first, then the project, then the open file), so a project file overrides
//!      a package at equal specificity and priority.
//!
//!    Within one rule, a later declaration of the same property replaces an earlier one.
//!    Two different rules can never tie: rule handles are unique and the order key is
//!    completed by the handle, so the result never depends on map or insertion order.
//! 3. **Inherited value.** For a property registered as inherited, the nearest ancestor's
//!    effective value ("direct over inherited").
//! 4. **Default.** The property's registered default, if any.
//!
//! # Incremental re-evaluation
//!
//! The cascade keeps, per rule, the items it matches and, per item, the rules matching it.
//! [`Cascade::items_changed`] re-matches only the changed items, their descendants (for
//! descendant and child combinators) and, when any selector uses `:has`, their ancestors; it
//! returns the items whose effective values may have changed. Rule edits re-match one rule
//! over the items of the kinds it can match. Results equal a full [`Cascade::rebuild`].

use crate::graph::ItemGraph;
use crate::selector::{Selector, Specificity};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Where a rule or declaration came from: a file and a byte span. Hosts map spans to lines.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceKey {
    pub file: String,
    pub start: u32,
    pub end: u32,
}

impl SourceKey {
    pub fn new(file: impl Into<String>, start: u32, end: u32) -> Self {
        SourceKey {
            file: file.into(),
            start,
            end,
        }
    }
}

impl fmt::Display for SourceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}..{}", self.file, self.start, self.end)
    }
}

/// Position used by the last tie-break: file rank in load order, then byte offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceOrder {
    pub file_rank: u32,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuleHandle(pub u32);

#[derive(Clone, Debug, PartialEq)]
pub struct Declaration<V> {
    pub property: String,
    pub value: V,
    pub source: SourceKey,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rule<V> {
    pub id: String,
    pub selector: Selector,
    pub priority: i32,
    pub source: SourceKey,
    pub order: SourceOrder,
    pub decls: Vec<Declaration<V>>,
}

impl<V> Rule<V> {
    pub fn new(id: impl Into<String>, selector: Selector) -> Self {
        Rule {
            id: id.into(),
            selector,
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
        let source = self.source.clone();
        self.decls.push(Declaration {
            property: property.into(),
            value,
            source,
        });
        self
    }
    /// The last declaration of `property` in this rule.
    pub fn declared(&self, property: &str) -> Option<&Declaration<V>> {
        self.decls.iter().rev().find(|d| d.property == property)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PropertySpec<V> {
    pub inherited: bool,
    pub default: Option<V>,
}

/// Why a value is the effective one.
#[derive(Clone, Debug, PartialEq)]
pub enum Origin<Id> {
    Inline {
        source: SourceKey,
    },
    Rule {
        rule_id: String,
        handle: RuleHandle,
        source: SourceKey,
        specificity: Specificity,
        priority: i32,
        order: SourceOrder,
    },
    Inherited {
        from: Id,
        origin: Box<Origin<Id>>,
    },
    Default,
}

impl<Id: fmt::Debug> fmt::Display for Origin<Id> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Origin::Inline { source } => write!(f, "set on the item at {source}"),
            Origin::Rule {
                rule_id,
                source,
                specificity,
                priority,
                ..
            } => write!(
                f,
                "rule `{rule_id}` at {source} (specificity {specificity}, priority {priority})"
            ),
            Origin::Inherited { from, origin } => write!(f, "inherited from {from:?}: {origin}"),
            Origin::Default => f.write_str("default"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Resolved<V, Id> {
    pub value: V,
    pub origin: Origin<Id>,
}

/// Why a candidate lost to the winner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LostTo {
    /// The item has an inline value.
    Inline,
    LowerSpecificity,
    /// Same specificity, lower priority.
    LowerPriority,
    /// Same specificity and priority, earlier in source order.
    EarlierSource,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate<V> {
    pub rule_id: String,
    pub handle: RuleHandle,
    pub source: SourceKey,
    pub specificity: Specificity,
    pub priority: i32,
    pub order: SourceOrder,
    pub value: V,
    pub lost_to: LostTo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Explanation<V, Id> {
    pub item: Id,
    pub property: String,
    pub effective: Option<Resolved<V, Id>>,
    /// Losing rule candidates, most to least preferred.
    pub overridden: Vec<Candidate<V>>,
}

impl<V: fmt::Debug, Id: fmt::Debug> fmt::Display for Explanation<V, Id> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.effective {
            Some(r) => writeln!(
                f,
                "{:?}.{} = {:?} ← {}",
                self.item, self.property, r.value, r.origin
            )?,
            None => writeln!(f, "{:?}.{} has no value", self.item, self.property)?,
        }
        for c in &self.overridden {
            writeln!(
                f,
                "  overridden: {:?} from rule `{}` at {} ({:?})",
                c.value, c.rule_id, c.source, c.lost_to
            )?;
        }
        Ok(())
    }
}

type PrecKey = (Specificity, i32, SourceOrder, RuleHandle);

/// The cascade over one item graph. `V` is the property value type.
#[derive(Clone, Debug)]
pub struct Cascade<V, Id: Ord> {
    rules: BTreeMap<RuleHandle, Rule<V>>,
    next: u32,
    props: BTreeMap<String, PropertySpec<V>>,
    inline: BTreeMap<(Id, String), (V, SourceKey)>,
    matched: BTreeMap<RuleHandle, BTreeMap<Id, Specificity>>,
    item_rules: BTreeMap<Id, BTreeMap<RuleHandle, Specificity>>,
    by_kind: BTreeMap<String, Vec<RuleHandle>>,
    universal: Vec<RuleHandle>,
    has_rules: usize,
}

impl<V: Clone, Id: Copy + Ord + fmt::Debug> Default for Cascade<V, Id> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone, Id: Copy + Ord + fmt::Debug> Cascade<V, Id> {
    pub fn new() -> Self {
        Cascade {
            rules: BTreeMap::new(),
            next: 0,
            props: BTreeMap::new(),
            inline: BTreeMap::new(),
            matched: BTreeMap::new(),
            item_rules: BTreeMap::new(),
            by_kind: BTreeMap::new(),
            universal: Vec::new(),
            has_rules: 0,
        }
    }

    pub fn register_property(&mut self, name: &str, inherited: bool, default: Option<V>) {
        self.props
            .insert(name.to_string(), PropertySpec { inherited, default });
    }

    pub fn rule(&self, h: RuleHandle) -> Option<&Rule<V>> {
        self.rules.get(&h)
    }
    pub fn rules(&self) -> impl Iterator<Item = (RuleHandle, &Rule<V>)> {
        self.rules.iter().map(|(h, r)| (*h, r))
    }
    pub fn handle_of(&self, rule_id: &str) -> Option<RuleHandle> {
        self.rules
            .iter()
            .find(|(_, r)| r.id == rule_id)
            .map(|(h, _)| *h)
    }

    /// Items matched by a rule, in id order.
    pub fn matches_of(&self, h: RuleHandle) -> Vec<(Id, Specificity)> {
        self.matched
            .get(&h)
            .map(|m| m.iter().map(|(i, s)| (*i, *s)).collect())
            .unwrap_or_default()
    }
    /// Rules matching an item, in handle order.
    pub fn rules_of(&self, id: Id) -> Vec<(RuleHandle, Specificity)> {
        self.item_rules
            .get(&id)
            .map(|m| m.iter().map(|(h, s)| (*h, *s)).collect())
            .unwrap_or_default()
    }

    fn reindex(&mut self) {
        self.by_kind.clear();
        self.universal.clear();
        self.has_rules = 0;
        for (h, r) in &self.rules {
            if r.selector.uses_has() {
                self.has_rules += 1;
            }
            match r.selector.required_kinds() {
                Some(ks) => {
                    for k in ks {
                        self.by_kind.entry(k).or_default().push(*h);
                    }
                }
                None => self.universal.push(*h),
            }
        }
    }

    fn candidates_for<G: ItemGraph<Id = Id>>(&self, g: &G, id: Id) -> Vec<RuleHandle> {
        let mut out: Vec<RuleHandle> = self.universal.clone();
        for (k, hs) in &self.by_kind {
            if g.is_kind(id, k) {
                out.extend(hs.iter().copied());
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Re-matches one item against every rule. Returns whether its matches changed.
    fn rematch_item<G: ItemGraph<Id = Id>>(&mut self, g: &G, id: Id) -> bool {
        let mut now = BTreeMap::new();
        for h in self.candidates_for(g, id) {
            if let Some(s) = self.rules[&h].selector.matches(g, id) {
                now.insert(h, s);
            }
        }
        let before = self.item_rules.remove(&id).unwrap_or_default();
        if before == now {
            if !now.is_empty() {
                self.item_rules.insert(id, now);
            }
            return false;
        }
        for h in before.keys() {
            if let Some(m) = self.matched.get_mut(h) {
                m.remove(&id);
            }
        }
        for (h, s) in &now {
            self.matched.entry(*h).or_default().insert(id, *s);
        }
        if !now.is_empty() {
            self.item_rules.insert(id, now);
        }
        true
    }

    fn forget_item(&mut self, id: Id) {
        if let Some(before) = self.item_rules.remove(&id) {
            for h in before.keys() {
                if let Some(m) = self.matched.get_mut(h) {
                    m.remove(&id);
                }
            }
        }
        let keys: Vec<_> = self
            .inline
            .range((id, String::new())..)
            .take_while(|((i, _), _)| *i == id)
            .map(|(k, _)| k.clone())
            .collect();
        for k in keys {
            self.inline.remove(&k);
        }
    }

    /// Matches every rule against every item from scratch.
    pub fn rebuild<G: ItemGraph<Id = Id>>(&mut self, g: &G) {
        self.reindex();
        self.matched.clear();
        self.item_rules.clear();
        let mut items = g.items();
        items.sort();
        items.dedup();
        for id in items {
            self.rematch_item(g, id);
        }
    }

    fn push_descendants<G: ItemGraph<Id = Id>>(g: &G, id: Id, out: &mut BTreeSet<Id>) {
        let mut stack = vec![id];
        while let Some(i) = stack.pop() {
            for c in g.children(i) {
                if out.insert(c) {
                    stack.push(c);
                }
            }
        }
    }

    /// Items were added or changed (attributes, classes, names, parents) or removed. Returns
    /// the items whose effective values may differ now, in id order.
    pub fn items_changed<G: ItemGraph<Id = Id>>(
        &mut self,
        g: &G,
        changed: &[Id],
        removed: &[Id],
    ) -> BTreeSet<Id> {
        let mut affected = BTreeSet::new();
        for &r in removed {
            self.forget_item(r);
        }
        let mut to_match = BTreeSet::new();
        for &c in changed {
            to_match.insert(c);
            Self::push_descendants(g, c, &mut to_match);
            if self.has_rules > 0 {
                let mut p = g.parent(c);
                while let Some(x) = p {
                    to_match.insert(x);
                    p = g.parent(x);
                }
            }
        }
        // A removed item's former ancestors may stop matching `:has`; its former descendants
        // are reported changed by the host.
        let mut changed_matches = BTreeSet::new();
        for id in to_match {
            if self.rematch_item(g, id) {
                changed_matches.insert(id);
            }
        }
        affected.extend(changed.iter().copied());
        affected.extend(changed_matches.iter().copied());
        // Inheritance: descendants of anything that changed may inherit a new value.
        if self.props.values().any(|p| p.inherited) {
            let roots: Vec<Id> = affected.iter().copied().collect();
            for r in roots {
                Self::push_descendants(g, r, &mut affected);
            }
        }
        affected
    }

    fn rule_affected<G: ItemGraph<Id = Id>>(&self, g: &G, items: &BTreeSet<Id>) -> BTreeSet<Id> {
        let mut out = items.clone();
        if self.props.values().any(|p| p.inherited) {
            for &i in items {
                Self::push_descendants(g, i, &mut out);
            }
        }
        out
    }

    fn match_rule_everywhere<G: ItemGraph<Id = Id>>(&mut self, g: &G, h: RuleHandle) {
        let mut items = g.items();
        items.sort();
        items.dedup();
        let kinds = self.rules[&h].selector.required_kinds();
        let mut m = BTreeMap::new();
        for id in items {
            if let Some(ks) = &kinds {
                if !ks.iter().any(|k| g.is_kind(id, k)) {
                    continue;
                }
            }
            if let Some(s) = self.rules[&h].selector.matches(g, id) {
                m.insert(id, s);
            }
        }
        for (id, s) in &m {
            self.item_rules.entry(*id).or_default().insert(h, *s);
        }
        self.matched.insert(h, m);
    }

    fn unmatch_rule(&mut self, h: RuleHandle) -> BTreeSet<Id> {
        let m = self.matched.remove(&h).unwrap_or_default();
        for id in m.keys() {
            if let Some(r) = self.item_rules.get_mut(id) {
                r.remove(&h);
                if r.is_empty() {
                    self.item_rules.remove(id);
                }
            }
        }
        m.into_keys().collect()
    }

    /// Adds a rule and matches it. Returns its handle and the items it affects.
    pub fn insert_rule<G: ItemGraph<Id = Id>>(
        &mut self,
        g: &G,
        rule: Rule<V>,
    ) -> (RuleHandle, BTreeSet<Id>) {
        let h = RuleHandle(self.next);
        self.next += 1;
        self.rules.insert(h, rule);
        self.reindex();
        self.match_rule_everywhere(g, h);
        let items: BTreeSet<Id> = self.matched[&h].keys().copied().collect();
        (h, self.rule_affected(g, &items))
    }

    /// Adds a rule without matching it; call [`Cascade::rebuild`] before resolving.
    pub fn push_rule(&mut self, rule: Rule<V>) -> RuleHandle {
        let h = RuleHandle(self.next);
        self.next += 1;
        self.rules.insert(h, rule);
        h
    }

    pub fn remove_rule<G: ItemGraph<Id = Id>>(&mut self, g: &G, h: RuleHandle) -> BTreeSet<Id> {
        if self.rules.remove(&h).is_none() {
            return BTreeSet::new();
        }
        let items = self.unmatch_rule(h);
        self.reindex();
        self.rule_affected(g, &items)
    }

    /// Replaces a rule in place, keeping its handle. Returns old ∪ new matches.
    pub fn replace_rule<G: ItemGraph<Id = Id>>(
        &mut self,
        g: &G,
        h: RuleHandle,
        rule: Rule<V>,
    ) -> BTreeSet<Id> {
        let mut items = self.unmatch_rule(h);
        self.rules.insert(h, rule);
        self.reindex();
        self.match_rule_everywhere(g, h);
        items.extend(self.matched[&h].keys().copied());
        self.rule_affected(g, &items)
    }

    pub fn set_inline(&mut self, id: Id, property: &str, value: V, source: SourceKey) {
        self.inline
            .insert((id, property.to_string()), (value, source));
    }
    pub fn clear_inline(&mut self, id: Id, property: &str) {
        self.inline.remove(&(id, property.to_string()));
    }

    fn rule_candidates(&self, id: Id, property: &str) -> Vec<(PrecKey, &Rule<V>, &V, &SourceKey)> {
        let mut out = Vec::new();
        if let Some(rs) = self.item_rules.get(&id) {
            for (h, s) in rs {
                let r = &self.rules[h];
                if let Some(d) = r.declared(property) {
                    out.push(((*s, r.priority, r.order, *h), r, &d.value, &d.source));
                }
            }
        }
        // Highest key first.
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out
    }

    fn own_value(&self, id: Id, property: &str) -> Option<Resolved<V, Id>> {
        if let Some((v, src)) = self.inline.get(&(id, property.to_string())) {
            return Some(Resolved {
                value: v.clone(),
                origin: Origin::Inline {
                    source: src.clone(),
                },
            });
        }
        let c = self.rule_candidates(id, property);
        c.first().map(|((s, p, o, h), r, v, src)| Resolved {
            value: (*v).clone(),
            origin: Origin::Rule {
                rule_id: r.id.clone(),
                handle: *h,
                source: (*src).clone(),
                specificity: *s,
                priority: *p,
                order: *o,
            },
        })
    }

    /// The effective value of `property` on `id`.
    pub fn resolve<G: ItemGraph<Id = Id>>(
        &self,
        g: &G,
        id: Id,
        property: &str,
    ) -> Option<Resolved<V, Id>> {
        if let Some(r) = self.own_value(id, property) {
            return Some(r);
        }
        let spec = self.props.get(property);
        if spec.is_some_and(|s| s.inherited) {
            let mut p = g.parent(id);
            while let Some(a) = p {
                if let Some(r) = self.own_value(a, property) {
                    return Some(Resolved {
                        value: r.value,
                        origin: Origin::Inherited {
                            from: a,
                            origin: Box::new(r.origin),
                        },
                    });
                }
                p = g.parent(a);
            }
        }
        spec.and_then(|s| s.default.clone()).map(|v| Resolved {
            value: v,
            origin: Origin::Default,
        })
    }

    /// The effective value, its origin, and every overridden rule candidate with the reason
    /// it lost.
    pub fn explain<G: ItemGraph<Id = Id>>(
        &self,
        g: &G,
        id: Id,
        property: &str,
    ) -> Explanation<V, Id> {
        let effective = self.resolve(g, id, property);
        let cands = self.rule_candidates(id, property);
        let inline = self.inline.contains_key(&(id, property.to_string()));
        let mut overridden = Vec::new();
        let winner_key = if inline {
            None
        } else {
            cands.first().map(|c| c.0)
        };
        let skip = usize::from(!inline && !cands.is_empty());
        for ((s, p, o, h), r, v, src) in cands.iter().skip(skip) {
            let lost_to = match winner_key {
                None => LostTo::Inline,
                Some((ws, wp, _, _)) => {
                    if *s < ws {
                        LostTo::LowerSpecificity
                    } else if *p < wp {
                        LostTo::LowerPriority
                    } else {
                        LostTo::EarlierSource
                    }
                }
            };
            overridden.push(Candidate {
                rule_id: r.id.clone(),
                handle: *h,
                source: (*src).clone(),
                specificity: *s,
                priority: *p,
                order: *o,
                value: (*v).clone(),
                lost_to,
            });
        }
        Explanation {
            item: id,
            property: property.to_string(),
            effective,
            overridden,
        }
    }

    /// Every property with a value on `id` (own, inherited or default), by name.
    pub fn effective_all<G: ItemGraph<Id = Id>>(
        &self,
        g: &G,
        id: Id,
    ) -> BTreeMap<String, Resolved<V, Id>> {
        let mut names: BTreeSet<String> = self.props.keys().cloned().collect();
        if let Some(rs) = self.item_rules.get(&id) {
            for h in rs.keys() {
                for d in &self.rules[h].decls {
                    names.insert(d.property.clone());
                }
            }
        }
        for (i, p) in self.inline.keys() {
            if *i == id {
                names.insert(p.clone());
            }
        }
        names
            .into_iter()
            .filter_map(|n| self.resolve(g, id, &n).map(|r| (n, r)))
            .collect()
    }

    /// A deterministic digest of every match, for determinism tests.
    pub fn match_table(&self) -> Vec<(Id, Vec<(RuleHandle, Specificity)>)> {
        self.item_rules
            .iter()
            .map(|(i, m)| (*i, m.iter().map(|(h, s)| (*h, *s)).collect()))
            .collect()
    }
}
