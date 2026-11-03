//! Stable identity: hierarchical name paths mapped to UUIDs through a lock section.
//!
//! Every elaborated item (component instance, net, sheet instance, pin, layout object) has a
//! [`NamePath`] such as `/power/buck[2]/U1`. The [`LockSection`] maps paths to [`Uuid`]s the
//! way `Cargo.lock` pins versions: a UUID is allocated the first time a path is seen and kept
//! from then on, so layout, review comments and ECO history attach to the UUID, not to the
//! name. [`LockSection::reconcile`] compares the lock with a new elaboration and reports each
//! item as kept, new, renamed or removed:
//!
//! 1. an exact path match keeps its UUID;
//! 2. an explicit rename (`renamed_from` in source, or a recorded alias) carries the UUID to
//!    the new path and records an alias, so a rename is an ECO *rename*, not delete + add;
//! 3. an unmatched old and new item of the same kind with the same *fingerprint* (for a
//!    component: its part and connected nets) pair up as an inferred rename, when the pairing
//!    is unique;
//! 4. anything else is new (fresh UUID) or removed.
//!
//! This module is pure data. [`LockSection::to_text`]/[`LockSection::from_text`] give a
//! canonical line format; file I/O belongs to the host.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Uuid(pub u128);

impl Uuid {
    pub const NIL: Uuid = Uuid(0);
    /// Parses the hyphenated 8-4-4-4-12 form (case-insensitive).
    pub fn parse(s: &str) -> Option<Uuid> {
        let hex: String = s.chars().filter(|c| *c != '-').collect();
        if hex.len() != 32 || s.len() != 36 {
            return None;
        }
        u128::from_str_radix(&hex, 16).ok().map(Uuid)
    }
    /// Sets the RFC 9562 version and variant bits.
    pub fn with_version(self, v: u8) -> Uuid {
        let mut x = self.0;
        x &= !(0xF << 76);
        x |= (v as u128 & 0xF) << 76;
        x &= !(0x3 << 62);
        x |= 0x2 << 62;
        Uuid(x)
    }
}

impl fmt::Display for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let h = format!("{:032x}", self.0);
        write!(
            f,
            "{}-{}-{}-{}-{}",
            &h[0..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..32]
        )
    }
}

/// 128-bit FNV-1a, used for name-derived UUIDs. Stable across platforms and releases.
pub fn fnv128(data: &[u8]) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    let mut h = OFFSET;
    for b in data {
        h ^= *b as u128;
        h = h.wrapping_mul(PRIME);
    }
    // Final avalanche so similar paths differ in every nibble.
    h ^= h >> 67;
    h = h.wrapping_mul(0x9E3779B97F4A7C15F39CC0605CEDC835);
    h ^= h >> 59;
    h
}

/// A hierarchical name path. Segments are sheet-instance names (with a channel index for
/// repeated sheets, `adc[3]`) followed by the item's local name.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamePath(pub Vec<String>);

impl NamePath {
    pub fn root() -> Self {
        NamePath(Vec::new())
    }
    pub fn parse(s: &str) -> Self {
        NamePath(
            s.split('/')
                .filter(|x| !x.is_empty())
                .map(|x| x.to_string())
                .collect(),
        )
    }
    pub fn child(&self, seg: impl Into<String>) -> Self {
        let mut v = self.0.clone();
        v.push(seg.into());
        NamePath(v)
    }
    pub fn parent(&self) -> Option<NamePath> {
        if self.0.is_empty() {
            None
        } else {
            Some(NamePath(self.0[..self.0.len() - 1].to_vec()))
        }
    }
    pub fn leaf(&self) -> &str {
        self.0.last().map(|s| s.as_str()).unwrap_or("")
    }
    pub fn depth(&self) -> usize {
        self.0.len()
    }
    pub fn starts_with(&self, p: &NamePath) -> bool {
        self.0.len() >= p.0.len() && self.0[..p.0.len()] == p.0[..]
    }
    /// Replaces the prefix `from` with `to` (for sheet-instance renames).
    pub fn reprefix(&self, from: &NamePath, to: &NamePath) -> Option<NamePath> {
        self.starts_with(from).then(|| {
            let mut v = to.0.clone();
            v.extend_from_slice(&self.0[from.0.len()..]);
            NamePath(v)
        })
    }
}

impl fmt::Display for NamePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("/");
        }
        for s in &self.0 {
            write!(f, "/{s}")?;
        }
        Ok(())
    }
}

/// Where new UUIDs come from.
pub trait UuidSource {
    fn allocate(&mut self, kind: &str, path: &NamePath) -> Uuid;
}

/// Name-derived UUIDs (version 8): the same namespace, kind and path always give the same
/// UUID. Used for reproducible builds and tests, and for items that are re-derived rather
/// than authored.
#[derive(Clone, Debug)]
pub struct DerivedUuids {
    pub namespace: u128,
}

impl UuidSource for DerivedUuids {
    fn allocate(&mut self, kind: &str, path: &NamePath) -> Uuid {
        let s = format!("{:032x}|{kind}|{path}", self.namespace);
        Uuid(fnv128(s.as_bytes())).with_version(8)
    }
}

/// Seeded pseudo-random UUIDs (version 4 layout) for first-compile allocation. A host seeds
/// it from the OS once per session; tests use a fixed seed.
#[derive(Clone, Debug)]
pub struct SeededUuids {
    state: u128,
}

impl SeededUuids {
    pub fn new(seed: u128) -> Self {
        SeededUuids {
            state: seed ^ 0x2545F4914F6CDD1D_9E3779B97F4A7C15,
        }
    }
}

impl UuidSource for SeededUuids {
    fn allocate(&mut self, _kind: &str, _path: &NamePath) -> Uuid {
        // xorshift128 on two lanes, then mix.
        self.state ^= self.state << 23;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 26;
        Uuid(fnv128(&self.state.to_le_bytes())).with_version(4)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LockEntry {
    pub uuid: Uuid,
    pub kind: String,
    /// Assigned designator for components, so re-annotation is stable.
    pub designator: Option<String>,
}

/// One item of a new elaboration, as presented to [`LockSection::reconcile`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LockItem {
    pub path: NamePath,
    pub kind: String,
    /// Explicit previous path from source (`renamed_from`).
    pub renamed_from: Option<NamePath>,
    /// Content fingerprint for inferred renames; `None` disables inference for this item.
    pub fingerprint: Option<String>,
    pub designator: Option<String>,
}

impl LockItem {
    pub fn new(path: NamePath, kind: &str) -> Self {
        LockItem {
            path,
            kind: kind.to_string(),
            renamed_from: None,
            fingerprint: None,
            designator: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IdChange {
    Kept {
        path: NamePath,
        uuid: Uuid,
    },
    New {
        path: NamePath,
        uuid: Uuid,
    },
    Renamed {
        from: NamePath,
        to: NamePath,
        uuid: Uuid,
        /// `true` when paired by fingerprint rather than an explicit rename.
        inferred: bool,
    },
    Removed {
        path: NamePath,
        uuid: Uuid,
    },
}

impl IdChange {
    pub fn uuid(&self) -> Uuid {
        match self {
            IdChange::Kept { uuid, .. }
            | IdChange::New { uuid, .. }
            | IdChange::Renamed { uuid, .. }
            | IdChange::Removed { uuid, .. } => *uuid,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockSection {
    pub entries: BTreeMap<NamePath, LockEntry>,
    /// Rename history: old path → new path. Followed transitively by `reconcile`.
    pub aliases: BTreeMap<NamePath, NamePath>,
    /// Fingerprints recorded at the last reconcile, for inferred renames.
    pub fingerprints: BTreeMap<NamePath, String>,
}

impl LockSection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn uuid_of(&self, p: &NamePath) -> Option<Uuid> {
        self.entries.get(p).map(|e| e.uuid)
    }

    pub fn path_of(&self, u: Uuid) -> Option<&NamePath> {
        self.entries
            .iter()
            .find(|(_, e)| e.uuid == u)
            .map(|(p, _)| p)
    }

    fn follow_alias(&self, p: &NamePath) -> NamePath {
        let mut cur = p.clone();
        let mut seen = BTreeSet::new();
        while let Some(n) = self.aliases.get(&cur) {
            if !seen.insert(cur.clone()) {
                break;
            }
            cur = n.clone();
        }
        cur
    }

    /// Reconciles the lock with a new set of items, allocating UUIDs for new ones. The lock is
    /// updated in place; the returned changes are sorted (kept, new, renamed, removed are
    /// ordered by path).
    pub fn reconcile(&mut self, items: &[LockItem], src: &mut dyn UuidSource) -> Vec<IdChange> {
        let mut items: Vec<&LockItem> = items.iter().collect();
        items.sort();
        let mut old: BTreeMap<NamePath, LockEntry> = std::mem::take(&mut self.entries);
        let old_fp = std::mem::take(&mut self.fingerprints);
        let mut out = Vec::new();
        let mut assigned: BTreeMap<NamePath, LockEntry> = BTreeMap::new();
        let mut pending: Vec<&LockItem> = Vec::new();

        // 1. Exact matches.
        for it in &items {
            if let Some(e) = old.get(&it.path) {
                if e.kind == it.kind {
                    let e = old.remove(&it.path).unwrap();
                    out.push(IdChange::Kept {
                        path: it.path.clone(),
                        uuid: e.uuid,
                    });
                    assigned.insert(
                        it.path.clone(),
                        LockEntry {
                            designator: it.designator.clone().or(e.designator),
                            ..e
                        },
                    );
                    continue;
                }
            }
            pending.push(it);
        }
        // 2. Explicit renames and recorded aliases.
        let mut still = Vec::new();
        for it in pending {
            let mut from = it.renamed_from.clone();
            if from.is_none() {
                // An alias chain ending at this path names its old identity.
                for (o, _) in self.aliases.iter() {
                    if self.follow_alias(o) == it.path && old.contains_key(o) {
                        from = Some(o.clone());
                        break;
                    }
                }
            }
            if let Some(f) = from {
                if old.get(&f).is_some_and(|e| e.kind == it.kind) {
                    let e = old.remove(&f).unwrap();
                    out.push(IdChange::Renamed {
                        from: f.clone(),
                        to: it.path.clone(),
                        uuid: e.uuid,
                        inferred: false,
                    });
                    self.aliases.insert(f, it.path.clone());
                    assigned.insert(
                        it.path.clone(),
                        LockEntry {
                            designator: it.designator.clone().or(e.designator),
                            ..e
                        },
                    );
                    continue;
                }
            }
            still.push(it);
        }
        // 3. Inferred renames: unique fingerprint matches of the same kind.
        let mut by_fp_old: BTreeMap<(String, String), Vec<NamePath>> = BTreeMap::new();
        for (p, e) in &old {
            if let Some(fp) = old_fp.get(p) {
                by_fp_old
                    .entry((e.kind.clone(), fp.clone()))
                    .or_default()
                    .push(p.clone());
            }
        }
        let mut by_fp_new: BTreeMap<(String, String), usize> = BTreeMap::new();
        for it in &still {
            if let Some(fp) = &it.fingerprint {
                *by_fp_new.entry((it.kind.clone(), fp.clone())).or_default() += 1;
            }
        }
        let mut fresh = Vec::new();
        for it in still {
            if let Some(fp) = &it.fingerprint {
                let key = (it.kind.clone(), fp.clone());
                if by_fp_new.get(&key) == Some(&1) {
                    if let Some(cands) = by_fp_old.get(&key) {
                        if cands.len() == 1 && old.contains_key(&cands[0]) {
                            let f = cands[0].clone();
                            let e = old.remove(&f).unwrap();
                            out.push(IdChange::Renamed {
                                from: f.clone(),
                                to: it.path.clone(),
                                uuid: e.uuid,
                                inferred: true,
                            });
                            self.aliases.insert(f, it.path.clone());
                            assigned.insert(
                                it.path.clone(),
                                LockEntry {
                                    designator: it.designator.clone().or(e.designator),
                                    ..e
                                },
                            );
                            continue;
                        }
                    }
                }
            }
            fresh.push(it);
        }
        // 4. New and removed.
        for it in fresh {
            let uuid = src.allocate(&it.kind, &it.path);
            out.push(IdChange::New {
                path: it.path.clone(),
                uuid,
            });
            assigned.insert(
                it.path.clone(),
                LockEntry {
                    uuid,
                    kind: it.kind.clone(),
                    designator: it.designator.clone(),
                },
            );
        }
        for (p, e) in old {
            out.push(IdChange::Removed {
                path: p,
                uuid: e.uuid,
            });
        }
        for it in &items {
            if let Some(fp) = &it.fingerprint {
                self.fingerprints.insert(it.path.clone(), fp.clone());
            }
        }
        // Aliases that point at removed paths are dropped; chains are kept compact.
        let live: BTreeSet<NamePath> = assigned.keys().cloned().collect();
        let compact: BTreeMap<NamePath, NamePath> = self
            .aliases
            .keys()
            .map(|k| (k.clone(), self.follow_alias(k)))
            .filter(|(_, v)| live.contains(v))
            .collect();
        self.aliases = compact;
        self.entries = assigned;
        out.sort();
        out
    }

    /// Canonical text: one `entry` line per path, then `alias` and `fp` lines, sorted.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        for (p, e) in &self.entries {
            s.push_str(&format!("entry {} {} {}", e.uuid, e.kind, p));
            if let Some(d) = &e.designator {
                s.push_str(&format!(" designator={d}"));
            }
            s.push('\n');
        }
        for (a, b) in &self.aliases {
            s.push_str(&format!("alias {a} -> {b}\n"));
        }
        for (p, f) in &self.fingerprints {
            s.push_str(&format!("fp {p} {f}\n"));
        }
        s
    }

    pub fn from_text(t: &str) -> Result<LockSection, String> {
        let mut l = LockSection::new();
        for (n, line) in t.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("entry") => {
                    let u = parts
                        .next()
                        .and_then(Uuid::parse)
                        .ok_or_else(|| format!("line {}: bad uuid", n + 1))?;
                    let kind = parts.next().ok_or(format!("line {}: no kind", n + 1))?;
                    let path = parts.next().ok_or(format!("line {}: no path", n + 1))?;
                    let designator = parts
                        .next()
                        .and_then(|x| x.strip_prefix("designator="))
                        .map(|x| x.to_string());
                    l.entries.insert(
                        NamePath::parse(path),
                        LockEntry {
                            uuid: u,
                            kind: kind.to_string(),
                            designator,
                        },
                    );
                }
                Some("alias") => {
                    let a = parts.next().ok_or(format!("line {}: bad alias", n + 1))?;
                    let _ = parts.next();
                    let b = parts.next().ok_or(format!("line {}: bad alias", n + 1))?;
                    l.aliases.insert(NamePath::parse(a), NamePath::parse(b));
                }
                Some("fp") => {
                    let p = parts.next().ok_or(format!("line {}: bad fp", n + 1))?;
                    let f: Vec<&str> = parts.collect();
                    l.fingerprints.insert(NamePath::parse(p), f.join(" "));
                }
                Some(other) => return Err(format!("line {}: unknown record `{other}`", n + 1)),
                None => {}
            }
        }
        Ok(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(p: &str, k: &str) -> LockItem {
        LockItem::new(NamePath::parse(p), k)
    }

    #[test]
    fn uuid_text() {
        let u = DerivedUuids { namespace: 7 }.allocate("component", &NamePath::parse("/a/U1"));
        let s = u.to_string();
        assert_eq!(s.len(), 36);
        assert_eq!(&s[14..15], "8");
        assert_eq!(Uuid::parse(&s), Some(u));
        let v = DerivedUuids { namespace: 7 }.allocate("component", &NamePath::parse("/a/U2"));
        assert_ne!(u, v);
    }

    #[test]
    fn reconcile_keep_new_rename_remove() {
        let mut lock = LockSection::new();
        let mut src = SeededUuids::new(1);
        let c = lock.reconcile(
            &[item("/U1", "component"), item("/R1", "component")],
            &mut src,
        );
        assert!(c.iter().all(|x| matches!(x, IdChange::New { .. })));
        let u1 = lock.uuid_of(&NamePath::parse("/U1")).unwrap();
        let r1 = lock.uuid_of(&NamePath::parse("/R1")).unwrap();

        // Explicit rename U1 → MCU; R1 removed; C1 new.
        let mut mcu = item("/MCU", "component");
        mcu.renamed_from = Some(NamePath::parse("/U1"));
        let c = lock.reconcile(&[mcu, item("/C1", "component")], &mut src);
        assert!(c.contains(&IdChange::Renamed {
            from: NamePath::parse("/U1"),
            to: NamePath::parse("/MCU"),
            uuid: u1,
            inferred: false
        }));
        assert!(c.contains(&IdChange::Removed {
            path: NamePath::parse("/R1"),
            uuid: r1
        }));
        assert_eq!(lock.uuid_of(&NamePath::parse("/MCU")), Some(u1));
        assert_eq!(
            lock.aliases.get(&NamePath::parse("/U1")),
            Some(&NamePath::parse("/MCU"))
        );

        // Round trip through text.
        let t = lock.to_text();
        assert_eq!(LockSection::from_text(&t).unwrap(), lock);
    }

    #[test]
    fn inferred_rename_by_fingerprint() {
        let mut lock = LockSection::new();
        let mut src = DerivedUuids { namespace: 1 };
        let mut a = item("/power/U3", "component");
        a.fingerprint = Some("TPS62130|VIN,SW,FB".into());
        let mut b = item("/power/C1", "component");
        b.fingerprint = Some("C0603|VIN,GND".into());
        lock.reconcile(&[a.clone(), b.clone()], &mut src);
        let u3 = lock.uuid_of(&a.path).unwrap();
        // The sheet is renamed; fingerprints still match one-to-one.
        let mut a2 = a.clone();
        a2.path = NamePath::parse("/buck/U3");
        let mut b2 = b.clone();
        b2.path = NamePath::parse("/buck/C1");
        let c = lock.reconcile(&[a2.clone(), b2], &mut src);
        assert!(c
            .iter()
            .any(|x| matches!(x, IdChange::Renamed { inferred: true, uuid, .. } if *uuid == u3)));
        // Ambiguous fingerprints never pair.
        let mut l2 = LockSection::new();
        let mut x = item("/C1", "component");
        x.fingerprint = Some("C|GND".into());
        let mut y = item("/C2", "component");
        y.fingerprint = Some("C|GND".into());
        l2.reconcile(&[x.clone(), y.clone()], &mut src);
        x.path = NamePath::parse("/C10");
        y.path = NamePath::parse("/C11");
        let c = l2.reconcile(&[x, y], &mut src);
        assert_eq!(
            c.iter()
                .filter(|x| matches!(x, IdChange::New { .. }))
                .count(),
            2
        );
    }
}
