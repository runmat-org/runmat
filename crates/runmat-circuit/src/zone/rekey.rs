//! A whole re-read numbers items afresh, so a cached fill finds its zone again by content.

use crate::layer_set::CopperIndex;
use super::{FillCache, LayerCache};
use crate::layout::{ItemId, Zone};
use std::collections::{BTreeMap, BTreeSet};

/// `z` without its fill and with the id `id`: what a cached layer stores as its zone.
fn bare_as(z: &Zone, id: ItemId) -> Zone {
    let mut b = z.clone();
    b.fill.clear();
    b.id = id;
    b
}

impl FillCache {
    /// Moves each cached zone onto the id its zone has in `zones` now. A cached zone keeps
    /// its id when the zone of that id has the same content; otherwise it moves to a zone of
    /// the same content (the id aside) that no other cached zone took. A cached zone with no
    /// match leaves the cache. Every kept layer re-merges on the next fill, since its report
    /// names footprints by id, and those may have been renumbered too.
    pub fn rekey(&mut self, zones: &BTreeMap<ItemId, Zone>) {
        let mut by_zone: BTreeMap<ItemId, Vec<(CopperIndex, LayerCache)>> = BTreeMap::new();
        for ((zid, layer), c) in std::mem::take(&mut self.layers) {
            by_zone.entry(zid).or_default().push((layer, c));
        }
        let mut taken: BTreeSet<ItemId> = BTreeSet::new();
        let mut moves: Vec<(ItemId, ItemId)> = Vec::new();
        // First, the zones whose id still holds the same content.
        for (oid, layers) in &by_zone {
            if zones
                .get(oid)
                .is_some_and(|z| bare_as(z, *oid) == layers[0].1.zone)
            {
                taken.insert(*oid);
                moves.push((*oid, *oid));
            }
        }
        let kept: BTreeSet<ItemId> = moves.iter().map(|(o, _)| *o).collect();
        for (oid, layers) in &by_zone {
            if kept.contains(oid) {
                continue;
            }
            let cached = &layers[0].1.zone;
            if let Some(z) = zones
                .values()
                .find(|z| !taken.contains(&z.id) && bare_as(z, *oid) == *cached)
            {
                taken.insert(z.id);
                moves.push((*oid, z.id));
            }
        }
        for (oid, nid) in moves {
            for (layer, mut c) in by_zone.remove(&oid).unwrap_or_default() {
                c.zone.id = nid;
                c.remerge = true;
                self.layers.insert((nid, layer), c);
            }
        }
    }
}
