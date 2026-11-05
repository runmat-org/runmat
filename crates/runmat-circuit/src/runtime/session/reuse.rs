//! A whole re-read keeps the fill tiles its change didn't reach.
//!
//! A change outside the item statements (a package, a net, the sync section, an ECO's
//! reprint) makes the session read the layout whole, with items numbered afresh. The fills
//! depend on geometry, not on ids, so the old and new readings are compared statement by
//! statement: an item statement whose text is in both is the same copper, and every other
//! item, in either reading, marks its box dirty on its layers. A footprint whose package
//! definition changed is dirty though its statement is not. A change to the outline,
//! stack-up, regions, limits, rules or a net's class is dirty everywhere.

use super::*;
use crate::syntax::board::{item_of, ParsedBoard};
use std::collections::HashMap;

/// Where fills may differ between `old` (read from `old_text`) and `new` (from `new_text`).
pub(super) fn whole_dirty(
    old_text: &str,
    old: &ParsedBoard,
    new_text: &str,
    new: &ParsedBoard,
    reach: crate::geom::Coord,
) -> FillDirty {
    let (Some(ob), Some(nb)) = (old.board.as_ref(), new.board.as_ref()) else {
        return FillDirty::everywhere();
    };
    let classes = |b: &Board| -> BTreeMap<String, Option<String>> {
        b.nets
            .iter()
            .map(|n| (n.name.clone(), n.class.clone()))
            .collect()
    };
    let same_ground = ob.outline == nb.outline
        && ob.stackup == nb.stackup
        && ob.stackups == nb.stackups
        && ob.regions == nb.regions
        && ob.limits == nb.limits
        && ob.rules == nb.rules
        && ob.net_class_rules == nb.net_class_rules
        && {
            let (a, b) = (classes(ob), classes(nb));
            a.iter()
                .all(|(name, class)| b.get(name).is_none_or(|c| c == class))
        };
    if !same_ground {
        return FillDirty::everywhere();
    }
    let texts = |p: &ParsedBoard, src: &str| -> Vec<(ItemId, String)> {
        p.statements
            .iter()
            .filter_map(|s| Some((s.item?, src.get(s.span.start..s.span.end)?.to_owned())))
            .collect()
    };
    let (old_items, new_items) = (texts(old, old_text), texts(new, new_text));
    let mut count: HashMap<&str, i64> = HashMap::new();
    for (_, t) in &new_items {
        *count.entry(t.as_str()).or_default() += 1;
    }
    let mut unmatched_old = Vec::new();
    for (id, t) in &old_items {
        match count.get_mut(t.as_str()) {
            Some(n) if *n > 0 => *n -= 1,
            _ => unmatched_old.push(*id),
        }
    }
    let mut count: HashMap<&str, i64> = HashMap::new();
    for (_, t) in &old_items {
        *count.entry(t.as_str()).or_default() += 1;
    }
    let mut unmatched_new = Vec::new();
    for (id, t) in &new_items {
        match count.get_mut(t.as_str()) {
            Some(n) if *n > 0 => *n -= 1,
            _ => unmatched_new.push(*id),
        }
    }
    // Footprints whose package is defined differently now, in either reading.
    let changed_packages: BTreeSet<&str> = old
        .packages
        .iter()
        .filter(|(name, p)| new.packages.get(*name) != Some(*p))
        .map(|(name, _)| name.as_str())
        .chain(
            new.packages
                .keys()
                .filter(|name| !old.packages.contains_key(*name))
                .map(String::as_str),
        )
        .collect();
    let repackaged = |b: &Board| -> Vec<ItemId> {
        b.footprints
            .values()
            .filter(|f| changed_packages.contains(f.package.name.as_str()))
            .map(|f| f.id)
            .collect()
    };
    unmatched_old.extend(repackaged(ob));
    unmatched_new.extend(repackaged(nb));
    let mut dirty = FillDirty::default();
    for (board, ids) in [(ob, &unmatched_old), (nb, &unmatched_new)] {
        for id in ids {
            let Some(item) = item_of(board, *id) else {
                continue;
            };
            let layers = item_layers(board, &item);
            let r = item_bbox(board, &item);
            if layers.is_empty() || r == crate::geom::Rect::EMPTY {
                // Copper the per-layer boxes don't describe (copper text, a fiducial).
                let copper = match &item {
                    BoardItem::Text(t) => matches!(t.layer, crate::layout::Layer::Copper(_)),
                    BoardItem::Graphic(_) | BoardItem::Dimension(_) => false,
                    _ => true,
                };
                if copper {
                    return FillDirty::everywhere();
                }
                continue;
            }
            for l in layers {
                dirty.add(l, r.inflate(reach));
            }
        }
    }
    dirty
}
