//! Spatial lookup of tracks and pads touching via barrels.

use crate::geom::{mm, shape::Prim};
use crate::layer_set::{CopperIndex, LayerSet};
use crate::layout::{Board, NetId, Via, Zone};
use crate::spatial::Grid;
use std::collections::BTreeMap;

struct Contact {
    net: NetId,
    layers: LayerSet,
    prim: Prim,
}

/// One board snapshot's contact geometry, shared across the complete via query.
pub(super) struct ViaConnections<'a> {
    contacts: Vec<Contact>,
    grid: Grid<usize>,
    zones: BTreeMap<NetId, Vec<&'a Zone>>,
}

impl<'a> ViaConnections<'a> {
    pub fn new(board: &'a Board) -> Self {
        let mut index = Self {
            contacts: Vec::new(),
            grid: Grid::new(mm(2.0)),
            zones: BTreeMap::new(),
        };
        for track in board.tracks.values() {
            let Some(net) = track.net else { continue };
            for prim in track.prims() {
                index.insert(Contact {
                    net,
                    layers: LayerSet::single(track.layer),
                    prim,
                });
            }
        }
        let layers = board.stackup.copper_count();
        for footprint in board.footprints.values() {
            for pad in footprint.pads(layers) {
                let Some(net) = pad.net else { continue };
                index.insert(Contact {
                    net,
                    layers: pad.layers.into_iter().collect(),
                    prim: pad.prim,
                });
            }
        }
        // Pours remain outline-containment queries; their large interiors do
        // not belong in the small-contact spatial grid.
        for zone in board.zones.values() {
            if let Some(net) = zone.net {
                index.zones.entry(net).or_default().push(zone);
            }
        }
        index
    }

    fn insert(&mut self, contact: Contact) {
        let slot = self.contacts.len();
        self.grid.insert(slot, contact.prim.bbox());
        self.contacts.push(contact);
    }

    pub fn connected_layers(&self, via: &Via) -> Vec<CopperIndex> {
        let Some(net) = via.net else { return vec![] };
        let barrel = Prim::circle(via.at, via.diameter / 2);
        let span = via.from..=via.to;
        let mut on = LayerSet::new();
        for slot in self.grid.query(&barrel.bbox().inflate(1)) {
            let contact = &self.contacts[slot];
            if contact.net == net
                && contact.layers.iter().any(|layer| span.contains(&layer))
                && barrel.gap(&contact.prim) <= 0.0
            {
                for layer in contact.layers.iter().filter(|layer| span.contains(layer)) {
                    on.insert(layer);
                }
            }
        }
        for zone in self.zones.get(&net).into_iter().flatten() {
            if zone.outline.contains(via.at).unwrap_or(true) {
                for &layer in &zone.layers {
                    if span.contains(&layer) {
                        on.insert(layer);
                    }
                }
            }
        }
        on.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The previous scan is the equivalence oracle for the spatial broad phase.
    fn scan(board: &Board, via: &Via) -> Vec<CopperIndex> {
        let Some(net) = via.net else { return vec![] };
        let barrel = Prim::circle(via.at, via.diameter / 2);
        let mut on = std::collections::BTreeSet::new();
        let span = via.from..=via.to;
        for track in board.tracks.values() {
            if track.net == Some(net)
                && span.contains(&track.layer)
                && track.prims().iter().any(|prim| barrel.gap(prim) <= 0.0)
            {
                on.insert(track.layer);
            }
        }
        for zone in board.zones.values() {
            if zone.net == Some(net) && zone.outline.contains(via.at).unwrap_or(true) {
                on.extend(
                    zone.layers
                        .iter()
                        .copied()
                        .filter(|layer| span.contains(layer)),
                );
            }
        }
        for footprint in board.footprints.values() {
            if !footprint.pad_nets.values().any(|&other| other == net) {
                continue;
            }
            for pad in footprint.pads(board.stackup.copper_count()) {
                if pad.net == Some(net) && barrel.gap(&pad.prim) <= 0.0 {
                    on.extend(
                        pad.layers
                            .iter()
                            .copied()
                            .filter(|layer| span.contains(layer)),
                    );
                }
            }
        }
        on.into_iter().collect()
    }

    #[test]
    fn spatial_via_contacts_match_the_scan_on_shallow_and_deep_boards() {
        for copper_layers in [8, 80] {
            let (_, _, _, board) = crate::fixture::large(&crate::fixture::LargeParams {
                channels: 2,
                copper_layers,
                route_every: 1,
                pours: true,
            });
            assert!(!board.vias.is_empty());
            let index = ViaConnections::new(&board);
            for via in board.vias.values() {
                assert_eq!(
                    index.connected_layers(via),
                    scan(&board, via),
                    "via {:?} on {copper_layers} layers",
                    via.id
                );
            }
            let mut span = board.vias.values().next().unwrap().clone();
            span.from = 0;
            span.to = CopperIndex::MAX;
            assert_eq!(index.connected_layers(&span), scan(&board, &span));
            span.from = CopperIndex::MAX;
            span.to = 0;
            assert_eq!(index.connected_layers(&span), scan(&board, &span));
        }
    }
}
