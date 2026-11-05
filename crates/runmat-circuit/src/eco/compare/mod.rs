//! The forward compare: what the board needs to match the schematic, in phases that each
//! push their lines: nets (renames first), components, net classes and rules, differential
//! pairs, and component classes and rooms (multi-channel).

use super::*;
use crate::layout::Board;
use std::collections::BTreeMap;

mod channels;
mod classes;
mod nets;
mod parts;

/// The inputs every phase reads.
#[derive(Clone, Copy)]
pub(super) struct Inputs<'a> {
    pub(super) s: &'a Schematic<'a>,
    pub(super) nl: &'a crate::elab::Netlist,
    pub(super) board: &'a Board,
    pub(super) o: &'a ComparatorOptions,
    pub(super) root: &'a str,
    pub(super) links: &'a Links,
    pub(super) base: &'a BoardSync,
}

/// What the nets phase settled, for the phases after it: board nets renamed to their
/// schematic names, and the schematic's nets the board doesn't have.
pub(super) struct Nets {
    pub(super) renamed: BTreeMap<String, String>,
    pub(super) new: Vec<String>,
}

impl Nets {
    /// A board net name as the schematic now calls it.
    pub(super) fn board_as_sch(&self, name: &str) -> String {
        self.renamed.get(name).cloned().unwrap_or_else(|| name.to_owned())
    }

    /// The line a change on net `name` needs first: the net's rename or its addition, when
    /// the board doesn't have it yet.
    pub(super) fn add_req(&self, name: &str) -> Option<String> {
        if !self.new.iter().any(|n| n == name) {
            return None;
        }
        if self.renamed.values().any(|v| v == name) {
            self.renamed
                .iter()
                .find(|(_, v)| v.as_str() == name)
                .map(|(k, _)| format!("rename_net:{k}"))
        } else {
            Some(format!("add_net:{name}"))
        }
    }
}

/// Schematic → board: everything the board would change to match the schematic.
pub fn compare(s: &Schematic, board: &Board, o: &ComparatorOptions) -> EcoPlan {
    let root = s.design.root.clone();
    let links = link(s, board, o);
    let x = Inputs { s, nl: s.netlist, board, o, root: &root, links: &links, base: &board.sync };
    let mut b = Builder {
        changes: Vec::new(),
        direction: Direction::Forward,
    };
    let nets = nets::compare_nets(&x, &mut b);
    let new_rooms = parts::compare_parts(&x, &nets, &mut b);
    classes::compare_classes(&x, &nets, &mut b);
    classes::compare_pairs(&x, &nets, &mut b);
    channels::compare_channels(&x, &new_rooms, &mut b);
    finish(Direction::Forward, b.changes, o)
}
