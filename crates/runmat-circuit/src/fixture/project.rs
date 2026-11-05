//! The large fixture as source files: the `.circuit` and `.circuit.layout` a person would open,
//! for the document runtime's scale tests and the Desktop's end-to-end edit timing.

use super::{large, LargeParams};
use crate::runtime::Files;
use crate::syntax::board::print_board;
use crate::syntax::print_circuit;
use runmat_design_syntax::document::UseItem;

/// Where [`large_files`] puts the design.
pub const LARGE_CIRCUIT: &str = "/p/big.circuit";
pub const LARGE_LAYOUT: &str = "/p/big.circuit.layout";

/// The design `large(params)` builds, printed as `big.circuit` and `big.circuit.layout`, and the
/// board's primitive count (footprints, tracks, vias, pours and pads).
pub fn large_files(params: &LargeParams) -> (Files, usize) {
    let (_, mut design, _, mut board) = large(params);
    design.name = "big".into();
    board.name = "big".into();
    let primitives = board.footprints.len() + board.tracks.len() + board.vias.len() + board.zones.len() + board.all_pads().len();
    let uses = vec![UseItem {
        symbol: Some(("circuit".into(), "big".into())),
        from: "./big.circuit".into(),
        alias: None,
        revision: None,
        span: Default::default(),
    }];
    let mut files = Files::new();
    files.insert(LARGE_CIRCUIT.into(), print_circuit(&design).expect("the fixture prints"));
    files.insert(LARGE_LAYOUT.into(), print_board(&board, &uses));
    (files, primitives)
}

/// The parameters of the 50k-primitive board the edit budget is measured on: 700 channels on
/// eight copper layers, every channel routed, with pours.
pub fn scale_params() -> LargeParams {
    LargeParams { channels: 700, copper_layers: 8, route_every: 1, pours: true }
}
