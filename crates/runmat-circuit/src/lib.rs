//! `runmat-circuit`: the core library of RunMat PCB.
//!
//! Logical design:
//! - [`ids`]: name paths, UUIDs and the lock section (stable identity; a rename is an ECO
//!   rename, not delete + add).
//! - [`ir`]: the authored model (sheets, components, nets, buses, harnesses, multi-channel
//!   repetition, variants, parameters with tolerances, net classes, differential pairs)
//!   behind the front-end seam the `.circuit` parser feeds.
//! - [`library`]: the pool (unit, entity, symbol, package, part) and IPC-7351 footprint
//!   generators.
//! - [`elab`]: hierarchy flattening, net extraction across sheets, buses and harnesses,
//!   annotation, variants.
//! - [`erc`]: electrical rules check with a pin-type matrix.
//!
//! Physical design:
//! - [`geom`]: integer-nanometre geometry, exact polygon booleans and offsets, primitives.
//! - [`layout`]: outline, stackup and rigid-flex regions, placement, tracks, vias, zones,
//!   keepouts, rooms, text, dimensions.
//! - [`board_rules`]: the board as a `runmat-rules` item graph; rule properties.
//! - [`zone`]: copper pour fill (clearances, thermal spokes, minimum width, islands).
//! - [`spatial`]: grid index; [`connectivity`]: physical copper islands.
//! - [`drc`]: the incremental DRC kernel.
//! - [`route`]: ratsnest, walkaround geometry, checked insertion, tuning lengths.
//!
//! Interchange and outputs:
//! - [`kicad`]: clean-room, lossless KiCad 8/9/10 board, footprint, schematic, symbol and
//!   rules files.
//! - [`fab`]: Gerber X2, Excellon, IPC-D-356, BOM, pick-and-place.
//! - [`review`]: design-review suites with typed fixes.
//! - [`fea`]: the board → FEA `pcb` source contract.
//! - [`fixture`]: a standard pool and a large-board generator for tests and benchmarks.
//!
//! Everything is deterministic: ordered maps, sorted outputs, exact integer predicates.

pub mod board_rules;
pub mod connectivity;
pub mod drc;
pub mod elab;
pub mod erc;
pub mod fab;
pub mod fea;
pub mod fixture;
pub mod geom;
pub mod ids;
pub mod ir;
pub mod kicad;
pub mod layout;
pub mod library;
pub mod review;
pub mod route;
pub mod spatial;
pub mod zone;
