//! `runmat-circuit`: the core library of RunMat ECAD.
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
//! - [`schematic`]: the schematic as drawn, a page per sheet instance with placed symbols,
//!   wires, junctions and labels.
//!
//! Physical design:
//! - [`geom`]: integer-nanometre geometry, exact polygon booleans and offsets, primitives.
//! - [`layout`]: outline, stackup and rigid-flex regions, placement, tracks, vias, zones,
//!   keepouts, rooms, text, dimensions.
//! - [`board_rules`]: the board as a `runmat-rules` item graph; rule properties.
//! - [`zone`]: copper pour fill (clearances, thermal spokes, minimum width, islands).
//! - [`spatial`]: grid index; [`connectivity`]: physical copper islands.
//! - [`drc`]: the incremental DRC kernel.
//! - [`eco`]: engineering change orders: schematic ↔ board sync as a typed, validated, undoable
//!   change list with cause and board impact, back-annotation and agent change sets.
//! - [`room_formats`]: Copy Room Formats, one channel's placement and routing replicated into
//!   the other channels' rooms.
//! - [`route`]: ratsnest, any-angle walkaround, checked insertion, tuning lengths.
//! - [`router`]: interactive-routing primitives: 45°/90° shaping, mitred and arc corners,
//!   clearance hulls, shortest and hugging walkaround.
//!
//! Interchange and outputs:
//! - [`kicad`]: clean-room, lossless KiCad 8/9/10 board, footprint, schematic, symbol and
//!   rules files.
//! - [`fab`]: Gerber X2, Excellon, IPC-D-356, BOM, pick-and-place; [`font`]: the stroke
//!   font text is drawn with; [`ipc2581`]: IPC-2581 revision C. ODB++ is not written yet:
//!   its directory format (matrix, per-layer feature files, EDA data, netlists) needs the
//!   ODB++ specification, which Siemens distributes on registration and which this
//!   clean-room implementation has not used; IPC-2581 carries the same data set.
//! - [`review`]: design-review suites with typed fixes.
//! - [`fea`]: the board → FEA `pcb` source contract.
//! - [`display`]: board display geometry (pads, tracks, pours, outline, keepouts, text,
//!   dimensions, ratsnest) for the projection.
//! - [`drawn`]: board commands drawn in the board view (route, via, pour, keepout, room,
//!   outline, dimension) from canvas point picks.
//! - [`runtime`]: the document runtime the Desktop and browser call: compile, review and edit
//!   a circuit's files, with minimal text edits.
//! - [`fixture`]: a standard pool and a large-board generator for tests and benchmarks.
//!
//! Everything is deterministic: ordered maps, sorted outputs, exact integer predicates.

pub mod autoroute;
pub mod board3d;
pub mod board_rules;
pub mod commands;
pub mod connectivity;
pub mod designators;
pub mod display;
pub mod drawn;
pub mod drc;
pub mod eco;
pub mod elab;
pub mod erc;
pub mod fab;
pub mod fea;
pub mod fixture;
pub mod font;
pub mod geom;
pub mod ids;
pub mod index;
pub mod ipc2581;
pub mod ir;
pub mod kicad;
pub mod layout;
pub mod library;
pub mod review;
pub mod room_formats;
pub mod runtime;
pub mod schematic;
pub mod simulation;
pub mod route;
pub mod router;
pub mod spatial;
pub mod syntax;
pub mod zone;
