//! Fixture generators for tests, benchmarks and examples.
//!
//! - [`standard_pool`]: a small but real pool: resistor, capacitor, inductor, LED, a
//!   quad op-amp with a power gate (multi-gate entity), a buck regulator, a USB-C receptacle,
//!   an I²C sensor, a crystal, an MCU in QFN-32, and a BGA FPGA.
//! - [`large`]: a realistic large design and board: `channels` repeated analog channels
//!   (hierarchical sheet, bus-indexed bindings, local and global nets), a BGA processor with a
//!   wide bus, 8 or more copper layers, thousands of nets, placed and partly routed, with
//!   ground and power pours. Routing is clean by construction: dog-bone fanouts (BGA balls to
//!   the diagonal), horizontal and vertical inner layers joined by buried vias, every object
//!   checked against clearances before it is placed; what does not fit stays an airwire.
//!   [`seed_violations`] then adds a known number of violations. Generation is
//!   deterministic for a given [`LargeParams`].
//! - [`eco_scenario`]: a board synced with a two-channel design, edited on the board side,
//!   and the schematic since edited every way an ECO handles.

mod eco;
mod large;
mod pool;
mod pool_daq;
mod pool_drone;
mod project;

pub use eco::*;
pub use large::*;
pub use pool::*;
pub use project::*;
