//! Physical quantity, unit and source-evidence authority shared by every frontend and solver.
//! Numeric buffers carry their quantity once at the boundary, never once per element.
mod catalog;
mod quantity;
pub use catalog::*;
pub use quantity::*;
mod unit;
mod parse;
pub use parse::{parse_unit_expecting, parse_unit_expression, UnitExpressionError};
pub use unit::*;
mod evidence;
pub use evidence::*;
mod arithmetic;
#[cfg(test)]
mod tests;
mod source_quantity;
pub use source_quantity::*;

pub mod canonical_scalar;

mod dynamic;
pub use dynamic::*;

mod field;
pub use field::*;

pub mod canonical_storage;

mod operator;
pub use operator::*;

mod constants;
pub use constants::*;

mod expression;
pub use expression::*;

mod derivative;
pub use derivative::*;

mod unit_message;
pub use unit_message::*;
mod unit_names;
pub use unit_names::*;

mod parameter_coordinate;
pub use parameter_coordinate::*;

/// How results are printed: legend range precision, printed unit symbols, and the line that
/// states an assessed result against its limit ("Peak 76.3 MPa (element) · limit 184 MPa ·
/// factor 2.41"). Legends, published assessments and the Desktop all read it from here.
pub mod result_text;

/// The shared material library (ids, labels, properties and their sources).
pub mod materials;
