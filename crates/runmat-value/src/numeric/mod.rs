use serde::{Deserialize, Serialize};

mod dtype;
mod complex_division;
mod integer;
mod scalar;
mod storage;
#[cfg(test)]
mod tests;
mod view;

pub use complex_division::{divide_f32 as divide_complex_f32, divide_f64 as divide_complex_f64};
pub use dtype::NumericDType;
pub use integer::IntegerStorage;
pub use scalar::{IntValue, NumericScalar};
pub use storage::NumericStorage;
pub use view::{NumericStorageView, NumericStorageViewMut};
