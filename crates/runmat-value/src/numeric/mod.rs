use serde::{Deserialize, Serialize};

mod dtype;
mod integer;
mod scalar;
mod storage;
#[cfg(test)]
mod tests;
mod view;

pub use dtype::NumericDType;
pub use integer::IntegerStorage;
pub use scalar::{IntValue, NumericScalar};
pub use storage::NumericStorage;
pub use view::{NumericStorageView, NumericStorageViewMut};
