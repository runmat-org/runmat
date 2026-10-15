//! Physical equivalent of Double-scalar × Single data, not a semantic cast.
//! Finite binary32 factors have an exact binary64 product (at most 48 bits).
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct ExactBinary32Product(u64);

impl ExactBinary32Product {
    pub fn for_double_bits(bits: u64) -> Option<Self> {
        let value = f64::from_bits(bits);
        (value.is_finite() && f64::from(value as f32).to_bits() == bits).then_some(Self(bits))
    }
    pub fn original_double_bits(self) -> u64 {
        self.0
    }
    pub fn binary32_bits(self) -> u32 {
        (f64::from_bits(self.0) as f32).to_bits()
    }
    pub fn evaluate(self, base: f32) -> f32 {
        (f64::from(base) * f64::from_bits(self.0)) as f32
    }
}
