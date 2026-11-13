//! Exact machine-lattice refinement identity without rounding a logarithm.
use super::super::invalid;
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_traits::One;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Level(u16);
impl Level {
    /// Largest q=2^ell with the actual positive dyadic mesh*q*q<=1.
    /// The physical mesh is NOT rounded or snapped to this integer identity.
    pub fn for_mesh(mesh: f64) -> BuiltinResult<Self> {
        if !mesh.is_finite() || mesh <= 0.0 || mesh > 1.0 {
            return Err(invalid("MADS mesh must be finite positive and at most one"));
        }
        let bits = mesh.to_bits();
        let stored = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let (exponent, power) = if stored == 0 {
            (
                (63 - fraction.leading_zeros() as i32) - 1074,
                fraction.is_power_of_two(),
            )
        } else {
            (stored - 1023, fraction == 0)
        };
        let mut level = (-exponent) / 2;
        if exponent % 2 == 0 && !power {
            level -= 1;
        }
        Ok(Self(u16::try_from(level).map_err(|_| {
            invalid("MADS refinement exponent is invalid")
        })?))
    }
    pub fn bits(self) -> usize {
        usize::from(self.0)
    }
    pub fn integer(self) -> BigInt {
        BigInt::one() << self.bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::ToPrimitive;
    #[test]
    fn arbitrary_initial_mesh_keeps_original_dyadic_scale_and_exact_frame_bound() {
        for mesh in [
            1.0,
            0.5,
            0.3,
            0.25,
            0.1,
            0.0625,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::from_bits(3),
        ] {
            let level = Level::for_mesh(mesh).unwrap();
            let q = level.integer();
            let exact = BigRational::from_float(mesh).unwrap();
            let bound = &exact * BigRational::from_integer(&q * &q);
            assert!(bound <= BigRational::one());
            assert!(&bound * BigRational::from_integer(BigInt::from(4)) > BigRational::one());
            assert_eq!(
                BigRational::from_float(q.to_f64().unwrap()).unwrap(),
                BigRational::from_integer(q)
            );
        }
        assert_eq!(Level::for_mesh(0.1).unwrap().integer(), BigInt::from(2));
        assert_eq!(Level::for_mesh(f64::from_bits(1)).unwrap().bits(), 537);
        assert!(Level::for_mesh(0.0).is_err());
        assert!(Level::for_mesh(f64::INFINITY).is_err());
    }
}
