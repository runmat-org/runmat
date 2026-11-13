//! PCG's native vector class on top of shared iterative value ownership.

use runmat_value::ComplexElement;

use super::super::super::iterative::value::IterativeValue;
use super::{NativePcgVector, VectorClass};

pub(super) use super::super::super::iterative::value::vector_shape;

pub(super) trait PcgValue: IterativeValue + NativePcgVector {
    const VECTOR_CLASS: VectorClass;
}

macro_rules! pcg_class {
    ($scalar:ty, $class:expr) => {
        impl PcgValue for $scalar {
            const VECTOR_CLASS: VectorClass = $class;
        }
    };
}

pcg_class!(f32, VectorClass::RealSingle);
pcg_class!(f64, VectorClass::RealDouble);
pcg_class!(ComplexElement<f32>, VectorClass::ComplexSingle);
pcg_class!(ComplexElement<f64>, VectorClass::ComplexDouble);
