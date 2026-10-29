//! Closed identities for the GLSL.std.450 operations which exclude binary64.

use naga::{MathFunction as Math, Scalar, ScalarKind, TypeInner, VectorSize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Intrinsic {
    Radians,
    Degrees,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Exp,
    Exp2,
    Log,
    Log2,
    Pow,
}

impl Intrinsic {
    pub(crate) fn from_math(fun: Math) -> Option<Self> {
        Some(match fun {
            Math::Radians => Self::Radians,
            Math::Degrees => Self::Degrees,
            Math::Sin => Self::Sin,
            Math::Cos => Self::Cos,
            Math::Tan => Self::Tan,
            Math::Asin => Self::Asin,
            Math::Acos => Self::Acos,
            Math::Atan => Self::Atan,
            Math::Atan2 => Self::Atan2,
            Math::Sinh => Self::Sinh,
            Math::Cosh => Self::Cosh,
            Math::Tanh => Self::Tanh,
            Math::Asinh => Self::Asinh,
            Math::Acosh => Self::Acosh,
            Math::Atanh => Self::Atanh,
            Math::Exp => Self::Exp,
            Math::Exp2 => Self::Exp2,
            Math::Log => Self::Log,
            Math::Log2 => Self::Log2,
            Math::Pow => Self::Pow,
            Math::Abs
            | Math::Min
            | Math::Max
            | Math::Clamp
            | Math::Saturate
            | Math::Ceil
            | Math::Floor
            | Math::Round
            | Math::Fract
            | Math::Trunc
            | Math::Modf
            | Math::Frexp
            | Math::Ldexp
            | Math::Dot
            | Math::Outer
            | Math::Cross
            | Math::Distance
            | Math::Length
            | Math::Normalize
            | Math::FaceForward
            | Math::Reflect
            | Math::Refract
            | Math::Sign
            | Math::Fma
            | Math::Mix
            | Math::Step
            | Math::SmoothStep
            | Math::Sqrt
            | Math::InverseSqrt
            | Math::Inverse
            | Math::Transpose
            | Math::Determinant
            | Math::QuantizeToF16
            | Math::CountTrailingZeros
            | Math::CountLeadingZeros
            | Math::CountOneBits
            | Math::ReverseBits
            | Math::ExtractBits
            | Math::InsertBits
            | Math::FirstTrailingBit
            | Math::FirstLeadingBit
            | Math::Pack4x8snorm
            | Math::Pack4x8unorm
            | Math::Pack2x16snorm
            | Math::Pack2x16unorm
            | Math::Pack2x16float
            | Math::Pack4xI8
            | Math::Pack4xU8
            | Math::Unpack4x8snorm
            | Math::Unpack4x8unorm
            | Math::Unpack2x16snorm
            | Math::Unpack2x16unorm
            | Math::Unpack2x16float
            | Math::Unpack4xI8
            | Math::Unpack4xU8 => return None,
        })
    }

    pub(crate) const fn spelling(self) -> &'static str {
        match self {
            Self::Radians => "radians",
            Self::Degrees => "degrees",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Pow => "pow",
        }
    }

    pub(crate) const fn arguments(self) -> usize {
        match self {
            Self::Pow | Self::Atan2 => 2,
            Self::Radians
            | Self::Degrees
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Asinh
            | Self::Acosh
            | Self::Atanh
            | Self::Exp
            | Self::Exp2
            | Self::Log
            | Self::Log2 => 1,
        }
    }

    pub(crate) fn helper(self, shape: Shape) -> HelperName {
        HelperName(self, shape)
    }
}

pub(crate) struct HelperName(Intrinsic, Shape);
impl std::fmt::Display for HelperName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "runmat_math64_{}", self.0.spelling())?;
        if let Some(width) = self.1.vector_width() {
            write!(f, "_vec{width}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Shape {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
}

impl Shape {
    pub(crate) fn of(inner: &TypeInner) -> Option<Self> {
        let double = Scalar {
            kind: ScalarKind::Float,
            width: 8,
        };
        match *inner {
            TypeInner::Scalar(scalar) if scalar == double => Some(Self::Scalar),
            TypeInner::Vector { size, scalar } if scalar == double => Some(match size {
                VectorSize::Bi => Self::Vec2,
                VectorSize::Tri => Self::Vec3,
                VectorSize::Quad => Self::Vec4,
            }),
            _ => None,
        }
    }

    pub(crate) const fn vector_width(self) -> Option<usize> {
        match self {
            Self::Scalar => None,
            Self::Vec2 => Some(2),
            Self::Vec3 => Some(3),
            Self::Vec4 => Some(4),
        }
    }
}
