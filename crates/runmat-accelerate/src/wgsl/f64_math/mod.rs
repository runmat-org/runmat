//! Typed, source-preserving binary64 math legalization, independent of a device.

mod intrinsic;
mod legalize;
#[cfg(test)]
mod tests;

pub(crate) use legalize::prepare;
#[cfg(test)]
pub(crate) use legalize::PreparationError;

/// Original source plus this compiler domain is the pipeline identity. Change
/// when helper algorithms, legalization or arithmetic controls change.
#[cfg(feature = "wgpu")]
pub(crate) const COMPILER_REVISION: u32 = 2;

use intrinsic::Intrinsic;

// Family definitions are kept separate from typed call rewriting.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Family {
    Words,
    Scaling,
    Exp,
    Log,
    Exp2,
    Log2,
    Expm1,
    Log1p,
    Pow,
    Hyperbolic,
    InverseTrig,
    RangeReduction,
    TrigKernels,
    Trigonometry,
    Conversions,
}

impl Family {
    fn source(self) -> &'static str {
        match self {
            Self::Words => super::ieee_words::helpers(),
            Self::Scaling => include_str!("scaling.wgsl"),
            Self::Exp => include_str!("exp.wgsl"),
            Self::Log => include_str!("log.wgsl"),
            Self::Exp2 => include_str!("exp2.wgsl"),
            Self::Log2 => include_str!("log2.wgsl"),
            Self::Expm1 => include_str!("expm1.wgsl"),
            Self::Log1p => include_str!("log1p.wgsl"),
            Self::Pow => include_str!("pow.wgsl"),
            Self::Hyperbolic => include_str!("hyperbolic.wgsl"),
            Self::InverseTrig => include_str!("inverse_trig.wgsl"),
            Self::RangeReduction => include_str!("range_reduction.wgsl"),
            Self::TrigKernels => include_str!("trig_kernels.wgsl"),
            Self::Trigonometry => include_str!("trigonometry.wgsl"),
            Self::Conversions => include_str!("conversions.wgsl"),
        }
    }
}

const FAMILIES: [Family; 15] = [
    Family::Words,
    Family::Scaling,
    Family::Exp,
    Family::Log,
    Family::Exp2,
    Family::Log2,
    Family::Expm1,
    Family::Log1p,
    Family::Pow,
    Family::Hyperbolic,
    Family::InverseTrig,
    Family::RangeReduction,
    Family::TrigKernels,
    Family::Trigonometry,
    Family::Conversions,
];

#[derive(Default)]
struct FamilySet([bool; FAMILIES.len()]);
impl FamilySet {
    fn insert(&mut self, family: Family) {
        self.0[family as usize] = true;
    }
    fn extend(&mut self, families: impl IntoIterator<Item = Family>) {
        for family in families {
            self.insert(family);
        }
    }
    fn iter(&self) -> impl Iterator<Item = Family> + '_ {
        FAMILIES
            .into_iter()
            .filter(|family| self.0[*family as usize])
    }
}

fn include_family(intrinsic: Intrinsic, families: &mut FamilySet) {
    families.insert(Family::Words);
    if !matches!(intrinsic, Intrinsic::Radians | Intrinsic::Degrees) {
        families.insert(Family::Scaling);
    }
    match intrinsic {
        Intrinsic::Exp => {
            families.insert(Family::Exp);
        }
        Intrinsic::Log => {
            families.insert(Family::Log);
        }
        Intrinsic::Exp2 => {
            families.insert(Family::Exp2);
        }
        Intrinsic::Log2 => {
            families.insert(Family::Log2);
        }
        Intrinsic::Pow => {
            families.insert(Family::Pow);
        }
        Intrinsic::Sin | Intrinsic::Cos | Intrinsic::Tan => {
            families.extend([
                Family::RangeReduction,
                Family::TrigKernels,
                Family::Trigonometry,
            ]);
        }
        Intrinsic::Radians | Intrinsic::Degrees => {
            families.insert(Family::Conversions);
        }
        Intrinsic::Asin | Intrinsic::Acos | Intrinsic::Atan | Intrinsic::Atan2 => {
            families.insert(Family::InverseTrig);
        }
        Intrinsic::Sinh
        | Intrinsic::Cosh
        | Intrinsic::Tanh
        | Intrinsic::Asinh
        | Intrinsic::Acosh
        | Intrinsic::Atanh => {
            families.extend([
                Family::Exp,
                Family::Log,
                Family::Expm1,
                Family::Log1p,
                Family::Hyperbolic,
            ]);
        }
    }
}
