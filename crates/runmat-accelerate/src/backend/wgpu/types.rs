#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericPrecision {
    F32,
    F64,
}

#[derive(Clone, Copy)]
pub enum BinaryOpCode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Hypot = 4,
    Atan2 = 5,
}

#[derive(Clone, Copy)]
pub enum UnaryOpCode {
    Sin = 0,
    Cos = 1,
    Abs = 2,
    Exp = 3,
    Log = 4,
    Sqrt = 5,
    Sign = 6,
    Real = 7,
    Imag = 8,
    Conj = 9,
    Angle = 10,
    Expm1 = 11,
    Log1p = 12,
    Log10 = 13,
    Log2 = 14,
    Pow2 = 15,
    Floor = 16,
    Ceil = 17,
    Fix = 18,
    Tan = 19,
    Asin = 20,
    Acos = 21,
    Atan = 22,
    Sinh = 23,
    Cosh = 24,
    Tanh = 25,
    Asinh = 26,
    Acosh = 27,
    Atanh = 28,
}

#[derive(Clone, Copy)]
pub enum ScalarOpCode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    RSub = 4,
    RDiv = 5,
}

#[derive(Clone, Copy)]
pub enum GlobalReduceOp {
    Sum = 0,
    Prod = 1,
    Min = 2,
    Max = 3,
    CountNonZero = 4,
}

#[derive(Clone, Copy)]
pub enum DimReduceOp {
    Sum = 0,
    Mean = 1,
    Prod = 2,
    AnyInclude = 3,
    AnyOmit = 4,
    AllInclude = 5,
    AllOmit = 6,
    CountNonZero = 7,
}

#[derive(Clone, Copy)]
pub enum DimReduceExtrema {
    Min = 0,
    Max = 1,
}
