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
}

#[derive(Clone, Copy)]
pub enum UnaryOpCode {
    Sin = 0,
    Cos = 1,
    Abs = 2,
    Exp = 3,
    Log = 4,
    Sqrt = 5,
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
    Min = 1,
    Max = 2,
}

#[derive(Clone, Copy)]
pub enum DimReduceOp {
    Sum = 0,
    Mean = 1,
}

#[derive(Clone, Copy)]
pub enum DimReduceExtrema {
    Min = 0,
    Max = 1,
}
