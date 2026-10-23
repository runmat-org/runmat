use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatetimeBinaryOperation {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
}
impl DatetimeBinaryOperation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Eq => "datetime.eq",
            Self::Ne => "datetime.ne",
            Self::Lt => "datetime.lt",
            Self::Le => "datetime.le",
            Self::Gt => "datetime.gt",
            Self::Ge => "datetime.ge",
            Self::Plus => "datetime.plus",
            Self::Minus => "datetime.minus",
        }
    }
    pub const fn method(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Plus => "plus",
            Self::Minus => "minus",
        }
    }
    pub const fn is_comparison(self) -> bool {
        !matches!(self, Self::Plus | Self::Minus)
    }
    pub const fn catalog_entry(self) -> &'static BuiltinCatalogEntry {
        match self {
            Self::Eq => &DATETIME_EQ_CATALOG_ENTRY,
            Self::Ne => &DATETIME_NE_CATALOG_ENTRY,
            Self::Lt => &DATETIME_LT_CATALOG_ENTRY,
            Self::Le => &DATETIME_LE_CATALOG_ENTRY,
            Self::Gt => &DATETIME_GT_CATALOG_ENTRY,
            Self::Ge => &DATETIME_GE_CATALOG_ENTRY,
            Self::Plus => &DATETIME_PLUS_CATALOG_ENTRY,
            Self::Minus => &DATETIME_MINUS_CATALOG_ENTRY,
        }
    }
    pub const fn argument_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Eq => &DATETIME_EQ_ERROR_INVALID_ARGUMENT,
            Self::Ne => &DATETIME_NE_ERROR_INVALID_ARGUMENT,
            Self::Lt => &DATETIME_LT_ERROR_INVALID_ARGUMENT,
            Self::Le => &DATETIME_LE_ERROR_INVALID_ARGUMENT,
            Self::Gt => &DATETIME_GT_ERROR_INVALID_ARGUMENT,
            Self::Ge => &DATETIME_GE_ERROR_INVALID_ARGUMENT,
            Self::Plus => &DATETIME_PLUS_ERROR_INVALID_ARGUMENT,
            Self::Minus => &DATETIME_MINUS_ERROR_INVALID_ARGUMENT,
        }
    }
    pub const fn output_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Eq => &DATETIME_EQ_ERROR_TOO_MANY_OUTPUTS,
            Self::Ne => &DATETIME_NE_ERROR_TOO_MANY_OUTPUTS,
            Self::Lt => &DATETIME_LT_ERROR_TOO_MANY_OUTPUTS,
            Self::Le => &DATETIME_LE_ERROR_TOO_MANY_OUTPUTS,
            Self::Gt => &DATETIME_GT_ERROR_TOO_MANY_OUTPUTS,
            Self::Ge => &DATETIME_GE_ERROR_TOO_MANY_OUTPUTS,
            Self::Plus => &DATETIME_PLUS_ERROR_TOO_MANY_OUTPUTS,
            Self::Minus => &DATETIME_MINUS_ERROR_TOO_MANY_OUTPUTS,
        }
    }
}
