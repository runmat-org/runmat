//! One declaration supplies numerical entries and typed class method bindings.
use super::*;
use crate::*;
use runmat_types::StaticMethodName;
macro_rules! methods {
    ($(($entry:ident,$method:literal,$slug:literal)),*$(,)?)=>{
        $(pub const $entry:BuiltinCatalogEntry=entry::entry(
            concat!("optim.problemdef.OptimizationExpression.",$method),
            &descriptor::VARIADIC,
            documentation::method(concat!("OptimizationExpression.",$method),$slug),
            inference::expression,
        );)*
        pub const OPTIMMODEL_NUMERICAL_METHODS:&[(StaticMethodName,&BuiltinCatalogEntry)]=&[
            $((StaticMethodName::new($method),&$entry)),*
        ];
    }
}
methods! {
    (OPTIMEXPR_SUM_CATALOG_ENTRY,"sum","optimization-expression-sum"),
    (OPTIMEXPR_MEAN_CATALOG_ENTRY,"mean","optimization-expression-mean"),
    (OPTIMEXPR_PROD_CATALOG_ENTRY,"prod","optimization-expression-prod"),
    (OPTIMEXPR_TRACE_CATALOG_ENTRY,"trace","optimization-expression-trace"),
    (OPTIMEXPR_CUMSUM_CATALOG_ENTRY,"cumsum","optimization-expression-cumsum"),
    (OPTIMEXPR_CUMPROD_CATALOG_ENTRY,"cumprod","optimization-expression-cumprod"),
    (OPTIMEXPR_DIFF_CATALOG_ENTRY,"diff","optimization-expression-diff"),
    (OPTIMEXPR_RESHAPE_CATALOG_ENTRY,"reshape","optimization-expression-reshape"),
    (OPTIMEXPR_DIAG_CATALOG_ENTRY,"diag","optimization-expression-diag"),
    (OPTIMEXPR_DOT_CATALOG_ENTRY,"dot","optimization-expression-dot"),
    (OPTIMEXPR_NORM_CATALOG_ENTRY,"norm","optimization-expression-norm"),
    (OPTIMEXPR_SQRT_CATALOG_ENTRY,"sqrt","optimization-expression-sqrt"),
    (OPTIMEXPR_EXP_CATALOG_ENTRY,"exp","optimization-expression-exp"),
    (OPTIMEXPR_LOG_CATALOG_ENTRY,"log","optimization-expression-log"),
    (OPTIMEXPR_SIN_CATALOG_ENTRY,"sin","optimization-expression-sin"),
    (OPTIMEXPR_COS_CATALOG_ENTRY,"cos","optimization-expression-cos"),
    (OPTIMEXPR_TAN_CATALOG_ENTRY,"tan","optimization-expression-tan"),
    (OPTIMEXPR_SEC_CATALOG_ENTRY,"sec","optimization-expression-sec"),
    (OPTIMEXPR_CSC_CATALOG_ENTRY,"csc","optimization-expression-csc"),
    (OPTIMEXPR_COT_CATALOG_ENTRY,"cot","optimization-expression-cot"),
    (OPTIMEXPR_ASIN_CATALOG_ENTRY,"asin","optimization-expression-asin"),
    (OPTIMEXPR_ACOS_CATALOG_ENTRY,"acos","optimization-expression-acos"),
    (OPTIMEXPR_ATAN_CATALOG_ENTRY,"atan","optimization-expression-atan"),
    (OPTIMEXPR_ASEC_CATALOG_ENTRY,"asec","optimization-expression-asec"),
    (OPTIMEXPR_ACSC_CATALOG_ENTRY,"acsc","optimization-expression-acsc"),
    (OPTIMEXPR_ACOT_CATALOG_ENTRY,"acot","optimization-expression-acot"),
    (OPTIMEXPR_SINH_CATALOG_ENTRY,"sinh","optimization-expression-sinh"),
    (OPTIMEXPR_COSH_CATALOG_ENTRY,"cosh","optimization-expression-cosh"),
    (OPTIMEXPR_TANH_CATALOG_ENTRY,"tanh","optimization-expression-tanh"),
    (OPTIMEXPR_SECH_CATALOG_ENTRY,"sech","optimization-expression-sech"),
    (OPTIMEXPR_CSCH_CATALOG_ENTRY,"csch","optimization-expression-csch"),
    (OPTIMEXPR_COTH_CATALOG_ENTRY,"coth","optimization-expression-coth"),
    (OPTIMEXPR_ASINH_CATALOG_ENTRY,"asinh","optimization-expression-asinh"),
    (OPTIMEXPR_ACOSH_CATALOG_ENTRY,"acosh","optimization-expression-acosh"),
    (OPTIMEXPR_ATANH_CATALOG_ENTRY,"atanh","optimization-expression-atanh"),
    (OPTIMEXPR_ASECH_CATALOG_ENTRY,"asech","optimization-expression-asech"),
    (OPTIMEXPR_ACSCH_CATALOG_ENTRY,"acsch","optimization-expression-acsch"),
    (OPTIMEXPR_ACOTH_CATALOG_ENTRY,"acoth","optimization-expression-acoth"),
    (OPTIMEXPR_CAT_CATALOG_ENTRY,"cat","optimization-expression-cat"),
    (OPTIMEXPR_HORZCAT_CATALOG_ENTRY,"horzcat","optimization-expression-horzcat"),
    (OPTIMEXPR_VERTCAT_CATALOG_ENTRY,"vertcat","optimization-expression-vertcat"),
}
