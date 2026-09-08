use crate::{
    BuiltinCatalogIdentity, ABS_CATALOG_ENTRY, COS_CATALOG_ENTRY, EXP_CATALOG_ENTRY,
    LDIVIDE_CATALOG_ENTRY, LOG_CATALOG_ENTRY, MINUS_CATALOG_ENTRY, PLUS_CATALOG_ENTRY,
    RDIVIDE_CATALOG_ENTRY, SIN_CATALOG_ENTRY, SQRT_CATALOG_ENTRY, TIMES_CATALOG_ENTRY,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayfunGpuCallback {
    Unary(ArrayfunGpuUnary),
    Binary(ArrayfunGpuBinary),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayfunGpuUnary {
    Sin,
    Cos,
    Abs,
    Exp,
    Log,
    Sqrt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayfunGpuBinary {
    Add,
    Subtract,
    Multiply,
    RightDivide,
    LeftDivide,
}

/// Classify a canonical builtin identity for `arrayfun`'s direct provider path.
///
/// This table belongs to the `arrayfun` contract: it says which callbacks can
/// replace element-by-element host invocation with one equivalent provider
/// operation. Runtime code receives the typed operation and does not infer
/// semantics from callable spelling.
pub fn arrayfun_gpu_callback(identity: BuiltinCatalogIdentity) -> Option<ArrayfunGpuCallback> {
    let unary = if identity == SIN_CATALOG_ENTRY.identity {
        ArrayfunGpuUnary::Sin
    } else if identity == COS_CATALOG_ENTRY.identity {
        ArrayfunGpuUnary::Cos
    } else if identity == ABS_CATALOG_ENTRY.identity {
        ArrayfunGpuUnary::Abs
    } else if identity == EXP_CATALOG_ENTRY.identity {
        ArrayfunGpuUnary::Exp
    } else if identity == LOG_CATALOG_ENTRY.identity {
        ArrayfunGpuUnary::Log
    } else if identity == SQRT_CATALOG_ENTRY.identity {
        ArrayfunGpuUnary::Sqrt
    } else {
        return binary_callback(identity).map(ArrayfunGpuCallback::Binary);
    };
    Some(ArrayfunGpuCallback::Unary(unary))
}

fn binary_callback(identity: BuiltinCatalogIdentity) -> Option<ArrayfunGpuBinary> {
    if identity == PLUS_CATALOG_ENTRY.identity {
        Some(ArrayfunGpuBinary::Add)
    } else if identity == MINUS_CATALOG_ENTRY.identity {
        Some(ArrayfunGpuBinary::Subtract)
    } else if identity == TIMES_CATALOG_ENTRY.identity {
        Some(ArrayfunGpuBinary::Multiply)
    } else if identity == RDIVIDE_CATALOG_ENTRY.identity {
        Some(ArrayfunGpuBinary::RightDivide)
    } else if identity == LDIVIDE_CATALOG_ENTRY.identity {
        Some(ArrayfunGpuBinary::LeftDivide)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ARRAYFUN_CATALOG_ENTRY;

    #[test]
    fn classifies_canonical_identities_without_name_dispatch() {
        assert_eq!(
            arrayfun_gpu_callback(SIN_CATALOG_ENTRY.identity),
            Some(ArrayfunGpuCallback::Unary(ArrayfunGpuUnary::Sin))
        );
        assert_eq!(
            arrayfun_gpu_callback(LDIVIDE_CATALOG_ENTRY.identity),
            Some(ArrayfunGpuCallback::Binary(ArrayfunGpuBinary::LeftDivide))
        );
        assert_eq!(arrayfun_gpu_callback(ARRAYFUN_CATALOG_ENTRY.identity), None);
    }
}
