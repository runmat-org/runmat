//! Canonical selected operation identities shared by affine proposals and proof.
//! Public text is resolved through the existing catalog; no second spelling table.
use runmat_builtins::BuiltinCatalogIdentity;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Operation {
    Sum,
    Mean,
    Trace,
    Cumsum,
    Product,
    Difference,
    Dot,
    Cumprod,
    Diagonal,
    Reshape,
}
impl Operation {
    pub(super) fn selected(name: &str) -> Option<Self> {
        let identity = runmat_builtins::builtin_catalog_entry_by_name(name)?.identity;
        Self::identity(identity)
    }
    fn identity(identity: BuiltinCatalogIdentity) -> Option<Self> {
        use runmat_builtins::*;
        if identity == SUM_CATALOG_ENTRY.identity {
            Some(Self::Sum)
        } else if identity == MEAN_CATALOG_ENTRY.identity {
            Some(Self::Mean)
        } else if identity == TRACE_CATALOG_ENTRY.identity {
            Some(Self::Trace)
        } else if identity == CUMSUM_CATALOG_ENTRY.identity {
            Some(Self::Cumsum)
        } else if identity == DOT_CATALOG_ENTRY.identity {
            Some(Self::Dot)
        } else if identity == DIFF_CATALOG_ENTRY.identity {
            Some(Self::Difference)
        } else if identity == PROD_CATALOG_ENTRY.identity {
            Some(Self::Product)
        } else if identity == CUMPROD_CATALOG_ENTRY.identity {
            Some(Self::Cumprod)
        } else if identity == DIAG_CATALOG_ENTRY.identity {
            Some(Self::Diagonal)
        } else if identity == RESHAPE_CATALOG_ENTRY.identity {
            Some(Self::Reshape)
        } else {
            None
        }
    }
}
