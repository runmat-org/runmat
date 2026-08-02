use crate::HirAssembly;
use runmat_test::plan::ProgramRevision;

pub struct SemanticTestSource<'a> {
    pub owner_identity: &'a str,
    pub relative_source_identity: &'a str,
    pub source_text: &'a str,
    pub assembly: &'a HirAssembly,
}

pub struct SemanticDiscoveryInput<'a> {
    pub program_revision: ProgramRevision,
    pub sources: &'a [SemanticTestSource<'a>],
}
