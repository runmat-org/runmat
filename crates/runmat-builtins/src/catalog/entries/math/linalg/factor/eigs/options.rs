//! Canonical identities and public spellings of eigs controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigsOption {
    Tolerance,
    Iterations,
    Subspace,
    Start,
    Failure,
    Display,
    Symmetric,
    Cholesky,
    Permutation,
    Definite,
}
impl EigsOption {
    pub const ALL: [Self; 10] = [
        Self::Tolerance,
        Self::Iterations,
        Self::Subspace,
        Self::Start,
        Self::Failure,
        Self::Display,
        Self::Symmetric,
        Self::Cholesky,
        Self::Permutation,
        Self::Definite,
    ];
    pub const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Tolerance => &["tolerance", "tol"],
            Self::Iterations => &["maxiterations", "maxit"],
            Self::Subspace => &["subspacedimension", "p"],
            Self::Start => &["startvector", "v0"],
            Self::Failure => &["failuretreatment", "fail"],
            Self::Display => &["display", "disp"],
            Self::Symmetric => &["isfunctionsymmetric", "issym"],
            Self::Cholesky => &["ischolesky", "cholb"],
            Self::Permutation => &["choleskypermutation", "permb"],
            Self::Definite => &["issymmetricdefinite", "spdb"],
        }
    }
    pub const fn name(self) -> &'static str {
        self.aliases()[0]
    }
    pub fn from_name(value: &str) -> Option<Self> {
        let value = value.trim();
        Self::ALL.into_iter().find(|key| {
            key.aliases()
                .iter()
                .any(|name| value.eq_ignore_ascii_case(name))
        })
    }
}
