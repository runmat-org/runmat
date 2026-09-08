#[derive(Debug)]
pub(crate) enum UniformScalarError {
    Heterogeneous,
    NonScalar,
    InvalidStorage(&'static str),
    Materialization(String),
    CharacterRank,
    CharacterCount,
    SizeOverflow,
}
