//! Public FEA syntax entrypoint, shared with frontend-free FEA consumers.

pub use runmat_design_syntax::*;

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    #[test]
    fn public_fea_entrypoints_share_the_canonical_syntax_types() {
        assert_eq!(
            TypeId::of::<super::FeaSyntaxModule>(),
            TypeId::of::<runmat_design_syntax::FeaSyntaxModule>()
        );
        assert_eq!(
            TypeId::of::<runmat_lexer::fea::FeaTokenStream>(),
            TypeId::of::<runmat_design_syntax::lexical::FeaTokenStream>()
        );
    }
}
