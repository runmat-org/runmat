use runmat_types::{ClassIdentity, StaticClassIdentity};

/// A class or MATLAB abstract category decoded once at a type-query boundary.
/// Concrete custom-class spelling is retained; only comparisons use the
/// public `isa` compatibility rule for ASCII case-insensitive class names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum IsaTypeDesignator {
    Numeric,
    Float,
    Integer,
    Logical,
    Character,
    String,
    Cell,
    Struct,
    FunctionHandle,
    GpuArray,
    Listener,
    MetaClass,
    Exception,
    Handle,
    Concrete(ClassIdentity),
}

impl IsaTypeDesignator {
    pub(super) fn parse(source: &str) -> Option<Self> {
        let source = source.trim();
        if source.is_empty() {
            return None;
        }
        let folded = source.to_ascii_lowercase();
        Some(match folded.as_str() {
            "numeric" => Self::Numeric,
            "float" => Self::Float,
            "integer" => Self::Integer,
            "logical" => Self::Logical,
            "char" => Self::Character,
            "string" => Self::String,
            "cell" => Self::Cell,
            "struct" => Self::Struct,
            "function_handle" => Self::FunctionHandle,
            "gpuarray" => Self::GpuArray,
            "listener" | "event.listener" => Self::Listener,
            "meta.class" => Self::MetaClass,
            "mexception" => Self::Exception,
            "handle" => Self::Handle,
            _ => Self::Concrete(ClassIdentity::new(source).ok()?),
        })
    }

    pub(super) fn concrete_is(&self, expected: StaticClassIdentity) -> bool {
        match self {
            Self::GpuArray => expected == runmat_types::standard::GPU_ARRAY,
            Self::Listener => expected == runmat_types::standard::EVENT_LISTENER,
            Self::MetaClass => expected == runmat_types::standard::META_CLASS,
            Self::Exception => expected == runmat_types::standard::MEXCEPTION,
            Self::Concrete(identity) => identity.eq_ignore_ascii_case(&expected.owned()),
            _ => false,
        }
    }

    #[cfg(test)]
    fn concrete(&self) -> Option<&ClassIdentity> {
        match self {
            Self::Concrete(identity) => Some(identity),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::standard;

    #[test]
    fn parser_preserves_custom_identity_and_public_category_aliases() {
        assert_eq!(
            IsaTypeDesignator::parse(" NUMERIC "),
            Some(IsaTypeDesignator::Numeric)
        );
        assert!(IsaTypeDesignator::parse(" listener ")
            .is_some_and(|type_name| type_name.concrete_is(standard::EVENT_LISTENER)));
        assert_eq!(
            IsaTypeDesignator::parse("Pkg.Widget")
                .and_then(|type_name| type_name.concrete().cloned()),
            ClassIdentity::new("Pkg.Widget").ok()
        );
        assert!(IsaTypeDesignator::parse(" ").is_none());
        assert!(IsaTypeDesignator::parse("Pkg..Widget").is_none());
    }
}
