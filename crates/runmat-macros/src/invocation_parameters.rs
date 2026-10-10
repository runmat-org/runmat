use syn::{GenericArgument, PathArguments, Type};

fn option_inner_type(ty: &Type) -> Option<&Type> {
    let Type::Path(outer) = ty else {
        return None;
    };
    let option = outer.path.segments.last()?;
    if option.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &option.arguments else {
        return None;
    };
    if arguments.args.len() != 1 {
        return None;
    }
    match arguments.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

pub(super) fn is_optional_runtime_context(ty: &Type) -> bool {
    let Some(Type::Path(inner)) = option_inner_type(ty) else {
        return false;
    };
    inner
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "RuntimeContext")
}

pub(super) fn is_optional_usize(ty: &Type) -> bool {
    let Some(Type::Path(inner)) = option_inner_type(ty) else {
        return false;
    };
    inner.path.is_ident("usize")
}

pub(super) fn is_optional_call_admission_sink(ty: &Type) -> bool {
    let Some(Type::Path(inner)) = option_inner_type(ty) else {
        return false;
    };
    inner
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "CallAdmissionSink")
}

#[cfg(test)]
mod tests {
    use super::{is_optional_call_admission_sink, is_optional_runtime_context, is_optional_usize};

    #[test]
    fn invocation_metadata_requires_structural_types() {
        let context =
            syn::parse_str::<syn::Type>("Option<crate::context::RuntimeContext>").unwrap();
        let outputs = syn::parse_str::<syn::Type>("Option<usize>").unwrap();
        let wrong_context = syn::parse_str::<syn::Type>("Option<Value>").unwrap();
        let wrong_outputs = syn::parse_str::<syn::Type>("Option<u32>").unwrap();
        assert!(is_optional_runtime_context(&context));
        assert!(is_optional_usize(&outputs));
        assert!(!is_optional_runtime_context(&wrong_context));
        assert!(!is_optional_usize(&wrong_outputs));
        let observation = syn::parse_str::<syn::Type>(
            "Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>",
        )
        .unwrap();
        assert!(is_optional_call_admission_sink(&observation));
        assert!(!is_optional_call_admission_sink(&wrong_context));
    }
}
