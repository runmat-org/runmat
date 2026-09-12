pub(crate) fn map_control_flow_with_builtin(
    mut error: crate::RuntimeError,
    builtin: &str,
) -> crate::RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(builtin);
    }
    if error.identifier.is_none() {
        let identifier_segment = builtin
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' {
                    character
                } else {
                    '_'
                }
            })
            .collect::<String>();
        error.identifier = Some(format!("RunMat:{identifier_segment}:Error"));
    }
    error
}

#[cfg(test)]
mod tests {
    use super::map_control_flow_with_builtin;

    #[test]
    fn fills_missing_builtin_context_and_identifier() {
        let error = crate::build_runtime_error("failed").build();
        let mapped = map_control_flow_with_builtin(error, "object.method");
        assert_eq!(mapped.context.builtin.as_deref(), Some("object.method"));
        assert_eq!(
            mapped.identifier.as_deref(),
            Some("RunMat:object_method:Error")
        );
    }

    #[test]
    fn preserves_existing_builtin_context_and_identifier() {
        let error = crate::build_runtime_error("failed")
            .with_builtin("original")
            .with_identifier("RunMat:Original:Failure")
            .build();
        let mapped = map_control_flow_with_builtin(error, "replacement");
        assert_eq!(mapped.context.builtin.as_deref(), Some("original"));
        assert_eq!(
            mapped.identifier.as_deref(),
            Some("RunMat:Original:Failure")
        );
    }
}
