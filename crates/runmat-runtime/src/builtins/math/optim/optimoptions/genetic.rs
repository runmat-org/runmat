//! Cooperative checks for GA fields selected by the existing constructor grammar.
use super::*;
pub(super) async fn validate_values(
    gathered: &crate::call::BuiltinArguments,
    roles: &preflight::ArgumentRoles<'_>,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<()> {
    if gathered.len() != roles.len() {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
            "optimoptions gathered argument-role cardinality changed",
        ));
    }
    for index in 0..gathered.len() {
        match roles.get(index)? {
            preflight::ArgumentRole::Option {
                solver: Solver::Ga,
                field,
            } => {
                crate::builtins::math::optim::ga::options::admission::validate_cooperative(
                    crate::builtins::math::optim::ga::options::Owner::Constructor,
                    field,
                    &gathered[index],
                    cooperation,
                )
                .await?
            }
            preflight::ArgumentRole::Structure { solver: Solver::Ga } => {
                let Value::Struct(options) = &gathered[index] else {
                    return Err(optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
                        "options role no longer has a structure",
                    ));
                };
                for (name, value) in &options.fields {
                    if name.eq_ignore_ascii_case("Solver") {
                        continue;
                    }
                    let field = admitted_option_field(Solver::Ga, name)?;
                    crate::builtins::math::optim::ga::options::admission::validate_cooperative(
                        crate::builtins::math::optim::ga::options::Owner::Constructor,
                        field,
                        value,
                        cooperation,
                    )
                    .await?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
