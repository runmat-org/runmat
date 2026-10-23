//! Canonical admitted empty containers for proven missing assignment receivers.
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::RuntimeError;
use runmat_types::AssignmentChildCreation;

pub fn assignment_container_in_context(
    context: Option<&RuntimeContext>,
    intent: AssignmentChildCreation,
) -> Result<RootedValue, RuntimeError> {
    match intent {
        AssignmentChildCreation::ExistingOnly => Err(crate::runtime_error::semantic_error(
            "AssignmentCreation",
            "existing-only assignment has no missing-container recipe",
        )),
        AssignmentChildCreation::NumericArray => {
            let value = crate::aggregate::structure::operations::empty_numeric_in_context(context)?;
            RootedValue::in_optional_context(value, context)
        }
        AssignmentChildCreation::CellArray => {
            crate::execution::structure::cell::from_values(context, &[0, 0], &[])
        }
        AssignmentChildCreation::ScalarStructure => {
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                context,
                std::iter::empty::<(&str, &runmat_value::Value)>(),
            )
        }
        AssignmentChildCreation::StructureArray => {
            crate::execution::structure::construct_generated_structure(
                context,
                std::iter::empty::<&str>(),
                &[0, 0],
                |_, _| unreachable!("empty field-free structure has no payload initializer"),
            )
        }
    }
}
