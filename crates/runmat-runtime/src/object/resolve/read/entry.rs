//! Borrowed frame admission at the same member read owner.
use super::*;
pub fn read_member_sequence_with_context<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    read_member_sequence_with_caller(
        context,
        base,
        field,
        allow_init,
        MemberCaller::named(caller_function_name),
    )
}
pub fn read_member_sequence_text_with_context<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    read_member_sequence_text_with_caller(
        context,
        base,
        field,
        allow_init,
        MemberCaller::named(caller_function_name),
    )
}
pub fn read_member_sequence_with_access<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
    access: &'a crate::object::protocol::ObjectAccessContext,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    read_member_sequence_with_caller(
        context,
        base,
        field,
        allow_init,
        MemberCaller::frame(caller_function_name, access),
    )
}
pub fn read_member_sequence_text_with_access<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
    access: &'a crate::object::protocol::ObjectAccessContext,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    read_member_sequence_text_with_caller(
        context,
        base,
        field,
        allow_init,
        MemberCaller::frame(caller_function_name, access),
    )
}
pub fn load_member_with_access<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
    access: &'a crate::object::protocol::ObjectAccessContext,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    futures::FutureExt::map(
        read_member_sequence_with_access(
            context,
            base,
            field,
            allow_init,
            caller_function_name,
            access,
        ),
        |result| result.and_then(RootedValueSequence::try_into_single),
    )
}

pub(crate) fn read_member_sequence_text_with_optional_access<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    let caller = match access {
        Some(access) => MemberCaller::frame(caller_function_name, access),
        None => MemberCaller::named(caller_function_name),
    };
    read_member_sequence_text_with_caller(context, base, field, allow_init, caller)
}

pub(crate) fn read_default_member_sequence_text_with_optional_access<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    let caller = match access {
        Some(access) => MemberCaller::frame(caller_function_name, access),
        None => MemberCaller::named(caller_function_name),
    };
    read_member_sequence_text_with_dispatch(
        context,
        base,
        field,
        allow_init,
        caller,
        ProtocolDispatch::Default,
    )
}
