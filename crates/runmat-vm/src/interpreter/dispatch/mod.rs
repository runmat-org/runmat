mod calls;
mod control_flow;
mod arrays;
mod exceptions;
mod stack;

pub use calls::{
    build_builtin_expand_at_args, build_builtin_expand_last_args, build_builtin_expand_multi_args,
    handle_builtin_outcome, handle_feval_dispatch, output_list_for_user_call, push_single_result,
    prepare_named_user_dispatch, push_user_call_outputs, unpack_prepared_user_call,
    BuiltinHandling, FevalHandling, PreparedUserDispatch,
};
pub use arrays::{create_matrix, create_matrix_dynamic, create_range, pack_to_col, pack_to_row, unpack};
pub use control_flow::{apply_control_flow_action, DispatchDecision};
pub use exceptions::{
    parse_exception, prepare_vm_error, redirect_exception_to_catch, ExceptionHandling,
};
pub use stack::{
    emit_stack_top, emit_var, load_bool, load_char_row, load_complex, load_const, load_local,
    load_string, load_var, store_local, store_var,
};
