use runmat_macros::runtime_builtin;
use runmat_value::Value;

fn lowering_required(name: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!(
        "{name}: this operation requires an active distributed execution context"
    ))
    .with_builtin(name)
    .with_identifier("RunMat:parallel:LoweringRequired")
    .build()
}

macro_rules! lowering_builtin {
    ($function:ident, $name:literal) => {
        #[runtime_builtin(
                                                                    name = $name,
                                                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::parallel::primitives"
                                                                )]
        async fn $function(_arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
            Err(lowering_required($name))
        }
    };
}

lowering_builtin!(distributed_builtin, "distributed");
lowering_builtin!(get_local_part_builtin, "getLocalPart");
lowering_builtin!(lab_barrier_builtin, "labBarrier");
lowering_builtin!(lab_broadcast_builtin, "labBroadcast");
lowering_builtin!(lab_send_builtin, "labSend");
lowering_builtin!(lab_receive_builtin, "labReceive");
lowering_builtin!(lab_probe_builtin, "labProbe");
lowering_builtin!(lab_send_receive_builtin, "labSendReceive");
lowering_builtin!(gplus_builtin, "gplus");
lowering_builtin!(spmd_barrier_builtin, "spmdBarrier");
lowering_builtin!(spmd_broadcast_builtin, "spmdBroadcast");
lowering_builtin!(spmd_send_builtin, "spmdSend");
lowering_builtin!(spmd_receive_builtin, "spmdReceive");
lowering_builtin!(spmd_probe_builtin, "spmdProbe");
lowering_builtin!(spmd_send_receive_builtin, "spmdSendReceive");
lowering_builtin!(spmd_plus_builtin, "spmdPlus");
