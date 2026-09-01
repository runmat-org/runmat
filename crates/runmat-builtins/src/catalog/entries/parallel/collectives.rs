use super::documentation::{
    GCAT_DOCUMENTATION, GOP_DOCUMENTATION, GPLUS_DOCUMENTATION, LAB_BARRIER_DOCUMENTATION,
    LAB_BROADCAST_DOCUMENTATION, LAB_PROBE_DOCUMENTATION, LAB_RECEIVE_DOCUMENTATION,
    LAB_SEND_DOCUMENTATION, LAB_SEND_RECEIVE_DOCUMENTATION, SPMD_BARRIER_DOCUMENTATION,
    SPMD_BROADCAST_DOCUMENTATION, SPMD_CAT_DOCUMENTATION, SPMD_PLUS_DOCUMENTATION,
    SPMD_PROBE_DOCUMENTATION, SPMD_RECEIVE_DOCUMENTATION, SPMD_REDUCE_DOCUMENTATION,
    SPMD_SEND_DOCUMENTATION, SPMD_SEND_RECEIVE_DOCUMENTATION,
};
use super::*;

const BROADCAST_INPUTS: [BuiltinParamDescriptor; 2] = [LAB_REQUIRED, ANY_OPTIONAL];
const SEND_INPUTS: [BuiltinParamDescriptor; 3] = [ANY_REQUIRED, LAB_REQUIRED, TAG_OPTIONAL];
const RECEIVE_INPUTS: [BuiltinParamDescriptor; 2] = [LAB_OPTIONAL, TAG_OPTIONAL];
const GPLUS_INPUTS: [BuiltinParamDescriptor; 2] = [ANY_REQUIRED, LAB_OPTIONAL];
const GCAT_INPUTS: [BuiltinParamDescriptor; 3] = [ANY_REQUIRED, DIMENSION_OPTIONAL, LAB_OPTIONAL];
const GOP_INPUTS: [BuiltinParamDescriptor; 3] = [REDUCER_REQUIRED, ANY_REQUIRED, LAB_OPTIONAL];
const SEND_RECEIVE_INPUTS: [BuiltinParamDescriptor; 4] =
    [LAB_REQUIRED, LAB_REQUIRED, ANY_REQUIRED, TAG_OPTIONAL];
const RECEIVE_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Received value.",
    },
    BuiltinParamDescriptor {
        name: "source",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based rank of the sending lab.",
    },
    BuiltinParamDescriptor {
        name: "tag",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Tag attached to the received message.",
    },
];

signature!(
    SPMD_CAT_SIGNATURES,
    "value = spmdCat(value, dimension, destination)",
    &GCAT_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_REDUCE_SIGNATURES,
    "value = spmdReduce(reducer, value, destination)",
    &GOP_INPUTS,
    &ANY_OUTPUT
);
signature!(SPMD_BARRIER_SIGNATURES, "spmdBarrier()", &[], &[]);
signature!(
    SPMD_BROADCAST_SIGNATURES,
    "value = spmdBroadcast(source, value)",
    &BROADCAST_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_SEND_SIGNATURES,
    "spmdSend(value, destination, tag)",
    &SEND_INPUTS,
    &[]
);
signature!(
    SPMD_RECEIVE_SIGNATURES,
    "[value, source, tag] = spmdReceive(source, tag)",
    &RECEIVE_INPUTS,
    &RECEIVE_OUTPUTS
);
signature!(
    SPMD_PROBE_SIGNATURES,
    "ready = spmdProbe(source, tag)",
    &RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    LAB_SEND_RECEIVE_SIGNATURES,
    "value = labSendReceive(destination, source, value, tag)",
    &SEND_RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_SEND_RECEIVE_SIGNATURES,
    "value = spmdSendReceive(destination, source, value, tag)",
    &SEND_RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_PLUS_SIGNATURES,
    "value = spmdPlus(value, destination)",
    &GPLUS_INPUTS,
    &ANY_OUTPUT
);
signature!(BARRIER_SIGNATURES, "labBarrier()", &[], &[]);
signature!(
    BROADCAST_SIGNATURES,
    "value = labBroadcast(source, value)",
    &BROADCAST_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SEND_SIGNATURES,
    "labSend(value, destination, tag)",
    &SEND_INPUTS,
    &[]
);
signature!(
    RECEIVE_SIGNATURES,
    "value = labReceive(source, tag)",
    &RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    PROBE_SIGNATURES,
    "ready = labProbe(source, tag)",
    &RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GPLUS_SIGNATURES,
    "value = gplus(value, destination)",
    &GPLUS_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GCAT_SIGNATURES,
    "value = gcat(value, dimension, destination)",
    &GCAT_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GOP_SIGNATURES,
    "value = gop(reducer, value, destination)",
    &GOP_INPUTS,
    &ANY_OUTPUT
);

descriptor!(LAB_BARRIER_DESCRIPTOR, BARRIER_SIGNATURES);
descriptor!(LAB_BROADCAST_DESCRIPTOR, BROADCAST_SIGNATURES);
descriptor!(LAB_SEND_DESCRIPTOR, SEND_SIGNATURES);
descriptor!(LAB_RECEIVE_DESCRIPTOR, RECEIVE_SIGNATURES);
descriptor!(LAB_PROBE_DESCRIPTOR, PROBE_SIGNATURES);
descriptor!(GPLUS_DESCRIPTOR, GPLUS_SIGNATURES);
descriptor!(LAB_SEND_RECEIVE_DESCRIPTOR, LAB_SEND_RECEIVE_SIGNATURES);
descriptor!(SPMD_BARRIER_DESCRIPTOR, SPMD_BARRIER_SIGNATURES);
descriptor!(SPMD_BROADCAST_DESCRIPTOR, SPMD_BROADCAST_SIGNATURES);
descriptor!(SPMD_SEND_DESCRIPTOR, SPMD_SEND_SIGNATURES);
pub const SPMD_RECEIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPMD_RECEIVE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};
descriptor!(SPMD_PROBE_DESCRIPTOR, SPMD_PROBE_SIGNATURES);
descriptor!(SPMD_SEND_RECEIVE_DESCRIPTOR, SPMD_SEND_RECEIVE_SIGNATURES);
descriptor!(SPMD_PLUS_DESCRIPTOR, SPMD_PLUS_SIGNATURES);
descriptor!(SPMD_CAT_DESCRIPTOR, SPMD_CAT_SIGNATURES);
descriptor!(SPMD_REDUCE_DESCRIPTOR, SPMD_REDUCE_SIGNATURES);
descriptor!(GCAT_DESCRIPTOR, GCAT_SIGNATURES);
descriptor!(GOP_DESCRIPTOR, GOP_SIGNATURES);

documented_parallel_data_entry!(
    LAB_BARRIER_CATALOG_ENTRY,
    "labBarrier",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Barrier),
    LAB_BARRIER_DOCUMENTATION,
    LAB_BARRIER_DESCRIPTOR
);
documented_parallel_data_entry!(
    LAB_BROADCAST_CATALOG_ENTRY,
    "labBroadcast",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Broadcast),
    LAB_BROADCAST_DOCUMENTATION,
    LAB_BROADCAST_DESCRIPTOR
);
documented_parallel_data_entry!(
    LAB_SEND_CATALOG_ENTRY,
    "labSend",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Send),
    LAB_SEND_DOCUMENTATION,
    LAB_SEND_DESCRIPTOR
);
documented_parallel_data_entry!(
    LAB_RECEIVE_CATALOG_ENTRY,
    "labReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Receive),
    LAB_RECEIVE_DOCUMENTATION,
    LAB_RECEIVE_DESCRIPTOR
);
documented_parallel_data_entry!(
    LAB_PROBE_CATALOG_ENTRY,
    "labProbe",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Probe),
    LAB_PROBE_DOCUMENTATION,
    LAB_PROBE_DESCRIPTOR
);
documented_parallel_data_entry!(
    GPLUS_CATALOG_ENTRY,
    "gplus",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Gplus),
    GPLUS_DOCUMENTATION,
    GPLUS_DESCRIPTOR
);
documented_parallel_data_entry!(
    GCAT_CATALOG_ENTRY,
    "gcat",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Cat),
    GCAT_DOCUMENTATION,
    GCAT_DESCRIPTOR
);
documented_parallel_data_entry!(
    GOP_CATALOG_ENTRY,
    "gop",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::FunctionalReduce),
    GOP_DOCUMENTATION,
    GOP_DESCRIPTOR
);
documented_parallel_data_entry!(
    LAB_SEND_RECEIVE_CATALOG_ENTRY,
    "labSendReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SendReceive),
    LAB_SEND_RECEIVE_DOCUMENTATION,
    LAB_SEND_RECEIVE_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_BARRIER_CATALOG_ENTRY,
    "spmdBarrier",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Barrier),
    SPMD_BARRIER_DOCUMENTATION,
    SPMD_BARRIER_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_BROADCAST_CATALOG_ENTRY,
    "spmdBroadcast",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Broadcast),
    SPMD_BROADCAST_DOCUMENTATION,
    SPMD_BROADCAST_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_SEND_CATALOG_ENTRY,
    "spmdSend",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Send),
    SPMD_SEND_DOCUMENTATION,
    SPMD_SEND_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_RECEIVE_CATALOG_ENTRY,
    "spmdReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Receive),
    SPMD_RECEIVE_DOCUMENTATION,
    SPMD_RECEIVE_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_PROBE_CATALOG_ENTRY,
    "spmdProbe",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Probe),
    SPMD_PROBE_DOCUMENTATION,
    SPMD_PROBE_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_SEND_RECEIVE_CATALOG_ENTRY,
    "spmdSendReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SendReceive),
    SPMD_SEND_RECEIVE_DOCUMENTATION,
    SPMD_SEND_RECEIVE_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_PLUS_CATALOG_ENTRY,
    "spmdPlus",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Gplus),
    SPMD_PLUS_DOCUMENTATION,
    SPMD_PLUS_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_CAT_CATALOG_ENTRY,
    "spmdCat",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Cat),
    SPMD_CAT_DOCUMENTATION,
    SPMD_CAT_DESCRIPTOR
);
documented_parallel_data_entry!(
    SPMD_REDUCE_CATALOG_ENTRY,
    "spmdReduce",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::FunctionalReduce),
    SPMD_REDUCE_DOCUMENTATION,
    SPMD_REDUCE_DESCRIPTOR
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &GCAT_CATALOG_ENTRY,
    &GOP_CATALOG_ENTRY,
    &GPLUS_CATALOG_ENTRY,
    &LAB_BARRIER_CATALOG_ENTRY,
    &LAB_BROADCAST_CATALOG_ENTRY,
    &LAB_PROBE_CATALOG_ENTRY,
    &LAB_RECEIVE_CATALOG_ENTRY,
    &LAB_SEND_CATALOG_ENTRY,
    &LAB_SEND_RECEIVE_CATALOG_ENTRY,
    &SPMD_BARRIER_CATALOG_ENTRY,
    &SPMD_BROADCAST_CATALOG_ENTRY,
    &SPMD_CAT_CATALOG_ENTRY,
    &SPMD_PLUS_CATALOG_ENTRY,
    &SPMD_PROBE_CATALOG_ENTRY,
    &SPMD_RECEIVE_CATALOG_ENTRY,
    &SPMD_REDUCE_CATALOG_ENTRY,
    &SPMD_SEND_CATALOG_ENTRY,
    &SPMD_SEND_RECEIVE_CATALOG_ENTRY,
];
