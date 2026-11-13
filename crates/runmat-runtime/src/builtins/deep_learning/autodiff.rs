use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, IntValue, NumericDType, NumericScalar, NumericStorage, ObjectInstance, StructValue,
    Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::{deep_learning_error, model, unsupported_error};

pub(super) const DLARRAY_CLASS: runmat_types::StaticClassIdentity = super::DLARRAY_CLASS;
mod tape;
#[cfg(test)]
use tape::annotate_tree;
#[cfg(test)]
use tape::AD_NODE_PROPERTY;
use tape::{
    active_tape, attach_node_to_object, push_binary_node, push_node, value_for_node, Node, NodeKind,
};
pub(super) use tape::{annotate_dlarray_value, dlarray_node_id, tape_is_active, with_tape_context};
#[cfg(test)]
pub(in crate::builtins::deep_learning) use tape::{test_tape_node_count, with_test_tape};
pub(crate) use tape::{TapePollAuthority, TapePollGuard};
mod value;
pub(super) use value::{dlarray_format_and_labels, host_tensor_from_value};
use value::{
    floating_binary_dtype, floating_tensor_from_f64, floating_tensor_values, node_dtype,
    result_metadata_owner, wrap_result, Payload,
};
pub(crate) mod gradient;
#[cfg(test)]
pub(super) use gradient::dlgradient_direct;

pub(crate) mod array_methods;
#[cfg(test)]
pub(super) use array_methods::{
    dlarray_minus_builtin, dlarray_mtimes_builtin, dlarray_plus_builtin, dlarray_rdivide_builtin,
    dlarray_sum_builtin, dlarray_times_builtin,
};

mod network;
pub(in crate::builtins::deep_learning) use network::{
    admit_resident_tape_parents, ResidentTapeAdmission,
};
pub(super) use network::{
    annotate_network_outputs, record_network_execution, record_resident_network_execution,
};

#[cfg(test)]
mod tests;
