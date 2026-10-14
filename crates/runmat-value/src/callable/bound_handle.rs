use std::ops::Deref;

use super::{BoundFunctionAuthority, CallableInstanceId};

#[derive(Clone, Debug, PartialEq)]
pub struct BoundFunctionHandleData {
    pub name: String,
    pub function: usize,
    pub authority: BoundFunctionAuthority,
    pub source_role: runmat_types::FunctionSourceRole,
    pub source_file: Option<String>,
}

/// Bound execution authority is independent of this handle's physical
/// instance. A cloned handle retains its instance; a newly issued one does not.
#[derive(Clone, Debug)]
pub struct BoundFunctionHandle {
    data: BoundFunctionHandleData,
    instance: CallableInstanceId,
}

impl BoundFunctionHandle {
    pub fn new(
        name: String,
        function: usize,
        authority: BoundFunctionAuthority,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
    ) -> Self {
        Self::from_data(BoundFunctionHandleData {
            name,
            function,
            authority,
            source_role,
            source_file,
        })
    }

    pub fn from_data(data: BoundFunctionHandleData) -> Self {
        Self {
            data,
            instance: CallableInstanceId::mint(),
        }
    }

    pub fn into_data(self) -> BoundFunctionHandleData {
        self.data
    }

    pub fn instance_id(&self) -> CallableInstanceId {
        self.instance
    }
}

impl Deref for BoundFunctionHandle {
    type Target = BoundFunctionHandleData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl PartialEq for BoundFunctionHandle {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
