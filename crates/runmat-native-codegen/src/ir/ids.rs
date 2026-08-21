use serde::{Deserialize, Serialize};

macro_rules! native_id {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub u32);
    };
}

native_id!(NativeBlockId);
native_id!(NativeInstructionId);
native_id!(NativeLocalId);
native_id!(NativeSafepointId);
native_id!(NativeValueId);
