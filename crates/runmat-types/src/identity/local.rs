use serde::{Deserialize, Serialize};

macro_rules! local_id {
    ($name:ident) => {
        #[derive(
            Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub usize);
    };
}

local_id!(ModuleId);
local_id!(FunctionId);
local_id!(ClassId);
local_id!(EntrypointId);
local_id!(BindingId);
local_id!(ExprId);
local_id!(StmtId);
local_id!(SourceId);
