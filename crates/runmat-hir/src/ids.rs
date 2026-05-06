use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    ($name:ident) => {
        #[derive(
            Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub usize);
    };
}

id_newtype!(ModuleId);
id_newtype!(FunctionId);
id_newtype!(ClassId);
id_newtype!(EntrypointId);
id_newtype!(BindingId);
id_newtype!(ExprId);
id_newtype!(StmtId);
id_newtype!(SourceId);

// Legacy VM/lowering-era variable identity. New semantic HIR should use `BindingId`;
// this remains only so pre-MIR migration code can keep compiling during Plans 0-3.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub usize);
