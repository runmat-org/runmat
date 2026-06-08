use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    ($name:ident) => {
        /// Local identity inside one HIR assembly.
        ///
        /// These IDs are stable only while referring to the same compiler product.
        /// Persisted or cross-session identity should use qualified semantic paths such
        /// as `DefPath`, not these arena-style numeric IDs.
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
