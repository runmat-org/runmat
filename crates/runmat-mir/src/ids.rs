use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    ($name:ident) => {
        #[derive(
            Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub usize);
    };
}

id_newtype!(BasicBlockId);
id_newtype!(MirLocalId);
id_newtype!(MirTempId);
