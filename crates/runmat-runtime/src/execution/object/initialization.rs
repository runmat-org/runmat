//! Allocation-free native owners installed before Object publication. Native
//! states requiring their own physical storage use separate admitted factories.
use runmat_value::{ExactObjectBuilder, ForeignRef, GraphicsReceiverRef, ObjectInstance};

pub(crate) enum ObjectNativeInitialization {
    ForeignHostControl(ForeignRef),
    DatasetAuthority(ForeignRef),
    MatfileAuthority(ForeignRef),
    TcpAuthority(ForeignRef),
    GraphicsReceiver(GraphicsReceiverRef),
}
impl ObjectNativeInitialization {
    pub(super) fn install_exact(self, builder: &mut ExactObjectBuilder) -> bool {
        match self {
            Self::ForeignHostControl(value) => builder.try_set_foreign_host_control(value),
            Self::DatasetAuthority(value) => builder.try_set_dataset_authority(value),
            Self::MatfileAuthority(value) => builder.try_set_matfile_authority(value),
            Self::TcpAuthority(value) => builder.try_set_tcp_authority(value),
            Self::GraphicsReceiver(value) => builder.try_set_graphics_receiver(value),
        }
    }
    pub(super) fn install_ordinary(self, object: ObjectInstance) -> ObjectInstance {
        match self {
            Self::ForeignHostControl(value) => object.with_foreign_host_control(value),
            Self::DatasetAuthority(value) => object.with_dataset_authority(value),
            Self::MatfileAuthority(value) => object.with_matfile_authority(value),
            Self::TcpAuthority(value) => object.with_tcp_authority(value),
            Self::GraphicsReceiver(value) => object.with_graphics_receiver(value),
        }
    }
}
