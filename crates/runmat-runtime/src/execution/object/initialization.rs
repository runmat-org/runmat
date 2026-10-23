//! Allocation-free native owners installed before Object publication. Native
//! states requiring their own physical storage use separate admitted factories.
#[cfg(feature = "plot-core")]
use runmat_value::GraphicsReceiverRef;
use runmat_value::{ExactObjectBuilder, ForeignRef, ObjectInstance};

pub(crate) enum ObjectNativeInitialization<'a> {
    ForeignHostControl(ForeignRef),
    DatasetAuthority(ForeignRef),
    MatfileAuthority(ForeignRef),
    TcpAuthority(ForeignRef),
    #[cfg(feature = "plot-core")]
    GraphicsReceiver(GraphicsReceiverRef),
    NumericFactor(&'a runmat_value::NumericFactorData),
}
impl ObjectNativeInitialization<'_> {
    pub(super) fn additional_quote(&self) -> (usize, usize) {
        match self {
            Self::NumericFactor(_) => (
                ExactObjectBuilder::numeric_factor_native_requested_bytes(),
                ExactObjectBuilder::numeric_factor_native_retained_claim_count(),
            ),
            _ => (0, 0),
        }
    }
    pub(super) fn install_exact(
        self,
        builder: &mut ExactObjectBuilder,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<bool, runmat_value::ExactHostNumericAllocationError> {
        Ok(match self {
            Self::ForeignHostControl(value) => builder.try_set_foreign_host_control(value),
            Self::DatasetAuthority(value) => builder.try_set_dataset_authority(value),
            Self::MatfileAuthority(value) => builder.try_set_matfile_authority(value),
            Self::TcpAuthority(value) => builder.try_set_tcp_authority(value),
            #[cfg(feature = "plot-core")]
            Self::GraphicsReceiver(value) => builder.try_set_graphics_receiver(value),
            Self::NumericFactor(value) => {
                return builder.try_set_numeric_factor_data(value.clone(), permit)
            }
        })
    }
    pub(super) fn install_ordinary(self, object: ObjectInstance) -> ObjectInstance {
        match self {
            Self::ForeignHostControl(value) => object.with_foreign_host_control(value),
            Self::DatasetAuthority(value) => object.with_dataset_authority(value),
            Self::MatfileAuthority(value) => object.with_matfile_authority(value),
            Self::TcpAuthority(value) => object.with_tcp_authority(value),
            #[cfg(feature = "plot-core")]
            Self::GraphicsReceiver(value) => object.with_graphics_receiver(value),
            Self::NumericFactor(value) => object.with_numeric_factor_data(value.clone()),
        }
    }
}
