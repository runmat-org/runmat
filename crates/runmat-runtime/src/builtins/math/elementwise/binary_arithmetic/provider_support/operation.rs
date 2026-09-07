use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::elementwise::binary_arithmetic) enum ArithmeticProviderOperation {
    Add,
    Subtract,
    Multiply,
}

impl ArithmeticProviderOperation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Add => "plus",
            Self::Subtract => "minus",
            Self::Multiply => "times",
        }
    }

    pub(super) async fn apply(
        self,
        provider: &dyn AccelProvider,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self {
            Self::Add => provider.elem_add(left, right).await,
            Self::Subtract => provider.elem_sub(left, right).await,
            Self::Multiply => provider.elem_mul(left, right).await,
        }
    }

    pub(super) fn scalar_left(
        self,
        provider: &dyn AccelProvider,
        right: &GpuTensorHandle,
        scalar: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self {
            Self::Add => provider.scalar_add(right, scalar),
            Self::Subtract => provider.scalar_rsub(right, scalar),
            Self::Multiply => provider.scalar_mul(right, scalar),
        }
    }

    pub(super) fn scalar_right(
        self,
        provider: &dyn AccelProvider,
        left: &GpuTensorHandle,
        scalar: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self {
            Self::Add => provider.scalar_add(left, scalar),
            Self::Subtract => provider.scalar_sub(left, scalar),
            Self::Multiply => provider.scalar_mul(left, scalar),
        }
    }
}
