use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::shared::infer_axis_setting(
        request,
        entry,
        "xticks",
        super::super::shared::AxisSettingForm::Ticks,
    )
}
