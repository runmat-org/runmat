pub mod cond;
pub mod det;
pub mod inv;
pub mod linsolve;
pub mod norm;
pub mod pinv;
pub mod rank;
pub mod rcond;

pub use cond::cond_host_real_for_provider;
pub use inv::inv_host_real_for_provider;
pub use norm::norm_host_real_for_provider;
pub use pinv::pinv_host_real_for_provider;
pub use rank::rank_host_real_for_provider;
pub use rcond::rcond_host_real_for_provider;
