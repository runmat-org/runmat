//! CLI organization selection; native lease transport has one shared owner.
pub(super) use runmat_fea_admission_native::NativeFeaAdmission as CliFeaAdmission;
use uuid::Uuid;

pub(super) fn organization(cli: &crate::cli::Cli) -> Option<Uuid> {
    use crate::cli::Commands;
    match &cli.command {
        Some(Commands::Run { execution, .. } | Commands::Mesh { execution, .. }) => execution.org,
        _ => cli.execution.org,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_frontdoors_share_org_selection_without_changing_login_or_cluster_flags() {
        use crate::cli::{Cli, ClusterCommand, Commands};
        use clap::Parser;
        let id = "org_00000000000000000000000000000001";
        for args in [
            vec!["runmat", "main.m", "--org", id],
            vec!["runmat", "run", "main.m", "--org", id],
            vec!["runmat", "mesh", "part.step", "--org", id],
            vec!["runmat", "run", "main.fea", "--org", id],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(organization(&cli), Some(Uuid::from_u128(1)));
        }
        let login = Cli::try_parse_from(["runmat", "login", "--org", id]).unwrap();
        assert!(
            matches!(login.command, Some(Commands::Login { org: Some(value), .. }) if value == Uuid::from_u128(1))
        );
        let cluster = Cli::try_parse_from(["runmat", "cluster", "list", "--org", id]).unwrap();
        assert!(
            matches!(cluster.command, Some(Commands::Cluster { cluster_command: ClusterCommand::List { org: Some(value), .. } }) if value == Uuid::from_u128(1))
        );
    }
}
