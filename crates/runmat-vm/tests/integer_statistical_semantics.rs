#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn documented_floating_random_and_clustering_forms_execute_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "rng('default'); g=gamrnd(2,3,1,2); if ~isa(g,'double') || ~isequal(size(g),[1 2]); error('gamrnd floating form failed'); end; d=lhsdesign(4,2,'Iterations',1); if ~isa(d,'double') || ~isequal(size(d),[4 2]); error('lhsdesign floating form failed'); end; idx=kmeans([0;1;9;10],2,'Start',[0;10]); if ~isa(idx,'double') || ~isequal(size(idx),[4 1]); error('kmeans floating form failed'); end; near=knnsearch([0;2;5],[1;4],'K',1,'NSMethod','exhaustive'); if ~isa(near,'double') || ~isequal(size(near),[2 1]); error('knnsearch floating form failed'); end; tree=linkage([0;2;5]); if ~isa(tree,'double') || ~isequal(size(tree),[2 3]); error('linkage floating form failed'); end;",
    )
    .expect("documented floating statistics forms");
}

#[test]
fn typed_integer_statistical_extensions_are_mode_gated() {
    let cases = [
        (
            "x=gamrnd(uint8(2),3);",
            "RunMat:compatibility:GamrndIntegerShapeParameterExtension",
        ),
        (
            "x=lhsdesign(uint8(4),2);",
            "RunMat:compatibility:LhsdesignIntegerDimensionExtension",
        ),
        (
            "x=isoutlier(int16([1;2;100]));",
            "RunMat:compatibility:IsoutlierIntegerDataExtension",
        ),
        (
            "x=kmeans(int16([0;1;9;10]),2);",
            "RunMat:compatibility:KmeansIntegerObservationDataExtension",
        ),
        (
            "x=knnsearch(int16([0;2;5]),[1;4]);",
            "RunMat:compatibility:KnnsearchIntegerObservationDataExtension",
        ),
        (
            "x=linkage(int16([0;2;5]));",
            "RunMat:compatibility:LinkageIntegerDataExtension",
        ),
    ];
    for (source, identifier) in cases {
        let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let error = execute_source(source).expect_err("typed integer extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}: {error}");
    }
}

#[test]
fn typed_integer_statistical_extensions_execute_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "rng('default'); g=gamrnd(uint8(2),uint16(3),uint8(1),uint16(2)); if ~isa(g,'double') || ~isequal(size(g),[1 2]); error('integer gamrnd failed'); end; d=lhsdesign(uint8(4),uint16(2),'Iterations',uint32(1)); if ~isequal(size(d),[4 2]); error('integer lhsdesign failed'); end; o=isoutlier(int16([1;2;100;4;5])); if ~islogical(o) || ~o(3); error('integer isoutlier failed'); end; c=kmeans(int16([0;1;9;10]),uint8(2),'Start',int16([0;10]),'MaxIter',uint16(10)); if ~isequal(size(c),[4 1]); error('integer kmeans failed'); end; n=knnsearch(int16([0;2;5]),uint16([1;4]),'K',uint8(1),'NSMethod','exhaustive'); if ~isequal(n,[1;3]); error('integer knnsearch failed'); end; z=linkage(int16([0;2;5])); if ~isequal(size(z),[2 3]); error('integer linkage failed'); end;",
    )
    .expect("RunMat integer statistical extensions");
}

#[test]
fn integer_grouping_preserves_wide_keys_without_a_double_mirror() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "base=uint64(9007199254740992); groups=[base;base+uint64(1);base]; [counts,keys,pct]=groupcounts(groups); if ~isequal(counts,[2;1]) || ~isa(keys,'uint64') || keys(1)~=base || keys(2)~=base+uint64(1) || abs(sum(pct)-100)>1e-12; error('wide integer groupcounts failed'); end;",
    )
    .expect("exact wide integer grouping");
}

#[test]
fn groupsummary_array_forms_preserve_wide_integer_groups_and_extrema() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "base=uint64(9007199254740992); data=[base+uint64(3);base+uint64(1);base+uint64(4)]; groups=[base;base+uint64(1);base]; [b,bg,bc]=groupsummary(data,groups,'max'); if ~isa(b,'uint64') || ~isequal(b,[base+uint64(4);base+uint64(1)]) || ~isa(bg,'uint64') || ~isequal(bg,[base;base+uint64(1)]) || ~isequal(bc,[2;1]); error('wide integer array groupsummary failed'); end;",
    )
    .expect("exact wide integer array groupsummary");
}

#[test]
fn grouping_bins_preserve_wide_edges_and_empty_right_edge_behavior() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "base=uint64(9007199254740992); g=[base;base+uint64(2)]; edges=[base;base+uint64(1);base+uint64(2);base+uint64(3)]; [c,~,~]=groupcounts(g,edges,'IncludeEmptyGroups',true); if ~isequal(c,[1;0;1]); error('wide groupcount bins failed'); end; [b,bg,bc]=groupsummary([10;30],g,edges,'mean','IncludeEmptyGroups',true); if b(1)~=10 || ~isnan(b(2)) || b(3)~=30 || ~isequal(bc,[1;0;1]); error('wide groupsummary bins failed'); end; [cr,~,~]=groupcounts(g,edges,'IncludedEdge','right','IncludeEmptyGroups',true); if ~isequal(cr,[1;1;0]); error('right included edge failed'); end;",
    )
    .expect("exact wide grouping bins");
}

#[test]
fn grouping_none_and_scalar_double_bin_count_forms_execute() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "g=[1;2;3;4]; [plain,plain_groups]=groupcounts(g,'none'); if ~isequal(plain,[1;1;1;1]) || ~isequal(plain_groups,g); error('groupcounts none failed'); end; [binned,~,~]=groupcounts(g,2); if ~isequal(binned,[2;2]); error('groupcounts scalar bin count failed'); end; [summary,summary_groups,summary_counts]=groupsummary([10;20;30;40],g,'none','mean'); if ~isequal(summary,[10;20;30;40]) || ~isequal(summary_groups,g) || ~isequal(summary_counts,[1;1;1;1]); error('groupsummary none failed'); end; [binned_summary,~,binned_counts]=groupsummary([10;20;30;40],g,2,'mean'); if ~isequal(binned_summary,[15;35]) || ~isequal(binned_counts,[2;2]); error('groupsummary scalar bin count failed'); end;",
    )
    .expect("documented none and scalar-double grouping bins");
}

#[test]
fn grouping_native_controls_are_strictly_gated() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "x=groupcounts([1;2],uint8(2));",
        "x=groupsummary([10;20],[1;2],{uint8(2)},'mean');",
    ] {
        let error = execute_source(source).expect_err("native grouping control must reject");
        assert!(
            error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")),
            "{source}: {error}"
        );
    }
}

#[test]
fn lossy_integer_floating_boundaries_reject_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "wide=uint64(9007199254740992)+uint64(1); x=gamrnd(wide,1);",
        "wide=uint64(9007199254740992)+uint64(1); x=kmeans([wide;wide+uint64(2)],1);",
        "wide=uint64(9007199254740992)+uint64(1); x=knnsearch([wide;wide+uint64(2)],wide);",
        "wide=uint64(9007199254740992)+uint64(1); x=linkage([wide;wide+uint64(2);wide+uint64(4)]);",
    ] {
        let error = execute_source(source).expect_err("lossy floating boundary must reject");
        assert!(
            error.message().contains("exact") || error.message().contains("represent"),
            "{source}: {error}"
        );
    }
}
