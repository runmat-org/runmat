use super::{
    creator_documentation as docs, creator_inference as infer, helper_documentation::example,
    helper_entries::creator, *,
};
use crate::*;
pub const GACREATIONUNIFORM_CATALOG_ENTRY:BuiltinCatalogEntry=creator("gacreationuniform",&GA_CREATION_HELPER_DESCRIPTOR,
    docs::doc("gacreationuniform","Create uniform population rows without fitness evaluation.",&[example(
        "o=optimoptions('ga','PopulationSize',2,'InitialPopulationMatrix',[-1;1]); p=struct('FitnessFcn',@(x)x(1)^2,'nvars',1,'lb',-2,'ub',2); X=gacreationuniform(1,p,o);",
        "assert(isequal(X,[-1;1]));")]),infer::uniform,false,false);
pub const GACREATIONLINEARFEASIBLE_CATALOG_ENTRY:BuiltinCatalogEntry=creator("gacreationlinearfeasible",&GA_CREATION_HELPER_DESCRIPTOR,
    docs::doc("gacreationlinearfeasible","Create rows admitted to the original linear and bound constraints.",&[example(
        "o=optimoptions('ga','PopulationSize',2,'InitialPopulationMatrix',[-3;3]); p=struct('FitnessFcn',@(x)x(1)^2,'nvars',1,'Aeq',1,'Beq',2,'lb',0,'ub',3); X=gacreationlinearfeasible(1,p,o);",
        "assert(isequal(size(X),[2 1])); assert(all(abs(X-2)<1e-8));")]),infer::linear,true,false);
pub const GACREATIONUNIFORMINT_CATALOG_ENTRY:BuiltinCatalogEntry=creator("gacreationuniformint",&GA_CREATION_HELPER_DESCRIPTOR,
    docs::doc("gacreationuniformint","Create representable integer coordinates and verify original feasibility.",&[example(
        "o=optimoptions('ga','PopulationSize',2,'InitialPopulationMatrix',[0.2;1.8]); p=struct('FitnessFcn',@(x)x(1)^2,'nvars',1,'lb',0,'ub',2,'intcon',1); X=gacreationuniformint(1,p,o);",
        "assert(isequal(X,[0;2]));")]),infer::integer,true,false);
pub const GACREATIONSOBOL_CATALOG_ENTRY:BuiltinCatalogEntry=creator("gacreationsobol",&GA_CREATION_HELPER_DESCRIPTOR,
    docs::doc("gacreationsobol","Create a Sobol population and admit original constraints.",&[example(
        "o=optimoptions('ga','PopulationSize',4); p=struct('FitnessFcn',@(x)x(1)^2,'nvars',2,'lb',[0 0],'ub',[1 1]); X=gacreationsobol(2,p,o);",
        "assert(isequal(size(X),[4 2])); assert(all(all(X>=0 & X<=1))); assert(isequal(X(1,:),[0 0]));")]),infer::sobol,true,false);
pub const GACREATIONNONLINEARFEASIBLE_CATALOG_ENTRY:BuiltinCatalogEntry=creator("gacreationnonlinearfeasible",&GA_NONLINEAR_CREATION_HELPER_DESCRIPTOR,
    docs::doc("gacreationnonlinearfeasible","Run selected constrained-feasibility starts without objective callbacks.",&[example(
        "o=optimoptions('ga','PopulationSize',2); p=struct('FitnessFcn',@(x)x(1)^2,'nvars',1,'lb',0,'ub',0,'nonlcon',@(x)deal([],x(1))); X=gacreationnonlinearfeasible(1,p,o,'NumStartPts',2,'UseParallel','off');",
        "assert(isequal(X,[0;0]));")]),infer::nonlinear,true,true);
