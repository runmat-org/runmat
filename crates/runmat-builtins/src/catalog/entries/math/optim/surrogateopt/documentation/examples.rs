use crate::*;
const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    example("unique-bounds", "Evaluate a fixed feasible point", r#"opts=optimoptions('surrogateopt','Display','off','PlotFcn',[]);
[x,f,flag,out,trials]=surrogateopt(@(x)sum(x.^2),[2 -1],[2 -1],opts);"#,
        "assert(isequal(x,[2 -1]) && f==5 && flag==10); assert(out.funccount==1); assert(isequal(trials.X,[2 -1]) && trials.Fval==5);"),
    example("known-initial", "Reuse complete initial measurements without a call", r#"initial=struct('X',[-1;0;1],'Fval',[1;0;1],'Ineq',zeros(3,0));
opts=optimoptions('surrogateopt','Display','off','PlotFcn',[],'InitialPoints',initial,'MaxFunctionEvaluations',0);
[x,f,flag,out,trials]=surrogateopt(@(x)error('unexpected objective call'),-2,2,opts);"#,
        "assert(x==0 && f==0 && flag==0); assert(out.funccount==0); assert(isequal(trials.X,[-1;0;1])); assert(isequal(trials.Fval,[1;0;1]));"),
    example("vectorized-final-batch", "Count rows across a partial final batch", r#"opts=optimoptions('surrogateopt','Display','off','PlotFcn',[],'InitialPoints',[-2;-1;0;1;2],...
'UseVectorized',true,'BatchUpdateInterval',3,'MaxFunctionEvaluations',5);
[x,f,flag,out,trials]=surrogateopt(@(x)x.^2,-3,3,opts);"#,
        "assert(x==0 && f==0 && flag==0); assert(out.funccount==5); assert(isequal(trials.X,[-2;-1;0;1;2])); assert(isequal(trials.Fval,[4;1;0;1;4]));"),
    BuiltinExample {
        id: "saved-total-budget", title: "Continue saved initial rows with a total budget",
        program: r#"opts=optimoptions('surrogateopt','Display','off','PlotFcn',[],'InitialPoints',[0;1;2],...
'MaxFunctionEvaluations',1,'CheckpointFile','checkpoints/surrogate-checkpoint.mat');
[x,f,flag,out]=surrogateopt(@(x)x(1)^2,-3,3,opts);
more=opts; more.MaxFunctionEvaluations=3;
[xr,fr,flagr,outr,trials]=surrogateopt('checkpoints/surrogate-checkpoint.mat',more);"#,
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "surrogate-checkpoint" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::Directory { relative_path: "checkpoints" }],
        }), requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(out.funccount==1 && outr.funccount==3); assert(x==0 && xr==0 && f==0 && fr==0 && flag==0 && flagr==0); assert(isequal(trials.X,[0;1;2]) && isequal(trials.Fval,[0;1;4]));" },
    },
];
