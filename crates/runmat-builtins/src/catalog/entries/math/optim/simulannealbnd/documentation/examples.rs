use crate::*;
const fn example(
    id: &'static str,
    program: &'static str,
    assertions: &'static str,
    graphics: bool,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title: id,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: if graphics {
            BuiltinExampleHarness::BrowserGraphics
        } else {
            BuiltinExampleHarness::Portable
        },
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}
pub(super) const ACCEPTANCESA: BuiltinExample = example(
    "actual-acceptancesa",
    "v=struct('fval',2,'temperature',1); accept=acceptancesa(v,0,1);",
    "assert(islogical(accept) && accept);",
    false,
);
pub(super) const ANNEALINGFAST:BuiltinExample=example("actual-annealingfast",
    "v=struct('x',[0;0],'temperature',[4;4]); p=struct('lb',[-10;-10],'ub',[10;10]); x=annealingfast(v,p);",
    "assert(isequal(size(x),[2 1])); assert(abs(norm(x)-4)<1e-12);",false);
pub(super) const ANNEALINGBOLTZ:BuiltinExample=example("actual-annealingboltz",
    "v=struct('x',[0;0],'temperature',[4;4]); p=struct('lb',[-10;-10],'ub',[10;10]); x=annealingboltz(v,p);",
    "assert(isequal(size(x),[2 1])); assert(abs(norm(x)-2)<1e-12);",false);
pub(super) const TEMPERATUREEXP: BuiltinExample = example(
    "actual-temperatureexp",
    "v=struct('k',[2;4]); opts=struct('InitialTemperature',[100;200]); t=temperatureexp(v,opts);",
    "assert(isequal(size(t),[2 1])); assert(norm(t-[100*.95^2;200*.95^4])<1e-12);",
    false,
);
pub(super) const TEMPERATUREFAST: BuiltinExample = example(
    "actual-temperaturefast",
    "v=struct('k',[2;4]); opts=struct('InitialTemperature',[100;200]); t=temperaturefast(v,opts);",
    "assert(isequal(size(t),[2 1])); assert(norm(t-[50;50])<1e-12);",
    false,
);
pub(super) const TEMPERATUREBOLTZ: BuiltinExample = example(
    "actual-temperatureboltz",
    "v=struct('k',[2;4]); opts=struct('InitialTemperature',[100;200]); t=temperatureboltz(v,opts);",
    "assert(isequal(size(t),[2 1])); assert(norm(t-[100/log(2);200/log(4)])<1e-12);",
    false,
);
pub(super) const SAPLOTBESTF:BuiltinExample=example("actual-saplotbestf",
    "opts=struct('FunctionTolerance',1e-6,'MaxFunctionEvaluations',10,'MaxIterations',10,'MaxTime',Inf); v=struct('x',[1 2],'bestx',[0 1],'fval',5,'bestfval',1,'temperature',[2;4],'iteration',0,'funccount',1,'averagechange',1,'t0',0); stop=saplotbestf(opts,v,'init'); saplotbestf(opts,v,'done');",
    "assert(islogical(stop) && ~stop);",true);
pub(super) const SAPLOTBESTX:BuiltinExample=example("actual-saplotbestx",
    "opts=struct('FunctionTolerance',1e-6,'MaxFunctionEvaluations',10,'MaxIterations',10,'MaxTime',Inf); v=struct('x',[1 2],'bestx',[0 1],'fval',5,'bestfval',1,'temperature',[2;4],'iteration',0,'funccount',1,'averagechange',1,'t0',0); stop=saplotbestx(opts,v,'init'); saplotbestx(opts,v,'done');",
    "assert(islogical(stop) && ~stop);",true);
pub(super) const SAPLOTF:BuiltinExample=example("actual-saplotf",
    "opts=struct('FunctionTolerance',1e-6,'MaxFunctionEvaluations',10,'MaxIterations',10,'MaxTime',Inf); v=struct('x',[1 2],'bestx',[0 1],'fval',5,'bestfval',1,'temperature',[2;4],'iteration',0,'funccount',1,'averagechange',1,'t0',0); stop=saplotf(opts,v,'init'); saplotf(opts,v,'done');",
    "assert(islogical(stop) && ~stop);",true);
pub(super) const SAPLOTX:BuiltinExample=example("actual-saplotx",
    "opts=struct('FunctionTolerance',1e-6,'MaxFunctionEvaluations',10,'MaxIterations',10,'MaxTime',Inf); v=struct('x',[1 2],'bestx',[0 1],'fval',5,'bestfval',1,'temperature',[2;4],'iteration',0,'funccount',1,'averagechange',1,'t0',0); stop=saplotx(opts,v,'init'); saplotx(opts,v,'done');",
    "assert(islogical(stop) && ~stop);",true);
pub(super) const SAPLOTSTOPPING:BuiltinExample=example("actual-saplotstopping",
    "opts=struct('FunctionTolerance',1e-6,'MaxFunctionEvaluations',10,'MaxIterations',10,'MaxTime',Inf); v=struct('x',[1 2],'bestx',[0 1],'fval',5,'bestfval',1,'temperature',[2;4],'iteration',0,'funccount',1,'averagechange',1,'t0',0); stop=saplotstopping(opts,v,'init'); saplotstopping(opts,v,'done');",
    "assert(islogical(stop) && ~stop);",true);
pub(super) const SAPLOTTEMPERATURE:BuiltinExample=example("actual-saplottemperature",
    "opts=struct('FunctionTolerance',1e-6,'MaxFunctionEvaluations',10,'MaxIterations',10,'MaxTime',Inf); v=struct('x',[1 2],'bestx',[0 1],'fval',5,'bestfval',1,'temperature',[2;4],'iteration',0,'funccount',1,'averagechange',1,'t0',0); stop=saplottemperature(opts,v,'init'); saplottemperature(opts,v,'done');",
    "assert(islogical(stop) && ~stop);",true);
