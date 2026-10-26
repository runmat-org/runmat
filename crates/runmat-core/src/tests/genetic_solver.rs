//! Public parser/VM/callback ingress, distinct from the GA controller controls.
use super::{execute_text_request_named_source, RunMatSession};

#[test]
fn public_genetic_examples_preserve_values_counts_and_population_provenance() {
    runmat_gc::gc_test_context(|| {
        for example in runmat_builtins::GA_CATALOG_ENTRY.documentation.examples {
            let runmat_builtins::BuiltinExampleVerification::Assertions { source } =
                example.verification
            else {
                panic!("genetic example must verify actual outcomes");
            };
            let program = format!("{}\n{source}", example.program);
            let name = format!("genetic-{}.m", example.id);
            execute_program(&name, &program);
        }
        for (name, program) in [
            ("genetic-nonlinear-creation.m", NONLINEAR_CREATION),
            ("genetic-hybrid.m", HYBRID),
            ("genetic-output-events.m", OUTPUT_EVENTS),
        ] {
            execute_program(name, program);
        }
    });
}

fn execute_program(name: &str, program: &str) {
    eprintln!("public genetic program: {name}");
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let outcome = execute_text_request_named_source(&mut session, name, program)
        .unwrap_or_else(|error| panic!("public genetic program {name}: {error:?}"));
    assert!(
        !outcome.has_errors(),
        "public genetic program {name}: {:?}",
        outcome.diagnostics
    );
}

// Independent callback counts exercise the selected heavyweight phases through
// the public parser/VM path, rather than invoking controller helpers directly.
const NONLINEAR_CREATION: &str = r#"
global ga_phase_fcalls ga_phase_ccalls;
ga_phase_fcalls=0; ga_phase_ccalls=0;
rng(17);
o=optimoptions('ga','Display','off','PopulationSize',2, ...
    'CreationFcn',{'gacreationnonlinearfeasible','NumStartPts',2,'UseParallel',false}, ...
    'InitialPopulationRange',[0;0],'NonlinearConstraintAlgorithm','penalty', ...
    'MaxGenerations',0);
[x,f,flag,out,pop,scores]=ga(@ga_phase_objective,1,[],[],[],[],-1,1,@ga_phase_constraints,o);
assert(x==0 && f==0 && flag==0);
assert(isequal(pop,[0;0]) && isequal(scores,[0;0]));
assert(ga_phase_fcalls==1 && out.funccount==1 && out.objectiveCallCount==1);
assert(ga_phase_ccalls>=3 && out.nonlinearCallCount==ga_phase_ccalls);
assert(out.nonlinearPointCount==ga_phase_ccalls);
function f=ga_phase_objective(x)
global ga_phase_fcalls;
ga_phase_fcalls=ga_phase_fcalls+1;
f=x(1)^2;
end
function [c,ceq]=ga_phase_constraints(x)
global ga_phase_ccalls;
ga_phase_ccalls=ga_phase_ccalls+1;
c=x(1)^2-1; ceq=[];
end
"#;

const HYBRID: &str = r#"
global ga_phase_hcalls;
ga_phase_hcalls=0;
h=optimset('Display','off','MaxIter',0,'MaxFunEvals',1);
o=optimoptions('ga','Display','off','PopulationSize',2, ...
    'InitialPopulationMatrix',[1;1],'MaxGenerations',0, ...
    'HybridFcn',{'fminsearch',h});
[x,f,flag,out]=ga(@ga_phase_hobjective,1,[],[],[],[],[],[],[],o);
assert(x==1 && f==1 && flag==0);
assert(out.funccount==1 && out.objectiveCallCount==1);
assert(out.hybridfunccount==1 && out.hybridflag==0 && ~out.hybridaccepted);
assert(ga_phase_hcalls==out.funccount+out.hybridfunccount);
assert(ga_phase_hcalls==2);
function f=ga_phase_hobjective(x)
global ga_phase_hcalls;
ga_phase_hcalls=ga_phase_hcalls+1;
f=x(1)^2;
end
"#;

const OUTPUT_EVENTS: &str = r#"
global ga_phase_ocalls ga_phase_flags;
ga_phase_ocalls=0; ga_phase_flags={};
o=optimoptions('ga','Display','off','PopulationSize',2,'EliteCount',2, ...
    'InitialPopulationMatrix',[0;0],'MaxGenerations',1,'OutputFcn',@ga_phase_observer);
[x,f,flag,out]=ga(@ga_phase_oobjective,1,[],[],[],[],[],[],[],o);
assert(x==0 && f==0 && flag==0);
assert(out.generations==1 && out.funccount==2 && out.objectiveCallCount==2);
assert(ga_phase_ocalls==out.funccount);
assert(numel(ga_phase_flags)==3);
assert(strcmp(ga_phase_flags{1},'init'));
assert(strcmp(ga_phase_flags{2},'iter'));
assert(strcmp(ga_phase_flags{3},'done'));
function f=ga_phase_oobjective(x)
global ga_phase_ocalls;
ga_phase_ocalls=ga_phase_ocalls+1;
f=x(1)^2;
end
function [state,options,changed]=ga_phase_observer(options,state,phase)
global ga_phase_ocalls ga_phase_flags;
ga_phase_flags{end+1}=phase;
assert(state.FunEval==ga_phase_ocalls);
assert(isequal(state.Population,[0;0]) && isequal(state.Score,[0;0]));
changed=false;
end
"#;

include!("pattern_genetic_search.rs");
