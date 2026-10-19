use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ParallelInferenceRule {
    Barrier,
    Broadcast,
    Cat,
    Codistributed,
    CodistributedBuild,
    Codistributor,
    Codistributor1d,
    Codistributor2dbc,
    CodistributorIsComplete,
    Distributed,
    FunctionalReduce,
    GetCodistributor,
    GlobalIndices,
    Gplus,
    Iscodistributed,
    LocalPart,
    Probe,
    Receive,
    Redistribute,
    Send,
    SendReceive,
}
