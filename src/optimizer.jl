function Optimize(opt::Optimizer, ::LossFunction, ::QuadRule)
    __notImplement(Optimize, typeof(opt), Optimizer)
end

function Optimize(
    opt::TrustRegion, loss::BifidelityType{LossFunction}, qrule::BifidelityType{QuadRule}
) end
