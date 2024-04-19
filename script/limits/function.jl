function limit(x; n=length(x), div=Int(floor(sqrt(n))))
  f = Int(floor(n/div))
  sub(range) = sum(i * x[i] for i in range)^2
  sum( sub((i-1)*f+1:(i+2)*f)/(1+x[i]^2) for i in 1:div-3) + sum( sub((i-1)*f+5:(i+4)*f+5)/(1+x[n-i]^2) for i in 1:div-5)
end

function start_limit(n :: Int; div=Int(floor(sqrt(n))))
  f = Int(floor(n/div))
  _n = f*div
  10 .* ones(_n)
end