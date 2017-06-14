local tester = torch.Tester()
local mytest = torch.TestSuite()


local function aliasMultinomial()
   local n_class = 10000
   local probs = torch.Tensor(n_class):uniform(0,1)
   probs:div(probs:sum())
   local a = torch.Timer()
   local toast = torch.aliasMultinomial(probs)
   print("AliasMultinomial setup in "..a:time().real.." seconds")
   a:reset()
   
   local output = torch.LongTensor(1000, 10000)
   toast_output = toast:batchdraw(output)
   local n_samples = output:nElement()
   print("AliasMultinomial draw "..n_samples.." elements from "..n_class.." classes ".."in "..a:time().real.." seconds")
   local counts = torch.Tensor(n_class):zero()
   mult_output = torch.multinomial(probs, n_samples, true)
   print("Multinomial draw "..n_samples.." elements from "..n_class.." classes ".." in "..a:time().real.." seconds")
   toast_output:apply(function(x)
         counts[x] = counts[x] + 1
   end)
   a:reset()
   
   counts:div(counts:sum())
   tester:eq(probs, counts, 0.001, "probs and counts should be approximately equal")
end

tester:add(aliasMultinomial)
tester:run()
