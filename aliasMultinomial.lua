local AM = torch.class('torch.AliasMultinomial') 

function AM:__init(probs)
   self.probs = probs
   self.J, self.q = torch.alias_multinomial_setup(probs)
end


function AM:batchdraw(output)
   local shape = output:size()
   self.probs.alias_multinomial_batchdraw(output:view(-1), self.J, self.q)
   return output
end
