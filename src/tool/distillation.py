import torch
import torch.nn as nn
from torch.nn import functional as F

#---------------------------------------------------#
#   Individual KD
#---------------------------------------------------#
class DistillerIKD:
  
    def __init__(self, teacherprediction, studentprediction, temperature, lnorm):
        self.teacherprediction = teacherprediction
        self.studentprediction = studentprediction
        self.temperature = temperature
        self.dim = lnorm

    # Adaptation    
    def convert4Dto2D(self, input, nature, rskd=False):
      """
                  nature describe if teacher or student
        Needed for adjusting the normalization of feature maps
        example: nature='s' for student, nature='t' for teacher

      """

      output2D = []
      a = input.size(1)
      if a >= 255 and input.shape == torch.Size([4,a,19, 19]) or torch.Size([4,a,38, 38]) :
        t = input.permute(0, 2, 1, 3)
        for i in range(t.size(0)):
          z = t[i, :, :, :]
          for d in range(z.size(0)):
            if rskd == True and nature == 't':
              target = F.softmax(z[d,:,:]/self.temperature, dim=1)
            elif rskd == True and nature == 's':
              target = F.log_softmax(z[d,:,:]/self.temperature, dim=1)
            else:
              target = z[d,:,:] 
            output2D.append(target)
        return output2D

    #---------------------------------------------------#
    #   logit loss
    #---------------------------------------------------#
    def rskd_loss(self, nature1='s', nature2='t'):
        """nature1 and nature2 are similar to nature""" 
        t = self.convert4Dto2D(self.teacherprediction, nature2,True)
        s = self.convert4Dto2D(self.studentprediction, nature1,True)
        kl_divergence = nn.KLDivLoss(reduction='batchmean', log_target=True)
        loss=0
        for i in range(len(s)):
            loss += kl_divergence((s[i]), t[i])*self.temperature
        return loss

    #---------------------------------------------------#
    #   feature based hint. 
    #---------------------------------------------------#
    def hint_loss(self, nature1='s', nature2='t'):
        t = self.convert4Dto2D(self.teacherprediction, nature2)
        s = self.convert4Dto2D(self.studentprediction, nature1)
        loss=0
        for i in range(len(s)):
            loss += torch.linalg.norm( torch.sub(s[i], t[i]) , ord=self.dim)
        return loss

#---------------------------------------------------#
#   Relational hint
#---------------------------------------------------#
class DistillerRKD:

    def __init__(self, guidedfeaturemaps, hintfeaturemaps, temperature=10, lnorm=2, distanceloss=1, angleloss=1):
        # hintfeaturemaps and guidedfeaturemaps must be lists of lengh 2
        self.temperature = temperature
        self.dim = lnorm
        self.angleloss = angleloss
        self.distanceloss = distanceloss
        self.hintfeaturemaps = hintfeaturemaps
        self.guidedfeaturemaps = guidedfeaturemaps   

    def get_module(self, x):
        if x.size(1) == 512:
          m = Conv2D_initializer(512, 1024, 1, 1)
        elif x.size(1) == 256:
          m = Conv2D_initializer(256, 256, 1, 1)
        elif x.size(1) == 128:
          m = Conv2D_initializer(128, 32, 1, 1)
        else:
          print('******* Error: No matching out_channels ******')
        return m


    def transform(self, x):
          a = x.size(1)
          b = x.size(2)
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          x = x.to(device)
          m = self.get_module(x)
          output = m(x)
          if b == 19:
            c = output.reshape([4, 256, 38, 38])
          elif b == 76:
            c = output.reshape([4, 256, 38, 38])
          elif b == 38:
            c = output.reshape([4, 256, 38, 38])
          return c


    def convert3Dto2D(self, h):
      out2D = []
      a = h.size(0)
      if a >= 255 and h.shape == torch.Size([a,19, 19]) or torch.Size([a,38, 38]) :
        t = h.permute(2, 0, 1)
        for i in range(t.size(0)):
          z = t[i, :, :]
          out2D.append(z)
        return out2D

    def convert4Dto2D(self, input, nature, rskd=False):
      output2D = []
      a = input.size(1)
      if a >= 255 and input.shape == torch.Size([4,a,19, 19]) or torch.Size([4,a,38, 38]) :
        t = input.permute(0, 2, 1, 3)
        for i in range(t.size(0)):
          z = t[i, :, :, :]
          for d in range(z.size(0)):
            if rskd and nature == 't':
              target = F.softmax(z[d,:,:]/(self.temperature), dim=1)
            elif rskd and nature == 's':
              target = F.log_softmax(z[d,:,:]/(self.temperature), dim=1)
            else:
              target = z[d,:,:] 
            output2D.append(target)
        return output2D


    def relationalpotentialdist(self, fm1, fm2, nature='s'):
        t1 = self.convert4Dto2D(self.transform(fm1), nature)
        t2 = self.convert4Dto2D(self.transform(fm2), nature)
        rel = 0
        for i in range(len(t1)):
           rel += torch.cdist(t1[i], t2[i])
        return rel

    def relationalpotentialangle(self, i, j):
        trans1 = self.transform(i)
        trans2 = self.transform(j)
        cosi = nn.CosineSimilarity(dim=0, eps=1e-6)
        fm=cosi(trans1, trans2)
        rel = self.convert3Dto2D(fm)
        return rel

    def relationalloss(self):
      a = self.hintfeaturemaps[0]
      b = self.hintfeaturemaps[1]
      c = self.guidedfeaturemaps[0]
      d = self.guidedfeaturemaps[1]

      rel_t = self.relationalpotentialdist(a, b)
      rel_s = self.relationalpotentialdist(c, d)

      relan_t = self.relationalpotentialangle(a, b)
      relan_s = self.relationalpotentialangle(c, d)

      loss1 = torch.linalg.norm( torch.sub(rel_s, rel_t) , ord=self.dim)
      loss2=0
      for k in range(len(relan_s)):
        loss2 += torch.linalg.norm( torch.sub(relan_s[k], relan_t[k]) , ord=self.dim)
      a = self.distanceloss+self.angleloss
      b = (self.distanceloss+self.angleloss)**2
      x = (self.distanceloss*loss1 + self.angleloss*loss2)*a/b # if both at same time, then evarage
      return x

#---------------------------------------------------#
#   Initialize Conv2D
#---------------------------------------------------#
class Conv2D_initializer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        if bias:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)

    def forward(self, x):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       x.to(device)
       l = self.conv
       l.to(device)
       #a = l(x)
       return l(x)



