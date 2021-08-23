import numpy as np
import matplotlib.pyplot as plt


'''
Here defines a class that implements a simple discrete distribution 
that is used to keep track of the confidence distribution of the interested 
parameter.
'''
class Simple_Dist:
    
    # param = [(x,density)]
    # constructed left-to-right on [init[0][0],end]
    # dist pairs stored as inclusive left end
    def __init__(self, init, end):
        self.param = init
        self.start = init[0][0]
        self.end = end
        
    def display(self):
        points = []
        a_param = self.param + [(self.end,self.param[-1][1])]
        for i, v in enumerate(a_param[:-1]):
            points.append(v)
            points.append((a_param[i+1][0],v[1]))
        x,y = zip(*points)
        plt.plot(x,y)
        plt.title("Median Density")
        plt.ylabel("Density")
        plt.xlabel("Value")
    
    def cdf(self,threshold):
        basis = []
        val = []
        a_param = self.param + [(self.end,self.param[-1][1])]
        for i, v in enumerate(a_param):
            if v[0] < threshold:
                basis.append(v[0])
                val.append(v[1])
            else:
                basis.append(threshold)
                cdf = 0
                for i, d in enumerate(val):
                    cdf += d*(basis[i+1]-basis[i])
                return cdf
    
    def invcdf(self, prob):
      a_param = self.param + [(self.end,self.param[-1][1])]
      cdf = 0
      for i, v in enumerate(a_param[:-1]):
        if cdf+v[1]*(a_param[i+1][0]-a_param[i][0])>prob:
          exceed = cdf+v[1]*(a_param[i+1][0]-a_param[i][0])-prob
          return a_param[i+1][0]-exceed/v[1]
        else:
          cdf += v[1]*(a_param[i+1][0]-a_param[i][0])
      return self.end

    def median(self):
      return self.invcdf(0.5)

    '''
    update the distribution so that the interval [left, right) is updated based on vote
    and the other parts of the distribution is updated based on !vote. 
    Note: new points are added if necessary 
    (e.g. left/right doesn't currently exist in dist, thus the distribution expands as iteration increases)
    '''
    def update_dist_interval(self, left, right, vote, alpha = 0.1):

        temp = self.param+[(self.end,self.param[-1][1])]
        
        # Find index of left and right and insert new data(s) if necessary
        index1 = 0
        for i in range(len(temp)):
          if temp[i][0] > right:
            break
          index1 += 1

        if index1 < len(self.param) and self.param[index1][0] != right:
            self.param.insert(index1,(right,self.param[index1-1][1]))
        elif index1 >= len(self.param) and right != self.end:
            self.param.insert(index1,(right,self.param[index1-1][1]))

        index2 = 0
        for i in range(len(temp)):
          if temp[i][0] >= left:
            break
          index2 += 1
          
        if index2 < len(self.param) and self.param[index2][0] != left:
            self.param.insert(index2,(left,self.param[index2-1][1]))
        elif index2 >= len(self.param):
            self.param.insert(index2,(left,self.param[index2-1][1]))

            
        # Modify distribution
        if vote == -1:
            for i, v in enumerate(self.param):
                if left <= v[0] < right:
                    self.param[i] = (v[0],v[1] * alpha)
                else:
                    self.param[i] = (v[0],v[1] * (1 - alpha))

            norm = self.cdf(self.end)
            for i, v in enumerate(self.param):
                self.param[i] = (v[0], v[1]/norm)
        else:
            for i, v in enumerate(self.param):
                if left <= v[0] < right:
                    self.param[i] = (v[0],v[1] * (1 - alpha))
                else:
                    self.param[i] = (v[0],v[1] * alpha)

            norm = self.cdf(self.end)
            for i, v in enumerate(self.param):
                self.param[i] = (v[0], v[1]/norm)
            # print('end')

    '''
    update the distribution so that the interval [self.start, threshold) is updated based on vote
    and [threshold, self.end) is updated based on !vote. 
    Note: new points are added if necessary 
    (e.g. threshold doesn't currently exist in dist, thus the distribution expands as iteration increases)
    '''
    def update_dist(self, threshold, vote, alpha = 0.1):
          
          # Find index of threshold and insert new data(s) if necessary
          index = next(i for i,v in enumerate(self.param+[(self.end,self.param[-1][1])]) if threshold < v[0])
          if index < len(self.param) and self.param[index][0] != threshold:
              self.param.insert(index,(threshold,self.param[index-1][1]))
          elif index >= len(self.param):
              self.param.insert(index,(threshold,self.param[index-1][1]))
              
          # Modify distribution
          if vote == -1:
              for i, v in enumerate(self.param):
                  if v[0] < threshold:
                      self.param[i] = (v[0],v[1] * (1 - alpha))
                  else:
                      self.param[i] = (v[0],v[1] * alpha)
              norm = self.cdf(self.end)
              for i, v in enumerate(self.param):
                  self.param[i] = (v[0], v[1]/norm)
          else:
              for i, v in enumerate(self.param):
                  if v[0] >= threshold:
                      self.param[i] = (v[0],v[1] * (1 - alpha))
                  else:
                      self.param[i] = (v[0],v[1] * alpha)
              norm = self.cdf(self.end)
              for i, v in enumerate(self.param):
                  self.param[i] = (v[0], v[1]/norm)