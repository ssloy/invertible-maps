import numpy as np
from PIL import Image, ImageDraw

class Mesh():
    def __init__(self, test_case): # generate the test problem
        n = self.size
        self.quads = [ [i+j*n, i+1+j*n, i+1+(j+1)*n, i+(j+1)*n] for j in range(n-1) for i in range(n-1) ] # connectivity
        self.boundary = [ i==0 or i==n-1 or j==0 or j==n-1 for j in range(n) for i in range(n) ] # vertex boundary flags

        self.x = np.array([ i/n for j in range(n) for i in range(n) ] + \
                          [ j/n for j in range(n) for i in range(n) ])  # regular grid

        if (test_case=="z"): # Belinsky Z test case
            self.x = np.array([   i/n + int(j>=n//2)*3/5 for j in range(n) for i in range(n) ] + \
                              [ 2*j/n - int(j>=n//2)*3/5 for j in range(n) for i in range(n) ])  # 2D geometry
        elif (test_case=="chicane"): # chicane test case
            self.x = np.array([   i/n + int(j>=n//2)*3/5 for j in range(n) for i in range(n) ] + \
                              [ 2*j/n                    for j in range(n) for i in range(n) ])  # 2D geometry
        elif (test_case=="disc"): # disc test case
            blist = [i for i in range(n)] + [(i+2)*n-1 for i in range(n-2)] + [n*n - 1 - i for i in range(n)] + [n*(n-1) - (i+1)*n for i in range(n-2)]
            for i,v in enumerate(blist):
                self.x[v    ] = np.cos(i/len(blist)*2.*np.pi+np.pi/4.)
                self.x[v+n*n] = np.sin(i/len(blist)*2.*np.pi+np.pi/4.)

    @property
    def size(self):
        return 8

    @property
    def nverts(self):
        return self.size*self.size

    def __str__(self): # wavefront .obj output
        ret = ""
        for v in range(self.nverts):
            ret = ret + ("v %f %f 0\n" % (self.x[v], self.x[v+self.nverts]))
        for f in self.quads:
            ret = ret + ("f %d %d %d %d\n" % (f[0]+1, f[1]+1, f[2]+1, f[3]+1))
        return ret
    def show(self):
        res = 1000
        off = 100
        image = Image.new(mode='L', size=(res, res), color=255)
        draw = ImageDraw.Draw(image)

        for quad in self.quads:
            for e in range(4):
                i = quad[e]
                j = quad[(e+1)%4]

                line = ((off+self.x[i]*res/2, off+self.x[i+self.nverts]*res/2), (off+self.x[j]*res/2, off+self.x[j+self.nverts]*res/2))
                draw.line(line, fill=128)
        del draw
#        image.save("winslow.png")
        image.show()
