import os, time

files = ['input/blur.A.bmp input/blur.Ap.bmp input/blur.B.bmp output/blur.bmp',
         'input/nasts.A.jpg input/nasts.Ap.jpg input/girl.B.jpg output/girl.bmp']

for f in files:
 for coh in [0, 0.25, 0.5, 0.75, 1]:
  for alg in ['-multi', '-ann', '-brute', '-ann']:
   for neigh in [5]:
    
    t0 = time.time()
    cmd = './src/analogy %s -neighborhood_size %s %s -coherence %s' % (f, neigh, alg, coh)
    print cmd
    os.system(cmd)
    t1 = time.time()
    print 'elapsed time: %s s' % (t1-t0)
