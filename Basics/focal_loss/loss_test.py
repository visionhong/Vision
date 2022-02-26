from __future__ import print_function

import random, time

from focal_loss.focalloss import *


start_time = time.time()
maxe = 0
for i in range(3):
    x = torch.rand(3, 2) * random.randint(1,10)
    x = Variable(x.cuda())
    l = torch.rand(3).ge(0.1).long()  # 0.1보다 작으면 0 크면 1
    l = Variable(l.cuda())

    output0 = FocalLoss(gamma=2, alpha=0)(x,l)
    output1 = nn.CrossEntropyLoss()(x,l)
    a = output0.item()
    b = output1.item()
    print('a',a)
    print('b',b)
    print()
    if abs(a-b) > maxe: maxe = abs(a-b)
print('time:',time.time() - start_time, 'max_error:',maxe)