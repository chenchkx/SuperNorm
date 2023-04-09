
import math
from torch.optim.lr_scheduler import LambdaLR

class LR_Scheduler(LambdaLR):
    def __init__(self, optimizer, total_steps, warmup_type, 
                 warmup_steps=50, step_size=50, gamma=0.5, init_rate=0.0, last_epoch=-1):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_type = warmup_type
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        self.init_rate = init_rate
        self.last_epoch = last_epoch
        super(LR_Scheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_warmup(self, step):
        lr_rate = (float(step) / float(max(1, self.warmup_steps)))*(1-self.init_rate) + self.init_rate
        return lr_rate
    
    def lr_decay(self, step):
        if 'cosine' in self.warmup_type:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_rate = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
        elif 'linear' in self.warmup_type:
            lr_rate = max(0.0, float(self.total_steps - step) / float(max(1.0, self.total_steps - self.warmup_steps)))
        elif 'step' in self.warmup_type:
            lr_rate = self.gamma ** int((step - self.warmup_steps)/self.step_size)
        elif 'None' in self.warmup_type:
            lr_rate = 1.0
        return lr_rate

    def lr_lambda(self, step):
        # if self.warmup_type == 'None':
        #     # return self.gamma ** int(step/self.step_size)
        #     return 1.0
        if step < self.warmup_steps:
            return self.lr_warmup(step)
        return self.lr_decay(step)
        
        # if 'cosine' in self.warmup_type:
        #     if step < self.warmup_steps:
        #         return self.lr_warmup(step, self.warmup_steps)
        #     progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        #     return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
        # elif 'linear' in self.warmup_type:
        #     if step < self.warmup_steps:
        #         return self.lr_warmup(step, self.warmup_steps)
        #     return max(0.0, float(self.total_steps - step) / float(max(1.0, self.total_steps - self.warmup_steps)))
        # elif 'step' in self.warmup_type:
        #     if step < self.warmup_steps:
        #         return self.lr_warmup(step, self.warmup_steps)
        #     return self.gamma ** int((step - self.warmup_steps)/self.step_size)
        # else: 
        #     return self.gamma ** int(step/self.step_size)