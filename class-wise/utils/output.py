import time
import datetime


def format_number(int_digit, float_digit, number):#格式化数字，整数保留int_digit位，小数保留float_digit位
    
    if isinstance(number, float):
        int_part = int(number)
        float_part = number - int_part
        return "{{:>{}d}}.{{:>0{}d}}".format(int_digit, float_digit).format( int_part, int(float_part * 10 ** float_digit))

    return number
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name:str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class LogProcessBar():
    def __init__(self, logfile, args):

        self.logfile = logfile
        with open(self.logfile, 'a') as f:
            for key in vars(args):
                f.write(key.ljust(15) + ' : ' + str(vars(args)[key]) + '\n')

    def log(self, msg):
        with open(self.logfile, 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + msg + '\n')
    def log_print(self, msg):
        self.log(msg)
        print(msg)

    def refresh(self, current, total, msg=None, fresh=True):
        L = []
        L.append("[{:>3d}/{:<3d}]".format(current + 1, total))

        if msg:
            L.append('-' + msg)
        msg = ''.join(L)
        if current < total - 1:
            if fresh: print('\r', msg, end='')
        elif current == total - 1:
            print('\r', msg)
            with open(self.logfile, 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\t'+msg+'\n')
        else:
            raise NotImplementedError






