import numpy as np


class Augmenter(object):
    """
    This class serve as a wrapper to apply augmentations from IMGAUG in CPU mode in
    the same way augmentations are applied when using the transform library from pytorch

    """

    # Here besides just applying the list, the class should also apply the scheduling

    def __init__(self, scheduler_strategy):
        if scheduler_strategy is not None and scheduler_strategy != 'None':
            self.scheduler = getattr(input.scheduler, scheduler_strategy)
        else:
            self.scheduler = None

    def __call__(self, iteration, img):
        #TODO: Check this format issue

        # THe scheduler receives an iteration number and returns a transformation, vec

        #print (img.shape)
        if self.scheduler is not None:
            #print (self.scheduler, iteration)
            t = self.scheduler(iteration)
            #print (t)
            img = t.augment_image(img)

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.scheduler:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
