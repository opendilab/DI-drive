'''
to be removed
'''
import torch


class LocationLoss(torch.nn.Module):

    def __init__(self, w=192, h=192, choice='l2'):
        super(LocationLoss, self).__init__()

        # IMPORTANT(bradyz): loss per sample.
        if choice == 'l1':
            self.loss = lambda a, b: torch.mean(torch.abs(a - b), dim=(1, 2))
        elif choice == 'l2':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError("Unknown loss: %s" % choice)

        self.img_size = torch.FloatTensor([w, h]).cuda()

    def forward(self, pred_location, gt_location):
        '''
        Note that ground-truth location is [0,img_size]
        and pred_location is [-1,1]
        '''
        gt_location = gt_location / (0.5 * self.img_size) - 1.0

        return self.loss(pred_location, gt_location)


def normalize(x, dim):
    x_normed = x / x.max(dim, keepdim=True)[0]
    return x_normed


def weight_decay_l1(loss, model, intention_factors, alpha, gating):

    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(torch.abs(w)), wdecay)

    if intention_factors is not None:

        intention, _ = torch.min(intention_factors, 1)
        intention = (1. > intention).float()
        if gating == 'hard':
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0] / torch.sum(intention)

        elif gating == 'easy':
            wdecay = wdecay * torch.sum(intention) / intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def weight_decay_l2(loss, model, intention_factors, alpha, gating):

    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w ** 2), wdecay)

    if intention_factors is not None:

        intention, _ = torch.min(intention_factors, 1)
        intention = (1. > intention).float()
        if gating == 'hard':
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0] / torch.sum(intention)

        elif gating == 'easy':
            wdecay = wdecay * torch.sum(intention) / intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def compute_branches_masks(controls, number_targets):
    """
        Args
            controls
            the control values that have the following structure
            command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go straight
            size of targets:
            How many targets is produced by the network so we can produce the masks properly
        Returns
            a mask to have the loss function applied
            only on over the correct branch.
    """
    """ A vector with a mask for each of the control branches"""
    controls_masks = []

    # when command = 2, branch 1 (follow lane) is activated
    controls_b1 = (controls == 2)
    controls_b1 = torch.tensor(controls_b1, dtype=torch.float32).cuda()
    controls_b1 = torch.cat([controls_b1] * number_targets, 1)
    controls_masks.append(controls_b1)
    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2] * number_targets, 1)
    controls_masks.append(controls_b2)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3] * number_targets, 1)
    controls_masks.append(controls_b3)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4] * number_targets, 1)
    controls_masks.append(controls_b4)
    # when command = 6, branch 5 (go strange) is activated
    # controls_b5 = (controls == 6)
    # controls_b5 = torch.tensor(controls_b5, dtype=torch.float32).cuda()
    # controls_b5 = torch.cat([controls_b5] * number_targets, 1)
    # controls_masks.append(controls_b5)
    # # when command = 7, branch 6 (go strange) is activated
    # controls_b6 = (controls == 7)
    # controls_b6 = torch.tensor(controls_b6, dtype=torch.float32).cuda()
    # controls_b6 = torch.cat([controls_b6] * number_targets, 1)
    # controls_masks.append(controls_b6)

    return controls_masks


def l2_loss(params):
    """
        Functional LOSS L2
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) - 1):
        loss_branches_vec.append(
            ((params['branches'][i] - params['targets']) ** 2 * params['controls_mask'][i]) *
            params['branch_weights'][i]
        )
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append((params['branches'][-1] - params['inputs']) ** 2 * params['branch_weights'][-1])
    return loss_branches_vec, {}


def l1_loss(params):
    """
        Functional LOSS L1
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) - 1):
        loss_branches_vec.append(
            torch.abs((params['branches'][i] - params['targets']) * params['controls_mask'][i]) *
            params['branch_weights'][i]
        )
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append(torch.abs(params['branches'][-1] - params['inputs']) * params['branch_weights'][-1])
    return loss_branches_vec, {}


def l1(params):
    return branched_loss(l1_loss, params)


def l2(params):
    return branched_loss(l2_loss, params)


def branched_loss(loss_function, params):
    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = compute_branches_masks(params['controls'], params['branches'][0].shape[1])
    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to  have 4 branches not using speed.

    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['Brake']

    # loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
    #                 loss_branches_vec[3]+loss_branches_vec[4]+loss_branches_vec[5]

    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + loss_branches_vec[3]

    speed_loss = loss_branches_vec[-1] / (params['branches'][0].shape[0])

    return torch.sum(loss_function) / (params['branches'][0].shape[0]
                                       ) + torch.sum(speed_loss) / (params['branches'][0].shape[0]), plotable_params


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == 'L1':

        return l1

    elif loss_name == 'L2':

        return l2

    else:
        raise ValueError(" Not found Loss name")
