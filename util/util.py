import sys, math
import torch.nn.init as init


def adjust_learning_rate(optimizer, lr):
    """
    checked
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_weight_(weight):
    """
    checked
    Initialize a weighting tensor
    """
    init.xavier_normal_(weight)


def init_embedding_(input_embedding):
    """
    checked
    Initialize embedding
    """
    init_weight_(input_embedding.weight)


def init_linear_(input_linear):
    """
    checked
    Initialize linear transformation
    """
    init_weight_(input_linear.weight)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm_(input_lstm):
    """
    checked
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        init_weight_(weight)

        weight = eval('input_lstm.weight_hh_l' + str(ind))
        init_weight_(weight)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            init_weight_(weight)

            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            init_weight_(weight)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_cnn_(input_cnn):
    """
    checked
    Initialize cnn
    """
    init_weight_(input_cnn.weight)
    if input_cnn.bias is not None:
        input_cnn.bias.data.zero_()

def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)

def progressbar(cur, total, other_information):
    percent = '{:.2%}'.format(cur / total)
    if type(other_information) is str:
        log("\r[%-50s] %s %s" % ('=' * int(math.floor(cur * 50 / total)), percent, other_information))
    else:
        log("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent))

