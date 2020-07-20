def get_same_padding(size, kernel_size, stride):
    def check(clen, cpad):
        return (clen + 2 * cpad - (kernel_size - 1) - 1) // stride + 1

    h_pad = 0
    while check(size[0], h_pad) != size[0] / stride:
        h_pad += 1

    w_pad = 0
    while check(size[1], w_pad) != size[1] / stride:
        w_pad += 1

    return h_pad, w_pad


def get_same_padding_transpose(size, kernel_size, stride):
    def check(clen, cpad):
        return (clen - 1) * stride - 2 * cpad + (kernel_size - 1) + 1

    h_pad = 0
    while check(size[0], h_pad) > size[0] * stride:
        h_pad += 1

    w_pad = 0
    while check(size[1], w_pad) > size[1] * stride:
        w_pad += 1

    return (h_pad, w_pad), size[0] * stride - check(size[0], h_pad)
