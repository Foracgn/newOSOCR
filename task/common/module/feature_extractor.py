def getFEConfig(hardness, nChannel=512, ich=1, strides=None, shape=None, expf=1):
    if strides is None:
        strides = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)]
    if shape is None:
        shape = [ich, 32, 128]
    else:
        shape = [ich, shape[0], shape[1]]
    return {
        'expf': expf,
        'strides': strides,
        'compressLayer': False,
        'shape': shape,
        'hardness': hardness,
        'oupch': nChannel,
    }