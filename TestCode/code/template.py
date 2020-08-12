def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('MSRN') >= 0:
        args.model = 'MSRN'
        args.n_blocks = 8
        args.n_feats = 64
        args.chop = True

    if args.template.find('MLEFGN') >= 0:
        args.model = 'MLEFGN'
        args.block = 5
        args.n_layer = 8
        args.n_feats = 64
        args.kernel_size = 3
        args.rate = 64
        args.in_dim = 64
        args.chop = True



