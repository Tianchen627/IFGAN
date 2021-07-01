import numpy as np
from Model import UPGAN
from Model import generator_concat-ConvE,generator_concat-ds25,generator_concat-ds100,generator_concat-inv25,generator_concat-inv100
from functools import reduce


def init_model(args,
               user_total,
               item_total,
               entity_total,
               relation_total,
               logger,
               i_map=None,
               e_map=None,
               new_map=None,
               share_total=0):
    logger.info("Building {}.".format("Discriminator"))
 
    if args.model_name == "UPGAN":
        model = UPGAN.build_model(args, user_total, item_total, entity_total, relation_total)
    else:
        raise NotImplementedError
    logger.info("Architecture: {}".format(model))

    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in model.parameters()])
    logger.info("Total params: {}".format(total_params))

    logger.info("Building {}.".format("Generator"))
    if args.G_name == "generator_concat-ConvE":
        G = generator_concat-ConvE.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "generator_concat-ds25":
        G = generator_concat-ds25.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "generator_concat-ds100":
        G = generator_concat-ds100.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "generator_concat-inv25":
        G = generator_concat-inv25.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "generator_concat-inv100":
        G = generator_concat-inv100.build_model(args, user_total, item_total, entity_total, relation_total)
    else:
        raise NotImplementedError
    logger.info("Architecture: {}".format(G))


    return model, G