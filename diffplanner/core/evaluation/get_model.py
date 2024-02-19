from diffplanner.models import build_submodule


def get_motion_model(name, ckpt_path):
    if name == 'kit_ml':
        model = build_submodule(dict(
            type='T2MMotionEncoder',
            input_size=251,
            movement_hidden_size=512,
            movement_latent_size=512,
            motion_hidden_size=1024,
            motion_latent_size=512,
        ))
    elif name == 'kit_ttc' and 'human' not in ckpt_path:
        model = build_submodule(dict(
            type='ACTOREncoder',
            max_seq_len=196,
            input_feats=247,
            latent_dim=512,
            num_layers=4,
            ff_size=2048,
            output_var=True
        ))
    elif name == 'kit_ttc':
        model = build_submodule(dict(
            type='ACTOREncoder',
            max_seq_len=196,
            input_feats=133,
            latent_dim=512,
            num_layers=4,
            ff_size=2048,
            output_var=True
        ))
    else:
        model = build_submodule(dict(
            type='T2MMotionEncoder',
            input_size=263,
            movement_hidden_size=512,
            movement_latent_size=512,
            motion_hidden_size=1024,
            motion_latent_size=512,
        ))
    model.load_pretrained(ckpt_path)
    return model

def get_text_model(name, ckpt_path):
    if name == 'kit_ml':
        model = build_submodule(dict(
            type='T2MTextEncoder',
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
            max_text_len=20
        ))
    elif name == 'kit_ttc' and 'human' not in ckpt_path:
        model = build_submodule(dict(
            type='TextEncoder',
            pretrained_model='clip',
            text_latent_dim=256,
            time_embed_dim=512,
            dropout=0,
            num_text_layers=2,
            text_num_heads=4,
            text_ff_size=2048,
            use_text_proj=True
        ))
    elif name == 'kit_ttc':
        model = build_submodule(dict(
            type='TextEncoder',
            pretrained_model='clip',
            text_latent_dim=256,
            time_embed_dim=512,
            dropout=0,
            num_text_layers=4,
            text_num_heads=4,
            text_ff_size=2048,
            use_text_proj=True
        ))
    else:
        model = build_submodule(dict(
            type='T2MTextEncoder',
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
            max_text_len=20
        ))
    model.load_pretrained(ckpt_path)
    return model
