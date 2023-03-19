def getPEConfig(metaPath, valFrac=0.8):
    return {
        "meta_path": metaPath,
        "num_channel": 512,
        "case_sensitive": False,
        "val_frac": valFrac
    }